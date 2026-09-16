"""
On-policy token-level distillation (AReaL "mopd" style).

The teacher scores the *student's own* sampled tokens: for every response
token we request log p_teacher(token_t | prefix) (vLLM `prompt_logprobs`) and
push the student towards the teacher with a reverse-KL-like per-token term

    L_distill = mean_t [ logp_student_t - logp_teacher_t ]      (k1 estimator)

which combines with the RL objective as `rl_coef * L_rl + distill_coef * L_distill`.
Student and teacher must share a tokenizer (Qwen3.5-2B / Qwen3.8-27B do).

Teacher backends:
  * LocalArchonTeacher  - another nano-archon model in-process (CPU tests, tiny models)
  * RemoteTeacher       - OpenAI-compatible /v1/completions with prompt_logprobs
                          (vLLM serving Qwen3.8-27B-Int4, or the local server)
  * ConstantTeacher     - test double
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class TeacherScorer(Protocol):
    def score(
        self,
        input_ids: list[int],
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> list[float]:
        """log p_teacher(token_t | tokens_<t) for t >= 1 → length len(input_ids) - 1."""
        ...


class LocalArchonTeacher:
    def __init__(self, model):
        from nano_areal.archon.generate import Generator
        self.gen = Generator(model.eval())

    def score(self, input_ids, pixel_values=None, image_grid_thw=None) -> list[float]:
        return self.gen.prompt_logprobs(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)


class RemoteTeacher:
    """vLLM-compatible `/v1/completions` with `prompt_logprobs=0`; prompt sent as token ids."""

    def __init__(self, base_url: str, model: str = "default", api_key: str = "EMPTY", timeout: float = 300.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def score(self, input_ids, pixel_values=None, image_grid_thw=None) -> list[float]:
        r = self._http.post(
            f"{self.base_url}/completions",
            json={"model": self.model, "prompt": list(input_ids), "max_tokens": 0, "prompt_logprobs": 0},
        )
        r.raise_for_status()
        entries = r.json()["choices"][0]["prompt_logprobs"]
        out: list[float] = []
        for tok, entry in zip(input_ids[1:], entries[1:]):
            if entry is None:
                out.append(0.0)
                continue
            info = entry.get(str(tok))
            out.append(float(info["logprob"]) if info else float("-inf"))
        return out


class ConstantTeacher:
    def __init__(self, value: float = -1.0):
        self.value = value

    def score(self, input_ids, pixel_values=None, image_grid_thw=None) -> list[float]:
        return [self.value] * (len(input_ids) - 1)


# ---------------------------------------------------------------------------
# Triggering + loss
# ---------------------------------------------------------------------------

@dataclass
class DistillConfig:
    rl_coef: float = 1.0
    distill_coef: float = 0.0            # 0 disables the teacher entirely
    # Only query the teacher for groups whose pass-rate is below this threshold
    # (AReaL mopd: distill where the student fails, let RL handle what it solves)
    trigger_max_pass_rate: float = 1.0
    success_threshold: float = 0.5       # reward > threshold counts as "solved"
    clamp_teacher_min: float = -20.0     # floor teacher logprobs before use

    @property
    def enabled(self) -> bool:
        return self.distill_coef > 0.0


def groups_to_distill(rewards: list[float], group_ids: list[str], cfg: DistillConfig) -> set[str]:
    """Group ids whose pass rate <= trigger threshold."""
    per: dict[str, list[float]] = {}
    for r, g in zip(rewards, group_ids):
        per.setdefault(g, []).append(r)
    out = set()
    for g, rs in per.items():
        rate = sum(1 for r in rs if r > cfg.success_threshold) / len(rs)
        if rate <= cfg.trigger_max_pass_rate:
            out.add(g)
    return out


def teacher_logprob_tensor(
    teacher: TeacherScorer,
    input_ids: Tensor,          # [T] (unpadded)
    pixel_values: Tensor | None,
    image_grid_thw: Tensor | None,
    T_pad: int,
    clamp_min: float,
) -> Tensor:
    """Teacher logprobs aligned with the trainer's shifted layout: [T_pad-1]."""
    lps = teacher.score(input_ids.tolist(), pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    t = torch.tensor(lps, dtype=torch.float32).clamp(min=clamp_min)
    return F.pad(t, (0, (T_pad - 1) - t.shape[0]))


def distill_loss(
    student_logprobs: Tensor,   # [B, T-1] with grad
    teacher_logprobs: Tensor,   # [B, T-1]
    loss_mask: Tensor,          # [B, T-1] 1 on response tokens of distilled rows
) -> tuple[Tensor, dict]:
    """
    Per-token k1 reverse-KL estimator on the sampled tokens,
    E_{x~student}[log p_s(x) - log p_t(x)]; gradient flows through log p_s only.
    """
    diff = student_logprobs - teacher_logprobs
    denom = loss_mask.sum().clamp(min=1)
    loss = (diff * loss_mask).sum() / denom
    return loss, {
        "distill_loss": loss.item(),
        "teacher_logp_mean": ((teacher_logprobs * loss_mask).sum() / denom).item(),
        "distill_frac": (loss_mask.sum(-1) > 0).float().mean().item(),
    }
