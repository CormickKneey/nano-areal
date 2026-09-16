"""
nano-archon trainer: GRPO + FSDP2 + torch.compile.

On single GPU (or macOS sync mode), FSDP is skipped and the model runs
directly on the target device.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nano_areal.archon.distill import (
    DistillConfig,
    TeacherScorer,
    distill_loss,
    groups_to_distill,
    teacher_logprob_tensor,
)
from nano_areal.archon.lora import (
    LoRAConfig,
    apply_lora,
    count_trainable,
    merge_lora_state_dict,
)
from nano_areal.archon.losses import (
    RejectionSamplingConfig,
    group_normalize_advantages,
    kl_penalty,
    policy_loss,
)

if TYPE_CHECKING:
    from nano_areal.archon.model import Qwen35Model
    from nano_areal.config import TrainConfig


# ---------------------------------------------------------------------------
# Trajectory data structure (filled by rollout workers)
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    # Tokenized full sequence: [prompt tokens] + [response tokens]
    input_ids: Tensor           # [T]
    # 1 for response tokens (used in loss), 0 for prompt tokens
    response_mask: Tensor       # [T]
    # Scalar reward for this trajectory
    reward: float
    # Policy version when this trajectory was generated (for off-policyness check)
    policy_version: int = 0
    # Optional: raw conversation turns for visualization
    turns: list[dict] = field(default_factory=list)
    # Per-turn rewards (before discount)
    turn_rewards: list[float] = field(default_factory=list)
    # GRPO group (all episodes sampled for the same prompt share it)
    group_id: str = ""
    # Behavior-policy log probs from the inference engine, aligned with
    # input_ids ([T], 0 at prompt positions). None → assume on-policy.
    behavior_logprobs: Tensor | None = None
    # Inference-engine finish reason; "length" marks a truncated response
    stop_reason: str = "stop"
    # Multimodal prompt inputs (None for text-only)
    pixel_values: Tensor | None = None      # [N_patches, D]
    image_grid_thw: Tensor | None = None    # [N_img, 3]

    @property
    def is_truncated(self) -> bool:
        return self.stop_reason == "length"


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    log_probs: Tensor,                    # [B, T] π_θ on response tokens (grad)
    ref_log_probs: Tensor,                # [B, T] π_ref (frozen reference)
    rewards: Tensor,                      # [B]
    masks: Tensor,                        # [B, T] response token mask
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    *,
    group_ids: list[str] | None = None,
    proximal_log_probs: Tensor | None = None,  # [B, T] π_prox; None → π_θ.detach()
    behavior_log_probs: Tensor | None = None,   # [B, T] π_behave from inference
    truncated: Tensor | None = None,            # [B] bool
    loss_type: str = "grpo",
    adv_norm: str = "grouped",
    kl_type: str = "k1",
    clip_eps_high: float | None = None,
    rejection: RejectionSamplingConfig | None = None,
    mask_truncated: bool = False,
) -> tuple[Tensor, dict]:
    """
    Policy-gradient objective + KL penalty.

      advantages  = group-normalized rewards (zero for incomplete groups)
      pg          = policy_loss(π_θ, π_prox, adv)  — grpo / dr_grpo / cispo / gspo
      loss        = pg + kl_coef * KL(π_θ || π_ref)

    Without `group_ids` the whole batch is treated as one group.
    """
    B = rewards.shape[0]
    if group_ids is None:
        group_ids = ["_batch"] * B

    advantages, adv_stats = group_normalize_advantages(
        rewards, group_ids, norm=adv_norm
    )

    n_truncated = 0
    if truncated is not None:
        n_truncated = int(truncated.sum())
        if mask_truncated:
            advantages = torch.where(
                truncated, torch.zeros_like(advantages), advantages
            )

    if proximal_log_probs is None:
        proximal_log_probs = log_probs.detach()

    pg, pg_stats = policy_loss(
        logprobs=log_probs,
        proximal_logprobs=proximal_log_probs,
        advantages=advantages,
        loss_mask=masks,
        loss_type=loss_type,
        clip_eps=clip_eps,
        clip_eps_high=clip_eps_high,
        behavior_logprobs=behavior_log_probs,
        rejection=rejection,
    )
    kl, kl_mean = kl_penalty(log_probs, ref_log_probs, masks, kl_type=kl_type)
    loss = pg + kl_coef * kl.sum() / masks.sum().clamp(min=1)

    stats = {
        "loss": loss.item(),
        "pg_loss": float(pg_stats["pg_loss"]),
        "kl": kl_mean.item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item() if B > 1 else 0.0,
        "adv_abs_mean": advantages.abs().mean().item(),
        "ratio_mean": float(pg_stats["ratio_mean"]),
        "ratio_max": float(pg_stats["ratio_max"]),
        "clip_frac": float(pg_stats["clip_frac"]),
        "approx_kl_prox": float(pg_stats["approx_kl_prox"]),
        "rejected_frac": float(pg_stats["rejected_frac"]),
        "truncated_frac": n_truncated / max(B, 1),
        **adv_stats,
    }
    if "behave_imp_weight_mean" in pg_stats:
        stats["behave_imp_weight_mean"] = float(pg_stats["behave_imp_weight_mean"])
    return loss, stats


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ArchonTrainer:
    def __init__(
        self,
        model: "Qwen35Model",
        config: "TrainConfig",
        tokenizer,
        teacher: TeacherScorer | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.policy_version = 0

        if config.gpu_ids and torch.cuda.is_available():
            device_str = f"cuda:{config.gpu_ids[0]}"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.dtype = _resolve_dtype(config.dtype, self.device)

        # Frozen reference model: a copy of the *base* weights (before LoRA),
        # not wrapped — always a single copy.
        self.ref_model = copy.deepcopy(model).to(self.device, dtype=self.dtype)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

        # Optional LoRA: freeze base weights, train adapters only
        self.lora: LoRAConfig | None = None
        if config.lora_rank > 0:
            self.lora = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                targets=tuple(config.lora_targets),
                freeze_vision=True,
            )
            apply_lora(model, self.lora)
        elif config.freeze_vision and model.visual is not None:
            model.visual.requires_grad_(False)

        model.set_gradient_checkpointing(config.gradient_checkpointing)

        # Move model to device
        self.model = model.to(self.device, dtype=self.dtype)

        # FSDP2 wrapping (multi-GPU) or plain module (single GPU / macOS)
        self.model = self._maybe_wrap_fsdp(self.model)

        # torch.compile
        if config.compile:
            self.model = torch.compile(self.model, backend=config.compile_backend)
            self.ref_model = torch.compile(self.ref_model, backend=config.compile_backend)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.n_trainable, self.n_total = count_trainable(model)
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.lr,
            weight_decay=config.weight_decay,
            fused=(self.device.type == "cuda"),  # fused only on CUDA
        )

        # Teacher distillation (AReaL mopd): off unless distill_coef > 0
        self.teacher = teacher
        self.distill = DistillConfig(
            rl_coef=config.rl_coef,
            distill_coef=config.distill_coef if teacher is not None else 0.0,
            trigger_max_pass_rate=config.distill_trigger_max_pass_rate,
        )

        self._step = 0
        self._accum_stats: list[dict] = []

    # ------------------------------------------------------------------
    # FSDP2 setup
    # ------------------------------------------------------------------

    def _maybe_wrap_fsdp(self, model: nn.Module) -> nn.Module:
        if len(self.config.gpu_ids) <= 1:
            return model  # single GPU: no FSDP needed

        try:
            import functools

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            from nano_areal.archon.model import DecoderLayer

            mp = MixedPrecision(
                param_dtype=self.dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=self.dtype,
            )
            auto_wrap = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={DecoderLayer},
            )
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp,
                auto_wrap_policy=auto_wrap,
                device_id=self.device,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"FSDP setup failed ({e}), falling back to DDP/plain")

        return model

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize_trajectory(self, traj: Trajectory) -> tuple[Tensor, Tensor]:
        """If trajectory already has input_ids, return them; otherwise tokenize turns."""
        if traj.input_ids is not None and traj.input_ids.numel() > 0:
            return traj.input_ids.to(self.device), traj.response_mask.to(self.device)

        # Fallback: concatenate turns and mark last assistant turn as response
        full_text = self.tokenizer.apply_chat_template(
            traj.turns, tokenize=False, add_generation_prompt=False
        )
        ids = self.tokenizer(full_text, return_tensors="pt").input_ids[0]
        mask = torch.zeros_like(ids)
        return ids.to(self.device), mask.to(self.device)

    def _pad_batch(
        self,
        sequences: list[Tensor],
        masks: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        max_len = max(s.shape[0] for s in sequences)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_ids, padded_masks = [], []
        for ids, mask in zip(sequences, masks):
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (0, pad_len), value=pad_id))
            padded_masks.append(F.pad(mask, (0, pad_len), value=0))

        return torch.stack(padded_ids), torch.stack(padded_masks)

    # ------------------------------------------------------------------
    # Log-prob computation (with optional recompute for off-policy data)
    # ------------------------------------------------------------------

    def _collate_images(
        self, trajectories: list[Trajectory]
    ) -> tuple[Tensor | None, Tensor | None]:
        """Concatenate per-trajectory pixel patches in batch order (image tokens are
        consumed sequentially over the flattened batch, so order == batch order)."""
        pv = [t.pixel_values for t in trajectories if t.pixel_values is not None and t.pixel_values.numel() > 0]
        if not pv:
            return None, None
        grid = [t.image_grid_thw for t in trajectories if t.image_grid_thw is not None]
        return (
            torch.cat(pv).to(self.device, dtype=self.dtype),
            torch.cat(grid).to(self.device),
        )

    @torch.no_grad()
    def _compute_ref_log_probs(
        self,
        input_ids: Tensor,
        response_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> Tensor:
        return self.ref_model.compute_log_probs(
            input_ids, response_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )

    def _compute_policy_log_probs(
        self,
        input_ids: Tensor,
        response_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> Tensor:
        return self.model.compute_log_probs(
            input_ids, response_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )

    def _teacher_log_probs(
        self,
        trajectories: list[Trajectory],
        seqs: list[Tensor],
        rewards: Tensor,
        group_ids: list[str],
        T: int,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Teacher logprobs [B, T-1] and a row mask [B] for groups selected by the trigger."""
        if self.teacher is None or not self.distill.enabled:
            return None, None
        selected = groups_to_distill(rewards.tolist(), group_ids, self.distill)
        rows = torch.tensor([g in selected for g in group_ids], device=self.device)
        if not bool(rows.any()):
            return None, None
        out = torch.zeros(len(trajectories), T - 1, device=self.device)
        for i, (traj, ids) in enumerate(zip(trajectories, seqs)):
            if not rows[i]:
                continue
            out[i] = teacher_logprob_tensor(
                self.teacher, ids.cpu(), traj.pixel_values, traj.image_grid_thw, T, self.distill.clamp_teacher_min
            ).to(self.device)
        return out, rows

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, trajectories: list[Trajectory]) -> dict:
        self.model.train()

        # Tokenize & pad
        seqs, masks = [], []
        for traj in trajectories:
            ids, mask = self._tokenize_trajectory(traj)
            seqs.append(ids)
            masks.append(mask)

        input_ids, response_mask = self._pad_batch(seqs, masks)
        pixel_values, image_grid_thw = self._collate_images(trajectories)
        rewards = torch.tensor(
            [t.reward for t in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        truncated = torch.tensor(
            [t.is_truncated for t in trajectories],
            dtype=torch.bool,
            device=self.device,
        )
        group_ids = [t.group_id or str(i) for i, t in enumerate(trajectories)]

        # Behavior log probs (from vllm), shifted to align with next-token prediction
        behavior_log_probs = None
        if all(t.behavior_logprobs is not None for t in trajectories):
            T = input_ids.shape[1]
            behavior_log_probs = torch.stack([
                F.pad(t.behavior_logprobs.float(), (0, T - len(t.behavior_logprobs)))
                for t in trajectories
            ]).to(self.device)[:, 1:]

        with torch.no_grad():
            ref_log_probs = self._compute_ref_log_probs(
                input_ids, response_mask, pixel_values, image_grid_thw
            )

        log_probs = self._compute_policy_log_probs(
            input_ids, response_mask, pixel_values, image_grid_thw
        )

        # Proximal policy: with one mini-batch per update π_prox == π_θ at this
        # point, so reuse the training forward instead of a second pass.
        proximal_log_probs = None
        if not self.config.reuse_train_logp:
            with torch.no_grad():
                proximal_log_probs = self._compute_policy_log_probs(
                    input_ids, response_mask, pixel_values, image_grid_thw
                )

        rejection = None
        if self.config.behav_imp_weight_cap is not None:
            rejection = RejectionSamplingConfig(
                action=self.config.rejection_action,
                upper=self.config.behav_imp_weight_cap,
            )

        loss, stats = grpo_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            masks=response_mask[:, 1:],
            clip_eps=self.config.clip_eps,
            kl_coef=self.config.kl_coef,
            group_ids=group_ids,
            proximal_log_probs=proximal_log_probs,
            behavior_log_probs=behavior_log_probs,
            truncated=truncated,
            loss_type=self.config.loss_type,
            adv_norm=self.config.adv_norm,
            kl_type=self.config.kl_type,
            clip_eps_high=self.config.clip_eps_high,
            rejection=rejection,
            mask_truncated=self.config.mask_truncated,
        )

        # Teacher distillation on the student's own tokens (mopd)
        teacher_lp, distill_rows = self._teacher_log_probs(
            trajectories, seqs, rewards, group_ids, input_ids.shape[1]
        )
        if teacher_lp is not None:
            d_mask = response_mask[:, 1:] * distill_rows.float().unsqueeze(-1)
            d_loss, d_stats = distill_loss(log_probs, teacher_lp, d_mask)
            loss = self.distill.rl_coef * loss + self.distill.distill_coef * d_loss
            stats.update(d_stats)
            stats["loss"] = loss.item()
        else:
            stats["distill_loss"] = 0.0
            stats["distill_frac"] = 0.0
        stats["n_images"] = 0 if image_grid_thw is None else int(image_grid_thw.shape[0])

        # Gradient accumulation
        loss = loss / self.config.grad_accum_steps
        loss.backward()

        self._accum_stats.append(stats)
        self._step += 1

        if self._step % self.config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.config.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.policy_version += 1

        return {**stats, "policy_version": self.policy_version, "step": self._step}

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        # Unwrap compile / FSDP before saving
        raw = getattr(self.model, "_orig_mod", self.model)
        if hasattr(raw, "module"):
            raw = raw.module
        torch.save(raw.state_dict(), f"{path}/model.pt")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")

    def load(self, path: str):
        raw = getattr(self.model, "_orig_mod", self.model)
        if hasattr(raw, "module"):
            raw = raw.module
        raw.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))
        self.optimizer.load_state_dict(
            torch.load(f"{path}/optimizer.pt", map_location=self.device)
        )

    def _raw_model(self) -> nn.Module:
        raw = getattr(self.model, "_orig_mod", self.model)
        if hasattr(raw, "module"):
            raw = raw.module
        return raw

    def get_state_dict(self) -> dict:
        """Current policy weights for weight sync. LoRA adapters are merged into
        the base weights so the inference engine loads plain dense tensors."""
        raw = self._raw_model()
        if self.lora is not None:
            return {k: v.cpu() for k, v in merge_lora_state_dict(raw).items()}
        return {k: v.cpu() for k, v in raw.state_dict().items()}


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
