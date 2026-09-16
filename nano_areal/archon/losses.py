"""
RL loss building blocks for nano-areal (mirrors AReaL areal/utils/functional).

Three log-prob streams are distinguished (AReaL decoupled PPO, #1453):

  behavior_logprobs  π_behave    logprobs returned by the inference engine
                                 when the trajectory was sampled (may be stale)
  proximal_logprobs  π_prox      recomputed by the trainer at the start of the
                                 update; the anchor for PPO clipping
  logprobs           π_θ         current policy, carries the gradient
  ref_logprobs       π_ref       frozen reference model, only used for the KL
                                 penalty

With a single mini-batch per update π_prox == π_θ.detach(), so the proximal
forward can be reused from the training forward (``reuse_train_logp``).
π_prox / π_behave is the staleness importance weight used for rejection
sampling (#1572).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

LossType = Literal["grpo", "dr_grpo", "cispo", "gspo"]
AdvNorm = Literal["grouped", "batch", "none"]
KLType = Literal["k1", "k3"]


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------


def group_normalize_advantages(
    rewards: Tensor,  # [B]
    group_ids: list[str],  # len B
    norm: AdvNorm = "grouped",
    std_normalize: bool = True,
    eps: float = 1e-6,
) -> tuple[Tensor, dict]:
    """
    GRPO advantage: reward centered (and optionally scaled) within each
    prompt group (AReaL #1619).

    Groups that are incomplete in this batch (size 1) or degenerate
    (all rewards equal) get zero advantage instead of a noisy one (#1563).
    Their tokens still take part in the KL term.
    """
    B = rewards.shape[0]
    if norm == "none":
        return rewards.detach(), {
            "n_groups": B,
            "n_incomplete_groups": 0,
            "n_zero_var_groups": 0,
        }
    if norm == "batch":
        std = (
            rewards.std()
            if std_normalize and B > 1
            else torch.ones((), device=rewards.device)
        )
        adv = (
            (rewards - rewards.mean()) / (std + eps)
            if B > 1
            else torch.zeros_like(rewards)
        )
        return adv.detach(), {
            "n_groups": 1,
            "n_incomplete_groups": int(B <= 1),
            "n_zero_var_groups": 0,
        }

    adv = torch.zeros_like(rewards)
    index: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        index.setdefault(g, []).append(i)

    n_incomplete = n_zero_var = 0
    for members in index.values():
        if len(members) < 2:
            n_incomplete += 1
            continue
        idx = torch.tensor(members, device=rewards.device)
        r = rewards[idx]
        centered = r - r.mean()
        std = r.std()
        if std < eps:
            n_zero_var += 1
            continue
        adv[idx] = centered / (std + eps) if std_normalize else centered

    stats = {
        "n_groups": len(index),
        "n_incomplete_groups": n_incomplete,
        "n_zero_var_groups": n_zero_var,
    }
    return adv.detach(), stats


# ---------------------------------------------------------------------------
# Rejection sampling on staleness (π_prox / π_behave)
# ---------------------------------------------------------------------------


@dataclass
class RejectionSamplingConfig:
    """
    Filter tokens whose behavior→proximal importance weight leaves
    [lower, upper]. ``action="mask"`` drops them from the loss;
    ``action="clamp"`` keeps them with a bounded weight.
    """

    action: Literal["mask", "clamp"] = "mask"
    upper: float = 5.0
    lower: float | None = None


def behavior_importance_weight(
    proximal_logprobs: Tensor,
    behavior_logprobs: Tensor,
    loss_mask: Tensor,
    cfg: RejectionSamplingConfig,
) -> tuple[Tensor, Tensor, float]:
    """Return (weight [B,T], new loss_mask [B,T], filtered_fraction)."""
    log_w = (proximal_logprobs - behavior_logprobs).detach()
    w = log_w.exp()
    within = w <= cfg.upper
    if cfg.lower is not None:
        within &= w >= cfg.lower

    n_valid = loss_mask.sum().clamp(min=1)
    if cfg.action == "mask":
        new_mask = loss_mask * within.to(loss_mask.dtype)
        filtered = ((loss_mask > 0) & ~within).sum() / n_valid
        w = torch.where(within, w, torch.zeros_like(w))
        return w, new_mask, float(filtered)
    w_clamped = (
        w.clamp(max=cfg.upper) if cfg.lower is None else w.clamp(cfg.lower, cfg.upper)
    )
    filtered = ((loss_mask > 0) & ~within).sum() / n_valid
    return w_clamped, loss_mask, float(filtered)


# ---------------------------------------------------------------------------
# Policy-gradient surrogates
# ---------------------------------------------------------------------------


def policy_loss(
    logprobs: Tensor,  # [B, T]  π_θ (grad)
    proximal_logprobs: Tensor,  # [B, T]  π_prox (no grad)
    advantages: Tensor,  # [B]     per-trajectory advantage
    loss_mask: Tensor,  # [B, T]  1 = response token
    loss_type: LossType = "grpo",
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    behavior_logprobs: Tensor | None = None,
    rejection: RejectionSamplingConfig | None = None,
) -> tuple[Tensor, dict]:
    """
    Pluggable surrogate objective. All variants reduce with a global
    token-sum denominator (Dr. GRPO style) unless noted.

      grpo     PPO-clip on per-token ratio, token-mean
      dr_grpo  PPO-clip, denominator = B * max_len (no length bias)
      cispo    REINFORCE with clipped, detached importance weight
      gspo     PPO-clip on sequence-level geometric-mean ratio
    """
    orig_mask = loss_mask
    n_orig = orig_mask.sum().clamp(min=1)
    adv = advantages.detach().unsqueeze(1)  # [B, 1]

    behave_w = None
    filtered_fraction = 0.0
    if rejection is not None and behavior_logprobs is not None:
        behave_w, loss_mask, filtered_fraction = behavior_importance_weight(
            proximal_logprobs, behavior_logprobs, loss_mask, rejection
        )

    log_ratio = logprobs - proximal_logprobs.detach()
    hi = clip_eps if clip_eps_high is None else clip_eps_high

    if loss_type == "gspo":
        seq_len = loss_mask.sum(dim=1, keepdim=True).clamp(min=1)
        seq_log_ratio = (log_ratio * loss_mask).sum(dim=1, keepdim=True) / seq_len
        # Per-token gradient path, sequence-level value (GSPO-token).
        ratio = (seq_log_ratio.detach() + log_ratio - log_ratio.detach()).exp()
    else:
        ratio = log_ratio.exp()

    if loss_type == "cispo":
        w = ratio.detach().clamp(1 - clip_eps, 1 + hi)
        pg = -w * adv * logprobs
        clip_mask = (ratio.detach() < 1 - clip_eps) | (ratio.detach() > 1 + hi)
    else:
        pg1 = -ratio * adv
        pg2 = -ratio.clamp(1 - clip_eps, 1 + hi) * adv
        pg = torch.max(pg1, pg2)
        clip_mask = pg1.detach() < pg2.detach()

    if behave_w is not None:
        pg = pg * behave_w

    if loss_type == "dr_grpo":
        denom = torch.tensor(float(loss_mask.numel()), device=pg.device)
    else:
        denom = n_orig
    # where() rather than multiply: padded positions may hold inf ratios
    pg = torch.where(loss_mask > 0, pg, torch.zeros_like(pg))
    loss = pg.sum() / denom

    m = orig_mask > 0
    stats = {
        "pg_loss": loss.detach(),
        "ratio_mean": ratio.detach()[m].mean() if m.any() else torch.zeros(()),
        "ratio_max": ratio.detach()[m].max() if m.any() else torch.zeros(()),
        "clip_frac": (clip_mask & m).sum() / n_orig,
        "approx_kl_prox": (-log_ratio.detach()[m]).mean()
        if m.any()
        else torch.zeros(()),
        "rejected_frac": filtered_fraction,
    }
    if behave_w is not None:
        stats["behave_imp_weight_mean"] = (
            behave_w[m].mean() if m.any() else torch.zeros(())
        )
    return loss, stats


# ---------------------------------------------------------------------------
# KL penalty against the frozen reference
# ---------------------------------------------------------------------------


def kl_penalty(
    logprobs: Tensor,
    ref_logprobs: Tensor,
    loss_mask: Tensor,
    kl_type: KLType = "k1",
) -> tuple[Tensor, Tensor]:
    """Return (per-token kl [B,T], token-mean kl scalar)."""
    diff = ref_logprobs.detach() - logprobs  # log(π_ref / π_θ)
    if kl_type == "k3":
        kl = diff.exp() - diff - 1.0  # unbiased, low-variance estimator
    else:
        kl = -diff  # k1: log(π_θ / π_ref)
    kl = kl * loss_mask
    return kl, kl.sum() / loss_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Turn-level GAE (AReaL #1600) — one "step" per LLM call of a multi-turn episode
# ---------------------------------------------------------------------------

def turn_level_gae(
    turn_rewards: list[float],
    turn_values: list[float] | None = None,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> list[float]:
    """
    Generalised advantage estimation over the turns of one episode.

    ``turn_rewards[t]`` is the reward received after turn t; ``turn_values[t]``
    an (optional) value estimate for the state before turn t — with no critic
    every value is 0 and the result reduces to discounted returns-to-go, which
    is what the proxy's geometric reward discount approximates. With
    ``gamma = lam = 1`` and zero values every turn gets the episode return.
    """
    T = len(turn_rewards)
    values = list(turn_values) if turn_values is not None else [0.0] * T
    values.append(0.0)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        delta = turn_rewards[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv
