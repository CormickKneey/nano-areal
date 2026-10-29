"""Policy losses for ``--loss grpo|cispo|gspo|drgrpo``.

GRPO is the default. The other three are the P1 toggles: same token
trajectories, different surrogate. Every variant keeps the KL penalty so
the switch is the advantage and the policy term.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

LossName = Literal["grpo", "cispo", "gspo", "drgrpo"]
LOSS_NAMES: tuple[str, ...] = ("grpo", "cispo", "gspo", "drgrpo")


def group_advantages(
    rewards: Tensor,
    group_size: int | None,
    *,
    scale: bool = True,
    on_incomplete: Literal["batch", "zeros"] = "batch",
    group_ids: list[str] | None = None,
) -> tuple[Tensor, dict]:
    """Turn scalar rewards into detached advantages.

    With ``group_size`` set and the batch divisible by it, each contiguous
    group is centered on its own. ``scale=False`` is Dr.GRPO (no division
    by the group std). A group with no spread gets zeros instead of a
    huge ratio from a tiny std.

    A short tail (sync step that did not fill G) is not a full group.
    ``on_incomplete="batch"`` falls back to one batch-wide normalization so
    the step can still train. ``on_incomplete="zeros"`` leaves that tail at
    zero advantage.
    """
    stats = {"incomplete_groups": 0, "skipped_groups": 0, "advantage_scope": "batch"}
    if rewards.numel() == 0:
        return rewards, stats

    if group_ids is not None:
        if len(group_ids) != rewards.numel():
            raise ValueError("group_ids must match the reward batch")
        advantages = torch.zeros_like(rewards)
        stats["advantage_scope"] = "group_ids"
        for group_id in dict.fromkeys(group_ids):
            indices = [index for index, value in enumerate(group_ids) if value == group_id]
            row = rewards[indices]
            advantages[indices] = _normalize(row, scale=scale)
            if row.numel() == 1:
                stats["incomplete_groups"] += 1
        return advantages.detach(), stats

    if group_size is None or group_size <= 1:
        return _normalize(rewards, scale=scale).detach(), stats

    n_full, tail = divmod(int(rewards.numel()), group_size)
    if n_full == 0 or tail:
        stats["incomplete_groups"] = 1
        if on_incomplete == "batch" or n_full == 0:
            return _normalize(rewards, scale=scale).detach(), stats

    stats["advantage_scope"] = "group"
    pieces: list[Tensor] = []
    full = rewards[: n_full * group_size].view(n_full, group_size)
    for row in full:
        if scale and float(row.std(unbiased=False)) < 1e-6:
            pieces.append(torch.zeros_like(row))
            stats["skipped_groups"] += 1
            continue
        pieces.append(_normalize(row, scale=scale))
    advantages = torch.cat(pieces, dim=0)
    if tail and on_incomplete == "zeros":
        advantages = torch.cat([advantages, torch.zeros(tail, device=rewards.device, dtype=rewards.dtype)])
    return advantages.detach(), stats


def _normalize(row: Tensor, *, scale: bool) -> Tensor:
    centered = row - row.mean()
    if not scale:
        return centered
    std = row.std(unbiased=False) if row.numel() > 1 else row.new_zeros(())
    if float(std) < 1e-6:
        return torch.zeros_like(row)
    return centered / (std + 1e-8)


def is_offpolicy(version_gap: int, max_head_offpolicyness: int) -> bool:
    """True when a rollout is too old for the current policy.

    ``max_head_offpolicyness == 0`` is the macOS sync rule: only the
    current version is kept. A negative gap (rollout newer than the
    trainer) is also invalid; sync admits only an exact version match.
    """
    return not 0 <= version_gap <= max_head_offpolicyness


def compute_policy_loss(
    loss: str,
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    masks: Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    group_size: int | None = None,
    on_incomplete: Literal["batch", "zeros"] = "batch",
    old_log_probs: Tensor | None = None,
    group_ids: list[str] | None = None,
) -> tuple[Tensor, dict]:
    """Dispatch one policy loss. ``log_probs`` must require grad to train."""
    if loss not in LOSS_NAMES:
        known = ", ".join(LOSS_NAMES)
        raise ValueError(f"Unknown loss {loss!r}. Expected one of: {known}")
    if rewards.numel() == 0:
        raise ValueError("Policy loss requires at least one trajectory")
    if old_log_probs is None:
        old_log_probs = log_probs.detach()
    if old_log_probs.shape != log_probs.shape or ref_log_probs.shape != log_probs.shape:
        raise ValueError(
            "Policy, behavior and reference logprobs must have identical shapes"
        )

    scale = loss != "drgrpo"
    advantages, adv_stats = group_advantages(
        rewards, group_size, scale=scale, on_incomplete=on_incomplete,
        group_ids=group_ids,
    )
    adv = advantages.unsqueeze(1)
    log_ratio = log_probs - old_log_probs.detach()
    token_ratio = log_ratio.exp()
    log_ratio_ref = ref_log_probs.detach() - log_probs
    kl = (log_ratio_ref.exp() - log_ratio_ref - 1) * masks

    if loss == "cispo":
        # Clipped importance weight is detached. Gradient stays on log π,
        # including samples PPO would treat as fully clipped.
        weight = token_ratio.clamp(1 - clip_eps, 1 + clip_eps).detach()
        pg = -(weight * adv) * log_probs
        ratio_for_stats = weight
    elif loss == "gspo":
        lengths = masks.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (log_ratio * masks).sum(dim=-1) / lengths
        seq_ratio = seq_log_ratio.exp().unsqueeze(1)
        clipped = seq_ratio.clamp(1 - clip_eps, 1 + clip_eps)
        pg1 = -seq_ratio * adv
        pg2 = -clipped * adv
        pg = torch.max(pg1, pg2)
        ratio_for_stats = seq_ratio.expand_as(token_ratio)
    else:
        # GRPO and Dr.GRPO share the token-level PPO clip.
        clipped = token_ratio.clamp(1 - clip_eps, 1 + clip_eps)
        pg = torch.max(-token_ratio * adv, -clipped * adv)
        ratio_for_stats = token_ratio

    per_token = (pg + kl_coef * kl) * masks
    if loss == "drgrpo":
        # Sequence-mean, then batch-mean: a long sample does not dominate.
        lengths = masks.sum(dim=-1).clamp(min=1)
        seq_loss = per_token.sum(dim=-1) / lengths
        loss_t = seq_loss.mean()
        n_tokens = masks.sum().clamp(min=1)
    else:
        n_tokens = masks.sum().clamp(min=1)
        loss_t = per_token.sum() / n_tokens

    stats = {
        "loss": loss_t.item(),
        "kl": (kl.sum() / n_tokens).item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std(correction=0).item(),
        "ratio_mean": ((ratio_for_stats * masks).sum() / n_tokens).item(),
        "loss_name": loss,
        **adv_stats,
    }
    return loss_t, stats
