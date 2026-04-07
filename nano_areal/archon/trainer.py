"""
nano-archon trainer: GRPO + FSDP2 + torch.compile.

On single GPU (or macOS sync mode), FSDP is skipped and the model runs
directly on the target device.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nano_areal.config import TrainConfig
    from nano_areal.archon.model import Qwen35Model


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


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    log_probs: Tensor,       # [B, T] — current policy log probs (response tokens only)
    ref_log_probs: Tensor,   # [B, T] — reference policy log probs
    rewards: Tensor,         # [B]    — scalar reward per trajectory
    masks: Tensor,           # [B, T] — response token mask
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> tuple[Tensor, dict]:
    """
    GRPO objective with PPO-clip and KL penalty.

    Within each group (same problem, G trajectories), advantages are
    computed by group-normalizing rewards.
    """
    B = rewards.shape[0]

    # Group-normalize rewards → advantages
    # Assume the batch is already one full group or multiple groups of equal size.
    # For simplicity, normalize over the entire batch.
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # [B]
    advantages = advantages.detach()

    # Per-token importance ratio
    ratio = (log_probs - ref_log_probs).exp()  # [B, T]

    # PPO-clip
    adv_expanded = advantages.unsqueeze(1)  # [B, 1] → broadcasts over T
    pg_loss1 = -ratio * adv_expanded
    pg_loss2 = -ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_expanded
    pg_loss = torch.max(pg_loss1, pg_loss2)  # [B, T]

    # KL penalty (forward KL: current || ref)
    kl = (log_probs - ref_log_probs) * masks  # [B, T]

    # Mask and reduce
    n_tokens = masks.sum()
    loss = ((pg_loss + kl_coef * kl) * masks).sum() / n_tokens.clamp(min=1)

    stats = {
        "loss": loss.item(),
        "kl": (kl.sum() / n_tokens.clamp(min=1)).item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "ratio_mean": (ratio * masks).sum().item() / n_tokens.clamp(min=1).item(),
    }
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
        self.dtype = torch.bfloat16

        # Move model to device
        self.model = model.to(self.device, dtype=self.dtype)

        # FSDP2 wrapping (multi-GPU) or plain module (single GPU / macOS)
        self.model = self._maybe_wrap_fsdp(self.model)

        # Frozen reference model (not wrapped — always single copy)
        self.ref_model = copy.deepcopy(model).to(self.device, dtype=self.dtype)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

        # torch.compile
        if config.compile:
            self.model = torch.compile(self.model, backend=config.compile_backend)
            self.ref_model = torch.compile(self.ref_model, backend=config.compile_backend)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            fused=(self.device.type == "cuda"),  # fused only on CUDA
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
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from nano_areal.archon.model import TransformerBlock
            import functools

            mp = MixedPrecision(
                param_dtype=self.dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=self.dtype,
            )
            auto_wrap = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerBlock},
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

    @torch.no_grad()
    def _compute_ref_log_probs(
        self,
        input_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        return self.ref_model.compute_log_probs(input_ids, response_mask)

    def _compute_policy_log_probs(
        self,
        input_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        return self.model.compute_log_probs(input_ids, response_mask)

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
        rewards = torch.tensor(
            [t.reward for t in trajectories],
            dtype=torch.float32,
            device=self.device,
        )

        # Recompute log-probs (required for off-policy data)
        with torch.no_grad():
            ref_log_probs = self._compute_ref_log_probs(input_ids, response_mask)

        log_probs = self._compute_policy_log_probs(input_ids, response_mask)

        loss, stats = grpo_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            masks=response_mask[:, 1:],
            clip_eps=self.config.clip_eps,
            kl_coef=self.config.kl_coef,
        )

        # Gradient accumulation
        loss = loss / self.config.grad_accum_steps
        loss.backward()

        self._accum_stats.append(stats)
        self._step += 1

        if self._step % self.config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
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

    def get_state_dict(self) -> dict:
        """Return current policy weights (for awex weight sync)."""
        raw = getattr(self.model, "_orig_mod", self.model)
        if hasattr(raw, "module"):
            raw = raw.module
        return {k: v.cpu() for k, v in raw.state_dict().items()}
