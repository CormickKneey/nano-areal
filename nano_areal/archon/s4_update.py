"""One episode-balanced GRPO step over an already validated complete group."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from nano_areal.archon.losses import compute_policy_loss, group_advantages


def trajectory_from_turn(turn):
    ids = torch.tensor(turn["input_ids"], dtype=torch.long)
    raw_mask = torch.tensor(turn["response_mask"])
    mask = raw_mask.bool()
    old = torch.tensor(turn["old_log_probs"], dtype=torch.float32)
    if (ids.ndim != 1 or mask.shape != ids.shape or old.shape != ids.shape
            or ids.numel() < 2 or bool(mask[0]) or not bool(mask.any())
            or not bool(((raw_mask == 0) | (raw_mask == 1)).all())
            or not bool(torch.isfinite(old[mask]).all()) or turn["truncated"]):
        raise ValueError("Malformed or truncated training turn")
    return SimpleNamespace(input_ids=ids, response_mask=mask, pixel_values=None), old


def bounded_step(trainer, episodes, snapshot, max_seconds=600):
    rewards = torch.tensor([episode["reward"] for episode in episodes])
    advantages, _ = group_advantages(rewards, len(episodes))
    if len(episodes) < 2 or not bool(advantages.abs().any()):
        raise ValueError("Homogeneous or incomplete group cannot update")
    previous = {name: value.detach().cpu().clone()
                for name, value in trainer.parameters.items()}
    trainer.optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    loss_sum = 0.0
    ratios = []
    turn_index = 0
    for index, episode in enumerate(episodes):
        token_count = sum(sum(turn["response_mask"]) for turn in episode["turns"])
        if not token_count:
            raise ValueError("Empty episode cannot update")
        for turn in episode["turns"]:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Bounded optimizer wall-clock limit exceeded")
            trajectory, old = trajectory_from_turn(turn)
            if hasattr(trainer, "select_turn"):
                trainer.select_turn(turn_index)
            trainer.model.eval()
            with torch.no_grad(), trainer.model.disable_adapter():
                reference = trainer.response_logprobs(trajectory)
            trainer.model.train()
            current = trainer.response_logprobs(trajectory)
            device = current.device
            behavior = old[trajectory.response_mask].to(device).unsqueeze(0)
            loss, stats = compute_policy_loss(
                "grpo", current, reference, rewards[index:index + 1].to(device),
                torch.ones_like(current), old_log_probs=behavior, kl_coef=0.01,
                advantages=advantages[index:index + 1].to(device),
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite GRPO loss")
            weight = current.numel() / token_count / len(episodes)
            snapshot("forward")
            (loss * weight).backward()
            snapshot("backward")
            loss_sum += float(loss.detach()) * weight
            ratios.append(stats["ratio_mean"])
            del current, reference, loss
            turn_index += 1
    if not all(parameter.grad is not None
               and bool(torch.isfinite(parameter.grad).all())
               for parameter in trainer.parameters.values()):
        raise ValueError("Missing or nonfinite adapter gradients")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        list(trainer.parameters.values()), 1.0,
    ))
    if not grad_norm or not torch.isfinite(torch.tensor(grad_norm)):
        raise ValueError("Zero or nonfinite GRPO gradient")
    trainer.optimizer.step()
    snapshot("optimizer")
    delta = sum(float((value.detach().cpu() - previous[name]).abs().sum())
                for name, value in trainer.parameters.items())
    if not delta:
        raise ValueError("Optimizer produced no adapter change")
    trainer.model.eval()
    return {"loss": loss_sum, "episode_rewards": rewards.tolist(),
            "episode_advantages": advantages.tolist(), "grad_norm": grad_norm,
            "adapter_delta_l1": delta, "ratio_mean": sum(ratios) / len(ratios),
            "optimizer_steps": 1, "elapsed_seconds": time.monotonic() - started,
            "weighting": "equal episodes; response-token mean within each episode",
            "reference": "same frozen aligned base with adapters disabled"}
