"""Audit the current trainer against S1 GRPO contracts, without model weights."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from nano_areal.archon.trainer import grpo_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s1/grpo-parity.json")
    )
    args = parser.parse_args()
    checks = {}
    try:
        policy = torch.tensor([[-0.5]], device=args.device, requires_grad=True)
        loss, _ = grpo_loss(
            policy,
            policy.detach(),
            torch.ones(1, device=args.device),
            torch.ones_like(policy),
            kl_coef=0,
        )
        loss.backward()
        checks["singleton_group_finite"] = bool(
            torch.isfinite(loss) and torch.isfinite(policy.grad).all()
        )
        policy = torch.tensor(
            [[-0.5, -0.4], [-0.6, -0.3]], device=args.device, requires_grad=True
        )
        reference = policy.detach() - 0.5
        _, stats = grpo_loss(
            policy,
            reference,
            torch.tensor([0.0, 1.0], device=args.device),
            torch.ones_like(policy),
            kl_coef=0,
        )
        checks["sync_ratio_independent_of_frozen_reference"] = (
            abs(stats["ratio_mean"] - 1) < 1e-6
        )
        parameters = inspect.signature(grpo_loss).parameters
        checks["explicit_behavior_logprobs"] = "old_log_probs" in parameters
        checks["explicit_problem_groups"] = "group_ids" in parameters
        report = {
            "status": "passed" if all(checks.values()) else "blocked",
            "device": args.device,
            "checks": checks,
            "scope": "S1 contract audit; not upstream numeric or VLM archon parity",
        }
    except (RuntimeError, ImportError) as error:
        report = {
            "status": "blocked",
            "device": args.device,
            "error_type": type(error).__name__,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
