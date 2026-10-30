"""Bounded native/production policy preflight; never hide failed update gates."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from probe_mac_vlm import memory_snapshot
from safetensors.torch import load_file

from nano_areal.archon.losses import group_advantages
from nano_areal.archon.qwen3vl_lora import NativeQwen3VLLoRATrainer
from nano_areal.sync.policy import PolicyBinding, checkpoint_files, file_digest


def restore(trainer, adapter):
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    for index, layer in enumerate(trainer.model.language_model.layers):
        for projection in ("q_proj", "v_proj"):
            module = getattr(layer.attn, projection)
            for side in ("a", "b"):
                key = (
                    f"base_model.model.model.language_model.layers.{index}."
                    f"self_attn.{projection}.lora_{side.upper()}.weight"
                )
                with torch.no_grad():
                    getattr(module, f"lora_{side}").copy_(weights.pop(key))
    if weights:
        raise ValueError("Unexpected adapter tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight"))
    parser.add_argument("--infer-model", type=Path, required=True)
    parser.add_argument("--train-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", type=Path)
    parser.add_argument("--max-logprob-mae", type=float, default=0.01)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.phase}.json"
    report = {
        "status": "started", "phase": args.phase,
        "scope": "S4 native bf16 / production 4-bit training preflight",
        "memory_before": memory_snapshot(os.getpid()),
        "updates": 0, "elapsed_seconds": 0,
    }
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        torch.manual_seed(42)
        trainer = NativeQwen3VLLoRATrainer(args.train_model)
        trainer.model.eval()
        torch.mps.synchronize()
        report["memory_loaded"] = memory_snapshot(os.getpid())
        report["trainable_parameters"] = sum(
            parameter.numel() for parameter in trainer.parameters.values()
        )
        adapter = args.output / "adapter-v0"
        manifest_path = args.output / "policy-v0.json"
        if args.phase == "prepare":
            previous = {
                name: value.detach().cpu().clone()
                for name, value in trainer.parameters.items()
            }
            trainer.export_peft_adapter(adapter)
            restore(trainer, adapter)
            report["adapter_roundtrip_exact"] = all(
                torch.equal(previous[name], value.detach().cpu())
                for name, value in trainer.parameters.items()
            )
            if not report["adapter_roundtrip_exact"]:
                raise ValueError("Native PEFT adapter roundtrip changed weights")
            binding = PolicyBinding({
                "version": 0,
                "base_model": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                "adapter": str(adapter.resolve()),
                "inference_files": checkpoint_files(args.infer_model),
                "training_files": checkpoint_files(args.train_model),
                "adapter_files": {
                    str(item.resolve()): file_digest(item)
                    for item in sorted(adapter.iterdir()) if item.is_file()
                },
                "training_base": "native archon bf16",
                "inference_base": "production vllm-metal MLX 4-bit",
                "base_equivalence": "unverified; must measure behavior logprobs",
            })
            binding.write(manifest_path)
            report.update(status="prepared", model=binding.model)
        else:
            binding = PolicyBinding.read(manifest_path)
            binding.verify_files()
            restore(trainer, adapter)
            summary = json.loads((args.group / "report.json").read_text())
            if (summary["attempted"] != 2 or summary["passed"] != 2
                    or summary["planned"] != 2):
                raise ValueError(
                    "Preflight requires a complete fixed two-episode group"
                )
            report["group"] = str(args.group)
            report["turns"] = []
            rewards = []
            total_error, total_tokens = 0.0, 0
            for index in range(2):
                episode = json.loads((args.group / f"episode-{index}.json").read_text())
                if episode["policy"] != {"model": binding.model, "version": 0}:
                    raise ValueError("Episode policy provenance mismatch")
                rewards.append(episode["reward"])
                for turn, item in enumerate(episode["turns"]):
                    if item["truncated"]:
                        raise ValueError("Truncated group cannot be used for training")
                    trajectory = SimpleNamespace(
                        input_ids=torch.tensor(item["input_ids"]),
                        response_mask=torch.tensor(item["response_mask"]),
                        pixel_values=None,
                    )
                    with torch.no_grad():
                        current = trainer.response_logprobs(trajectory).cpu()[0]
                    old = torch.tensor(item["old_log_probs"])[
                        trajectory.response_mask.bool()
                    ]
                    error = (current - old).abs()
                    if not bool(torch.isfinite(error).all()):
                        raise ValueError("Nonfinite policy comparison")
                    total_error += float(error.sum())
                    total_tokens += error.numel()
                    report["turns"].append({
                        "episode": index, "turn": turn,
                        "sequence_tokens": len(item["input_ids"]),
                        "response_tokens": error.numel(),
                        "logprob_mae": float(error.mean()),
                        "logprob_max_error": float(error.max()),
                    })
                    report["memory_forward"] = memory_snapshot(os.getpid())
                    report["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
                    report["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
                    save()
            report["response_tokens"] = total_tokens
            report["behavior_logprob_mae"] = total_error / total_tokens
            report["max_logprob_mae"] = args.max_logprob_mae
            report["policy_parity_passed"] = (
                report["behavior_logprob_mae"] <= args.max_logprob_mae
            )
            report["episode_rewards"] = rewards
            advantages, _ = group_advantages(
                torch.tensor(rewards), 2, group_ids=["fixed-task"] * 2
            )
            report["episode_advantages"] = advantages.tolist()
            report["reward_spread"] = max(rewards) - min(rewards)
            report["nonzero_grpo_possible"] = report["reward_spread"] > 0
            report["status"] = "blocked"
            report["blockers"] = []
            if not report["policy_parity_passed"]:
                report["blockers"].append("bf16/4-bit behavior policy drift")
            if not report["nonzero_grpo_possible"]:
                report["blockers"].append(
                    "Homogeneous episode rewards; zero advantages"
                )
            if not report["blockers"]:
                report["status"] = "preflight-passed-no-update"
            binding.verify_files()
        report["memory_after"] = memory_snapshot(os.getpid())
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    print(json.dumps(report, indent=2), flush=True)
    passed = report["status"] in {"prepared", "preflight-passed-no-update"}
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
