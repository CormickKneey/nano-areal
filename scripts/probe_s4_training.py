"""Sequential policy preflight, bounded single-step update and immutable export."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import socket
import threading
import time
from pathlib import Path

import torch
from probe_mac_vlm import command_output, memory_snapshot
from safetensors.torch import load_file

from nano_areal.archon.losses import group_advantages
from nano_areal.archon.qwen3vl_lora import NativeQwen3VLLoRATrainer
from nano_areal.archon.s4_update import bounded_step, trajectory_from_turn
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


def complete_group(directory, binding):
    summary = json.loads((directory / "report.json").read_text())
    count = summary["planned"]
    if not 2 <= count <= 8 or summary["attempted"] != count:
        raise ValueError("Preflight requires a complete declared episode group")
    if len(summary["episodes"]) != count:
        raise ValueError("Missing episode accounting")
    episodes = []
    for index, row in enumerate(summary["episodes"]):
        if row["index"] != index or not row.get("training_eligible", row["passed"]):
            raise ValueError("Incomplete or unreconciled episode cannot train")
        episode = json.loads((directory / f"episode-{index}.json").read_text())
        if (episode["episode"] != index or episode["policy"]
                != {"model": binding.model, "version": binding.version}):
            raise ValueError("Episode policy provenance mismatch")
        verification = row["result"]["verification"]
        if (episode["reward"] != row["result"]["verified_reward"]
                or episode["reward"] != verification["pass_rate"]
                or not 0 <= episode["reward"] <= 1):
            raise ValueError("Episode reward differs from external verification")
        if not episode["turns"] or any(item["truncated"] for item in episode["turns"]):
            raise ValueError("Empty or truncated group cannot be used for training")
        episodes.append(episode)
    return episodes


class MemoryTrace:
    def __init__(self):
        self.samples = []
        self.recommended_max_bytes = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        try:
            self.recommended_max_bytes = torch.mps.recommended_max_memory()
            while not self.stop_event.is_set():
                self.samples.append({
                    "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                    "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                })
                self.stop_event.wait(0.1)
        except Exception as error:
            self.error = {"type": type(error).__name__, "message": str(error)}

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        return {
            "scope": "100ms sampled MPS peaks, not allocator-exact peaks",
            "samples": len(self.samples),
            "mps_allocated_peak_bytes": max(
                (row["mps_allocated_bytes"] for row in self.samples), default=None,
            ),
            "mps_driver_peak_bytes": max(
                (row["mps_driver_bytes"] for row in self.samples), default=None,
            ),
            "process_rss_highwater_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "mps_recommended_max_bytes": self.recommended_max_bytes,
            "sampling_valid": bool(self.samples) and self.error is None,
            "sampler_error": self.error,
        }


def require_stopped_server():
    try:
        connection = socket.create_connection(("127.0.0.1", 8000), timeout=0.5)
    except ConnectionRefusedError:
        return
    connection.close()
    raise ValueError("Stop the owned production server before loading native weights")


def swap_used_mib(snapshot):
    match = re.search(r"used = ([\d.]+)M", snapshot["swap"])
    if not match:
        raise ValueError("Unable to parse macOS swap measurement")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "update"))
    parser.add_argument("--infer-model", type=Path, required=True)
    parser.add_argument("--train-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", type=Path)
    parser.add_argument("--policy-version", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-logprob-mae", type=float, default=0.01)
    parser.add_argument("--aligned-arithmetic", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_logprob_mae <= 0.01:
        parser.error("The behavior MAE bound must be positive and at most 0.01")
    if args.phase != "preflight" and args.policy_version != 0:
        parser.error("Prepare/update only accepts v0; update creates immutable v1")
    if args.phase != "prepare" and args.group is None:
        parser.error("Preflight/update requires a complete episode group")
    if args.phase == "prepare" and args.aligned_arithmetic:
        parser.error("Aligned arithmetic requires an existing bound policy and group")
    require_stopped_server()
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"-v{args.policy_version}" if args.policy_version else ""
    path = args.output / f"{args.phase}{suffix}.json"
    if args.phase == "update" and any(item.exists() for item in (
        path, args.output / "adapter-v1", args.output / "policy-v1.json",
    )):
        parser.error("Immutable update evidence already exists; do not overwrite it")
    report = {
        "status": "started", "phase": args.phase,
        "scope": "S4 native / production training preflight",
        "memory_before": memory_snapshot(os.getpid()),
        "memory_pressure_before": command_output(["memory_pressure", "-Q"]),
        "mps_allocator_environment": {
            name: os.environ.get(name) for name in (
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            )
        },
        "updates": 0, "elapsed_seconds": 0,
    }
    started = time.monotonic()
    trace = MemoryTrace() if args.phase == "update" else None
    aligned_runtime = None

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if args.phase == "update" and (args.output / "adapter-v1").exists():
            raise FileExistsError("Immutable v1 adapter already exists")
        adapter = args.output / f"adapter-v{args.policy_version}"
        manifest_path = args.output / f"policy-v{args.policy_version}.json"
        if args.phase != "prepare":
            binding = PolicyBinding.read(manifest_path)
            binding.verify_files()
            if (checkpoint_files(args.train_model) != binding.manifest["training_files"]
                    or checkpoint_files(args.infer_model)
                    != binding.manifest["inference_files"]):
                raise ValueError("Requested checkpoint differs from bound policy")
            episodes = complete_group(args.group, binding)
        if trace:
            trace.thread.start()
        torch.manual_seed(42)
        trainer = NativeQwen3VLLoRATrainer(args.train_model)
        trainer.model.eval()
        torch.mps.synchronize()
        report["memory_loaded"] = memory_snapshot(os.getpid())
        report["trainable_parameters"] = sum(
            parameter.numel() for parameter in trainer.parameters.values()
        )
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
                "training_base": str(args.train_model.resolve()),
                "inference_base": "production vllm-metal MLX 4-bit",
                "base_equivalence": "unverified; must measure behavior logprobs",
            })
            binding.write(manifest_path)
            report.update(status="prepared", model=binding.model)
        else:
            restore(trainer, adapter)
            if args.aligned_arithmetic:
                from nano_areal.archon.s4_native import (
                    AlignedS4Runtime,
                    load_affine_projections,
                )

                aligned_runtime = AlignedS4Runtime(
                    trainer, load_affine_projections(binding), episodes)
                report["training_arithmetic"] = {
                    **aligned_runtime.contract(), "policy_model": binding.model,
                    "probe_source_sha256": file_digest(Path(__file__)),
                    "policy_manifest_sha256": file_digest(manifest_path),
                    "group_evidence_sha256": {
                        str(item): file_digest(item) for item in (
                            args.group / "report.json", *[
                                args.group / f"episode-{index}.json"
                                for index in range(len(episodes))])},
                }
            report["group"] = str(args.group)
            report["turns"] = []
            rewards = []
            total_error, total_tokens = 0.0, 0
            for index, episode in enumerate(episodes):
                rewards.append(episode["reward"])
                for turn, item in enumerate(episode["turns"]):
                    if item["truncated"]:
                        raise ValueError("Truncated group cannot be used for training")
                    trajectory, captured = trajectory_from_turn(item)
                    if aligned_runtime is not None:
                        aligned_runtime.select_turn(len(report["turns"]))
                    with torch.no_grad():
                        current = trainer.response_logprobs(trajectory).cpu()[0]
                    old = captured[trajectory.response_mask]
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
                torch.tensor(rewards), len(rewards),
                group_ids=["fixed-task"] * len(rewards),
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
            if args.phase == "update" and not report["blockers"]:
                update_started = time.monotonic()
                if aligned_runtime is not None:
                    aligned_runtime.start_update()
                report["phase_memory"] = []

                def snapshot(stage):
                    if stage == "optimizer":
                        report["updates"] = 1
                    torch.mps.synchronize()
                    report["phase_memory"].append({
                        "stage": stage, **memory_snapshot(os.getpid()),
                        "update_elapsed_seconds": time.monotonic() - update_started,
                        "replay_segment_calls": getattr(
                            aligned_runtime, "segment_calls", None),
                        "replay_cache_reclaims": getattr(
                            aligned_runtime, "cache_reclaims", None),
                        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                    })
                    save()

                report["optimizer"] = bounded_step(trainer, episodes, snapshot)
                report["updates"] = report["optimizer"]["optimizer_steps"]
                changed = args.output / "adapter-v1"
                trainer.export_peft_adapter(changed)
                snapshot("export")
                before = file_digest(adapter / "adapter_model.safetensors")
                after = file_digest(changed / "adapter_model.safetensors")
                if before == after:
                    raise ValueError("Exported adapter did not change")
                previous = {name: value.detach().cpu().clone()
                            for name, value in trainer.parameters.items()}
                restore(trainer, changed)
                if not all(torch.equal(previous[name], value.detach().cpu())
                           for name, value in trainer.parameters.items()):
                    raise ValueError("Updated PEFT export is not an exact roundtrip")
                updated = PolicyBinding({
                    **{key: value for key, value in binding.manifest.items()
                       if key != "model"}, "version": 1,
                    "adapter": str(changed.resolve()),
                    "adapter_files": {str(item.resolve()): file_digest(item)
                                      for item in sorted(changed.iterdir())},
                    "parent_policy": binding.model,
                    "complete_group_files": {
                        str(item.resolve()): file_digest(item)
                        for item in [args.group / "report.json", *[
                            args.group / f"episode-{index}.json"
                            for index in range(len(episodes))
                        ]]
                    },
                })
                updated.write(args.output / "policy-v1.json")
                report.update(status="updated-exported-feedback-pending",
                              model=updated.model, adapter_sha256_before=before,
                              adapter_sha256_after=after, adapter_roundtrip_exact=True)
            binding.verify_files()
            if aligned_runtime is not None:
                aligned_runtime.verify_sources()
                if report["training_arithmetic"]["probe_source_sha256"] != file_digest(
                        Path(__file__)):
                    raise ValueError("Bound preflight/update source changed")
                evidence = report["training_arithmetic"]["group_evidence_sha256"]
                if evidence != {name: file_digest(Path(name)) for name in evidence}:
                    raise ValueError("Bound group evidence changed during evaluation")
        report["memory_after"] = memory_snapshot(os.getpid())
        report["memory_pressure_after"] = command_output(["memory_pressure", "-Q"])
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        if aligned_runtime is not None:
            report["replay_progress"] = {
                "segment_calls": getattr(aligned_runtime, "segment_calls", None),
                "cache_reclaims": getattr(aligned_runtime, "cache_reclaims", None),
                "last_segment": getattr(aligned_runtime, "last_segment", None),
            }
        try:
            report["memory_after"] = memory_snapshot(os.getpid())
            report["memory_pressure_after"] = command_output(["memory_pressure", "-Q"])
        except Exception as measurement_error:
            report["memory_after_error"] = {
                "type": type(measurement_error).__name__,
                "message": str(measurement_error),
            }
        raise
    finally:
        if aligned_runtime is not None:
            try:
                aligned_runtime.close()
            except Exception as cleanup_error:
                report["cleanup_error"] = {
                    "type": type(cleanup_error).__name__, "message": str(cleanup_error),
                }
                if report["status"] != "error":
                    report.update(
                        status="error", error_type=type(cleanup_error).__name__,
                        error=str(cleanup_error),
                    )
        if trace and trace.thread.ident is not None:
            report["memory_trace"] = trace.finish()
            if "memory_after" in report:
                report["memory_gate"] = {
                    "scope": "Measured bounded sequences only; not 8192-token training",
                    "swap_growth_mib": swap_used_mib(report["memory_after"])
                    - swap_used_mib(report["memory_before"]),
                    "driver_within_recommended_budget": (
                        report["memory_trace"]["sampling_valid"]
                        and report["memory_trace"]["mps_driver_peak_bytes"]
                        <= report["memory_trace"]["mps_recommended_max_bytes"]
                    ),
                }
                gate = report["memory_gate"]
                gate["passed"] = (gate["swap_growth_mib"] <= 0
                                  and gate["driver_within_recommended_budget"])
                if (report["status"] == "updated-exported-feedback-pending"
                        and not gate["passed"]):
                    report["status"] = "updated-exported-memory-blocked"
        save()
    print(json.dumps(report, indent=2), flush=True)
    passed = report["status"] in {
        "prepared", "preflight-passed-no-update", "updated-exported-feedback-pending",
    }
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
