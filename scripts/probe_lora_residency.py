"""Measure a local bf16 MPS base with language-only LoRA beside inference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from probe_mac_vlm import memory_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--inference-pid", type=int, required=True)
    parser.add_argument("--inference-report", type=Path)
    parser.add_argument("--step", action="store_true")
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s1/lora-residency.json")
    )
    args = parser.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    def snapshot():
        torch.mps.synchronize()
        return {
            "training": memory_snapshot(os.getpid()),
            "inference": memory_snapshot(args.inference_pid),
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        }

    report = {
        "status": "blocked",
        "scope": "Residency / tiny text step only; not context-length feasibility",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
        "memory_before": snapshot(),
        "method": "LoRA",
        "rank": 8,
        "reference_model_loaded": False,
    }
    started = time.monotonic()
    try:
        if args.inference_report:
            inference = json.loads(args.inference_report.read_text())
            if inference["pid"] != args.inference_pid:
                raise ValueError("Inference report PID does not match live process")
            report["inference_completions_before"] = inference["hold"][
                "completed_requests"
            ]
        if not report["memory_before"]["inference"].get("server_rss_bytes"):
            raise ValueError("Inference process is not resident")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model_path.resolve()),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="mps",
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                bias="none",
                target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
            ),
        )
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        vision = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if ".visual." in name
        ]
        report.update(
            trainable_parameters=sum(parameter.numel() for _, parameter in trainable),
            trainable_tensors=len(trainable),
            language_only_lora=bool(trainable)
            and all(
                ".language_model." in name and ".lora_" in name for name, _ in trainable
            ),
            freeze_vit=bool(vision)
            and all(not parameter.requires_grad for _, parameter in vision),
            base_dtype=str(model.get_input_embeddings().weight.dtype),
            base_device=str(model.get_input_embeddings().weight.device),
            all_parameters_on_mps=all(
                parameter.device.type == "mps" for parameter in model.parameters()
            ),
            memory_after_load=snapshot(),
        )
        if (
            not report["language_only_lora"]
            or not report["freeze_vit"]
            or not report["all_parameters_on_mps"]
            or report["base_dtype"] != "torch.bfloat16"
        ):
            raise ValueError("bf16 MPS / LoRA / frozen ViT contract failed")
        if args.step:
            model.train()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            tokenizer = AutoTokenizer.from_pretrained(
                str(args.model_path.resolve()), local_files_only=True
            )
            batch = tokenizer(
                "A small red square is shown in the image.", return_tensors="pt"
            ).to("mps")
            optimizer = torch.optim.AdamW(
                [parameter for _, parameter in trainable], lr=1e-4
            )
            loss = model(**batch, labels=batch["input_ids"], use_cache=False).loss
            loss.backward()
            gradients = [parameter.grad for _, parameter in trainable]
            report["finite_gradients"] = all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in gradients
            )
            optimizer.step()
            report["loss"] = float(loss.detach())
            report["sequence_tokens"] = batch["input_ids"].numel()
            report["memory_after_step"] = snapshot()
            if not report["finite_gradients"]:
                raise ValueError("Nonfinite or missing adapter gradients")
        report["memory_final"] = snapshot()
        if args.hold_seconds:
            time.sleep(args.hold_seconds)
            report["memory_after_hold"] = snapshot()
        if args.inference_report:
            inference = json.loads(args.inference_report.read_text())
            report["inference_completions_after"] = inference["hold"][
                "completed_requests"
            ]
            report["concurrent_inference_passed"] = (
                report["inference_completions_after"]
                > report["inference_completions_before"]
                and inference["hold"]["last_completion"]["status"] == "passed"
            )
            if not report["concurrent_inference_passed"]:
                raise ValueError("No successful concurrent inference observed")
        report["status"] = "passed"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["memory_after_error"] = snapshot()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
