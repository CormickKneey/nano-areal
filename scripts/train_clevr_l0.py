"""Short native LoRA loss-drop probe on local official CLEVR data, not an RL run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from probe_mac_vlm import memory_snapshot
from torch.nn import functional as F
from transformers import AutoProcessor

from nano_areal.archon.qwen3vl import Qwen3VLModel, add_lora
from nano_areal.clevr import load_clevr


def run(args):
    examples = load_clevr(args.data_root, args.split, args.samples)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    batches = []
    for example in examples:
        with Image.open(example.image_path) as image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": example.question + " Answer briefly."},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            full = processor.apply_chat_template(
                messages + [{"role": "assistant", "content": example.answer}],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        length = prompt["input_ids"].shape[1]
        if not torch.equal(prompt["input_ids"], full["input_ids"][:, :length]):
            raise ValueError("CLEVR prompt is not a prefix of its answer sequence")
        inputs = {
            key: value.to(args.device)
            for key, value in full.items()
            if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        }
        if inputs["input_ids"].shape[1] > args.max_context:
            raise ValueError("CLEVR example exceeds the short-probe context budget")
        positions = torch.arange(
            length - 1, full["input_ids"].shape[1] - 1, device=args.device
        )
        batches.append((inputs, positions))
    model = Qwen3VLModel.from_checkpoint(args.model_path, args.device)
    parameters = add_lora(model)
    model.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    def loss_for(batch):
        inputs, positions = batch
        logits = model(**inputs, logits_to_keep=positions).float()
        targets = inputs["input_ids"][:, positions + 1]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten())

    def evaluate():
        model.eval()
        with torch.no_grad():
            return sum(float(loss_for(batch)) for batch in batches) / len(batches)

    report = {
        "status": "failed",
        "scope": "Local CLEVR supervised overfit probe, not held-out accuracy",
        "examples": len(examples),
        "loss_before": evaluate(),
        "steps": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(batches[step % len(batches)])
            loss.backward()
            if not all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise ValueError("Missing or nonfinite CLEVR adapter gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            report["steps"].append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "memory": memory_snapshot(os.getpid()),
                }
            )
        report["loss_after"] = evaluate()
        if not report["loss_after"] < report["loss_before"]:
            raise ValueError("CLEVR loss did not decrease on the same examples")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", default=".s1-cache/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s3/clevr-loss.json")
    )
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        parser.error("--steps must be positive")
    run(arguments)
