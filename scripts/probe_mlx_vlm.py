"""Run local Qwen3-VL text/image inference on Metal, without an HTTP adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
import os
import platform
import time
from pathlib import Path

from probe_mac_vlm import memory_snapshot, red_png


def collect_completion(chunks) -> dict:
    text = []
    tokens = []
    logprobs = []
    final = None
    for chunk in chunks:
        text.append(chunk.text)
        if chunk.finish_reason is None or (
            chunk.token_ids and len(chunk.token_ids) > len(tokens)
        ):
            if chunk.token is not None:
                tokens.append(int(chunk.token))
                logprobs.append(float(chunk.logprobs.reshape(-1)[chunk.token].item()))
        if chunk.finish_reason is not None:
            final = chunk
    if final is None:
        raise ValueError("Generation did not return a terminal result")
    valid = bool(logprobs) and all(
        math.isfinite(value) and value <= 0 for value in logprobs
    )
    matched = tokens == final.token_ids and len(tokens) == final.generation_tokens
    return {
        "status": "passed"
        if valid and matched and final.finish_reason == "stop"
        else "blocked",
        "response_text": "".join(text),
        "finish_reason": final.finish_reason,
        "prompt_tokens": final.prompt_tokens,
        "completion_tokens": final.generation_tokens,
        "token_ids": tokens,
        "token_logprobs": logprobs,
        "finite_logprobs": valid,
        "logprobs_match_usage": matched,
        "includes_stop_token": final.finish_reason == "stop",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal",), default="metal")
    parser.add_argument("--sync", action="store_true", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s1/mlx-probe.json")
    )
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("MLX inference requires macOS")
    import mlx.core as mx
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from PIL import Image

    def snapshot():
        return {
            **memory_snapshot(os.getpid()),
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
        }

    report = {
        "backend": args.backend,
        "sync": args.sync,
        "mode": "direct_mlx_not_vllm_http",
        "python": platform.python_version(),
        "pid": os.getpid(),
        "logprob_normalization_dtype": "float32",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-vlm", "transformers")
        },
        "memory_before_load": snapshot(),
        "training_co_residency": "not_measured",
        "l1_max_model_len": {"candidate": 8192, "validated": False},
    }
    model, processor = load(str(args.model_path.resolve()))
    report["memory_after_load"] = snapshot()
    for name, question, images in (
        ("text", "Reply with exactly OK.", None),
        (
            "single_image",
            "Name the color of the image in one word.",
            [Image.open(io.BytesIO(red_png()))],
        ),
    ):
        prompt = apply_chat_template(
            processor, model.config, question, num_images=len(images or [])
        )
        inputs = prepare_inputs(processor, images=images, prompts=prompt)
        image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_tokens = int((inputs["input_ids"] == image_id).sum().item())
        prompt_tokens = int(inputs["input_ids"].size)
        mx.reset_peak_memory()
        started = time.monotonic()
        result = collect_completion(
            stream_generate(
                model,
                processor,
                prompt,
                image=images,
                max_tokens=args.max_tokens,
                temperature=0,
                logits_processors=[lambda tokens, logits: logits.astype(mx.float32)],
            )
        )
        result.update(
            elapsed_seconds=round(time.monotonic() - started, 3),
            processor_prompt_tokens=prompt_tokens,
            image_tokens=image_tokens,
            processor_matches_generation=prompt_tokens == result["prompt_tokens"],
            memory_after=snapshot(),
        )
        if images:
            result["red_image_recognized"] = (
                result["response_text"].strip().lower().strip(".!\"'") == "red"
            )
        if not result["processor_matches_generation"] or (
            images and not result["red_image_recognized"]
        ):
            result["status"] = "blocked"
        report[name] = result
    report["smoke_status"] = (
        "passed"
        if all(report[name]["status"] == "passed" for name in ("text", "single_image"))
        else "blocked"
    )
    report["status"] = "partial"
    report["reason"] = (
        "Direct MLX only; HTTP integration and training residency are separate"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.hold_seconds:
        deadline = time.monotonic() + args.hold_seconds
        completed = 0
        prompt = apply_chat_template(processor, model.config, "Reply with exactly OK.")
        while time.monotonic() < deadline:
            result = collect_completion(
                stream_generate(
                    model,
                    processor,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=0,
                    logits_processors=[
                        lambda tokens, logits: logits.astype(mx.float32)
                    ],
                )
            )
            completed += 1
            report["hold"] = {
                "completed_requests": completed,
                "last_completion": result,
                "memory_after": snapshot(),
                "unix_time": time.time(),
            }
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
            time.sleep(1)
    return 0 if report["smoke_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
