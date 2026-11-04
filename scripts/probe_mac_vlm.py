"""S1 local-server probe. Uses only the standard library; never downloads weights."""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

INFER_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
TRAIN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def local_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError("Use a loopback HTTP URL without credentials")
    return value.rstrip("/")


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def memory_snapshot(server_pid: int | None = None) -> dict:
    result = {
        "vm_stat": command_output(["vm_stat"]),
        "swap": command_output(["sysctl", "vm.swapusage"]),
    }
    if server_pid:
        rss = command_output(["ps", "-o", "rss=", "-p", str(server_pid)])
        result["server_rss_bytes"] = int(rss) * 1024 if rss and rss.isdigit() else None
    return result


def red_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    pixels = b"".join(b"\x00" + b"\xff\x00\x00" * 64 for _ in range(64))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def processor_probe(path: Path) -> dict:
    try:
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Name the color of the image in one word.",
                    },
                    {"type": "image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[Image.open(io.BytesIO(red_png()))], return_tensors="pt"
        )
        token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        count = int((inputs["input_ids"] == token_id).sum())
        return {
            "status": "passed" if count > 0 else "blocked",
            "image_tokens": count,
            "prompt_tokens": inputs["input_ids"].numel(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "source": "local HF processor; compare against server usage",
        }
    except Exception as error:
        return {"status": "blocked", "error_type": type(error).__name__}


def request_json(
    base_url: str, route: str, payload: dict | None, timeout: float
) -> dict:
    headers = {"Content-Type": "application/json"}
    if os.environ.get("S1_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["S1_API_KEY"]
    request = urllib.request.Request(
        base_url + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def summarize_completion(response: dict) -> dict:
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    content = choice.get("message", {}).get("content") or ""
    entries = (choice.get("logprobs") or {}).get("content") or []
    valid = bool(entries) and all(
        isinstance(entry.get("logprob"), (int, float))
        and math.isfinite(entry["logprob"])
        and entry["logprob"] <= 0
        for entry in entries
    )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    usage_valid = all(
        isinstance(value, int) and value > 0
        for value in (prompt_tokens, completion_tokens)
    )
    return {
        "status": "passed"
        if content
        and choice.get("finish_reason") == "stop"
        and usage_valid
        and valid
        and len(entries) == completion_tokens
        else "blocked",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "logprob_tokens": len(entries),
        "finite_logprobs": valid,
        "logprobs_match_usage": bool(entries) and len(entries) == completion_tokens,
        "response_text": content[:200],
        "reasoning_content_present": bool(
            choice.get("message", {}).get("reasoning_content")
        ),
    }


def chat_probe(args: argparse.Namespace, with_image: bool) -> dict:
    prompt = "Name the color of the image in one word."
    content = [{"type": "text", "text": prompt}]
    if with_image:
        data = base64.b64encode(red_png()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}
        )
    payload = {
        "model": args.infer_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 1,
    }
    snapshots = [memory_snapshot(args.server_pid)]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.25):
            snapshots.append(memory_snapshot(args.server_pid))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.monotonic()
    try:
        result = summarize_completion(
            request_json(args.base_url, "/chat/completions", payload, args.timeout)
        )
        if with_image:
            result["red_image_recognized"] = (
                result["response_text"].strip().lower().strip(".!\"'") == "red"
            )
            if not result["red_image_recognized"]:
                result["status"] = "blocked"
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        result = {"status": "blocked", "error_type": type(error).__name__}
        if isinstance(error, urllib.error.HTTPError):
            result["http_status"] = error.code
    finally:
        stopped.set()
        sampler.join(timeout=4)
    snapshots.append(memory_snapshot(args.server_pid))
    rss = [
        snapshot["server_rss_bytes"]
        for snapshot in snapshots
        if snapshot.get("server_rss_bytes") is not None
    ]
    result.update(
        elapsed_seconds=round(time.monotonic() - started, 3),
        memory_before=snapshots[0],
        memory_after=snapshots[-1],
        server_peak_sampled_rss_bytes=max(rss) if rss else None,
        memory_samples=len(snapshots),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal", "cuda"), default="metal")
    parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--base-url", type=local_url, default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument("--infer-model", default=INFER_MODEL)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-mac", action="store_true")
    parser.add_argument(
        "--processor-path",
        type=Path,
        help="Local Qwen3-VL processor directory; never downloads weights",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s1/mac-probe.json")
    )
    args = parser.parse_args()
    if args.require_mac and platform.system() != "Darwin":
        parser.error("This probe requires macOS when --require-mac is set")
    if args.backend == "metal" and not args.sync:
        parser.error("The Metal probe requires --sync")
    packages = {}
    for package in (
        "torch",
        "transformers",
        "vllm",
        "vllm-metal",
        "mlx",
        "mlx-vlm",
        "deepseek-harness-sdk",
        "deepseek-harness-runtime-bin",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "sync": args.sync,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "physical_memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "packages": packages,
        "infer_model": args.infer_model,
        "train_model": TRAIN_MODEL,
        "memory_before": memory_snapshot(args.server_pid),
        "training_contract": {
            "method": "LoRA",
            "freeze_vit": True,
            "co_residency": "not_measured",
        },
        "dsh_prompt_tokens": None,
        "image_tokens": None,
        "reasoning_parser": "not_verified",
        "l1_max_model_len": {"candidate": 8192, "validated": False},
    }
    if args.preflight_only:
        report["status"] = "blocked"
        report["reason"] = (
            "Preflight only; no model request or training residency measured"
        )
    else:
        report["text"] = chat_probe(args, False)
        report["single_image"] = chat_probe(args, True)
        report["smoke_status"] = (
            "passed"
            if report["text"]["status"] == "passed"
            and report["single_image"]["status"] == "passed"
            else "blocked"
        )
        if args.processor_path:
            report["processor"] = processor_probe(args.processor_path)
            report["image_tokens"] = report["processor"].get("image_tokens")
        text_tokens = report["text"].get("prompt_tokens")
        image_tokens = report["single_image"].get("prompt_tokens")
        if isinstance(text_tokens, int) and isinstance(image_tokens, int):
            report["image_prompt_token_delta"] = image_tokens - text_tokens
        report["status"] = "blocked"
        report["reason"] = (
            "S1 requires exact processor image-token count, dsh prompt tokens, "
            "and bf16+LoRA co-residency; chat results alone are insufficient"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
