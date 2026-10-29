"""Report whether this machine can run the Mac VLM smoke.

Prints one JSON object. Exit 0 means the probe itself ran. A non-Mac host, a
missing ``vllm-metal``, or a missing ``deepseek_harness`` import is listed
under ``blocked`` rather than treated as a passed smoke. ``--require-mac``
exits 2 when the host is not Darwin, for a later run on Mac.lan.

    uv run python scripts/probe_mac_vlm.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys


INFER_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
TRAIN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 Mac VLM smoke probe")
    parser.add_argument("--require-mac", action="store_true")
    args = parser.parse_args()

    system = platform.system()
    machine = platform.machine()
    blocked: list[str] = []
    if system != "Darwin":
        blocked.append(
            "vllm-metal single-image chat and unified-memory measurement require Mac.lan "
            f"(this host is {system} {machine})"
        )
    if shutil.which("vllm-metal") is None:
        blocked.append("vllm-metal is not on PATH")
    if importlib.util.find_spec("deepseek_harness") is None:
        blocked.append(
            "deepseek_harness is not installed, so a live sdk-minimal turn cannot be launched"
        )
    if blocked:
        blocked.append("4bit inference resident together with bf16+LoRA was not measured")

    report = {
        "probe": "s1-mac-vlm",
        "agent": "dsh",
        "profile": "sdk-minimal",
        "host": {"system": system, "machine": machine, "python": platform.python_version()},
        "infer_model": INFER_MODEL,
        "train_model": TRAIN_MODEL,
        "blocked": blocked,
        "ready_for_inference_smoke": not blocked,
        "suggested_command": (
            "uv run python train.py --backend metal --sync --agent dsh "
            f"--infer-model {INFER_MODEL} --train-model {TRAIN_MODEL} --loss grpo"
        ),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.require_mac and system != "Darwin":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
