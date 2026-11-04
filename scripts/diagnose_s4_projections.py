"""Compare affine projection arithmetic on untouched production trace inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diagnose_s4_trace import differences
from probe_s4_training import require_stopped_server
from safetensors import safe_open
from safetensors.torch import load_file

from nano_areal.archon import s4_operators
from nano_areal.archon.s4_operators import (
    affine_decode,
    affine_weights,
    prefill_split_count,
)
from nano_areal.sync.policy import PolicyBinding, file_digest


def projection_variants(hidden, weights, packed, scales, biases, rows):
    precise = affine_weights(packed, scales, biases)
    variants = {
        "dense_bf16": torch.nn.functional.linear(hidden, weights),
        "affine_fp32": torch.nn.functional.linear(hidden.float(), precise).bfloat16(),
        "affine_bias_rounding": affine_decode(hidden, packed, scales, biases),
    }
    if rows >= 16:
        partitions = prefill_split_count(rows, weights.shape[1], weights.shape[0])
        if partitions >= 32:
            raise ValueError("Only split reductions below 32 are supported")
        width = weights.shape[1] // partitions
        partials = [torch.nn.functional.linear(hidden[..., start:start + width],
                                              weights[:, start:start + width])
                    for start in range(0, weights.shape[1], width)]
        lanes = []
        for lane in range(min(8, partitions)):
            total = partials[lane]
            for index in range(lane + 8, partitions, 8):
                total = total + partials[index]
            lanes.append(total)
        result = lanes[0]
        for total in lanes[1:]:
            result = result + total
        variants["splitk_bf16"] = result
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.layers <= 36:
        parser.error("Use new evidence and between 1 and 36 layers")
    if not 0 < args.max_seconds <= 600:
        parser.error("Diagnostic deadline must be in (0, 600] seconds")
    require_stopped_server()
    binding = PolicyBinding.read(args.policy)
    binding.verify_files()
    trace_hash = file_digest(args.trace)
    with safe_open(str(args.trace), framework="pt") as source:
        contract = json.loads(source.metadata()["contract"])
    if (not 1 <= len(contract["schedule"]) <= 32
            or contract["evidence"].get(str(args.policy)) != file_digest(args.policy)):
        raise ValueError("Trace policy or schedule differs from the supplied evidence")
    for name, digest in contract["evidence"].items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Original trace evidence changed")
    trace = load_file(str(args.trace))
    report = {"scope": "Unbound shared-input projections; not policy acceptance",
              "status": "running", "updates": 0, "device": args.device,
              "trace_sha256": trace_hash, "evidence_sha256": contract["evidence"],
              "source_sha256": {str(path): file_digest(path) for path in (
                  Path(__file__), Path(s4_operators.__file__),
              )}, "layers": args.layers, "max_seconds": args.max_seconds, "rows": []}
    started = time.monotonic()
    try:
        for path in binding.manifest["inference_files"]:
            if not path.endswith(".safetensors"):
                continue
            with safe_open(path, framework="pt") as source:
                for name in source.keys():
                    if (not name.startswith("language_model.model.layers.")
                            or not name.endswith(".scales")):
                        continue
                    base = name.removeprefix("language_model.model.").removesuffix(
                        ".scales")
                    if int(base.split(".")[1]) >= args.layers:
                        continue
                    stage = base.replace(".self_attn.", ".attn.")
                    prefix = name.removesuffix(".scales")
                    packed, scales, biases = [source.get_tensor(prefix + suffix).to(
                        args.device) for suffix in (".weight", ".scales", ".biases")]
                    weights = affine_weights(packed, scales, biases).bfloat16()
                    layer, operation = stage.rsplit(".", 1)
                    if operation in ("q_proj", "k_proj", "v_proj"):
                        input_stage = layer.removesuffix(".attn") + ".input_layernorm"
                    elif operation in ("gate_proj", "up_proj"):
                        input_stage = layer.removesuffix(".mlp") + (
                            ".post_attention_layernorm")
                    else:
                        input_stage = stage + ".input"
                    for turn, phase, start, end in contract["schedule"]:
                        if time.monotonic() - started > args.max_seconds:
                            raise TimeoutError("Shared projection deadline exceeded")
                        segment = f"turn.{turn}.{phase}."
                        hidden = trace[segment + input_stage].to(args.device).bfloat16()
                        expected = trace[segment + stage]
                        count = contract["shapes"][segment + stage][1]
                        variants = projection_variants(
                            hidden, weights, packed, scales, biases, count)
                        report["rows"].append({"stage": segment + stage, "rows": count,
                                               "operators": {key: differences(
                                                   value, expected)
                                                   for key, value in variants.items()}})
                    del packed, scales, biases, weights, variants
            report["elapsed_seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        binding.verify_files()
        if trace_hash != file_digest(args.trace):
            raise ValueError("Trace changed during shared projection evaluation")
        if contract["evidence"] != {name: file_digest(Path(name))
                                    for name in contract["evidence"]}:
            raise ValueError("Original evidence changed during evaluation")
        report["status"] = "diagnostic-complete"
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
