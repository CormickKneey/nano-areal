"""Observe selected production attention calls, then isolate native arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diagnose_s4_replay import main as replay
from diagnose_s4_trace import differences
from safetensors.torch import save_file

from nano_areal.archon import s4_operators
from nano_areal.archon.s4_operators import splitkv_attention
from nano_areal.sync.policy import file_digest


def attention_variants(query, key, value, expected, decode):
    expanded_key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    expanded_value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    variants = {}
    for name, dtype in (("native_sdpa", torch.bfloat16), ("fp32_sdpa", torch.float32)):
        result = torch.nn.functional.scaled_dot_product_attention(
            query.to(dtype), expanded_key.to(dtype), expanded_value.to(dtype))
        variants[name] = differences(result.bfloat16(), expected)
    if decode:
        variants["splitkv_bf16"] = differences(splitkv_attention(
            query, expanded_key, expanded_value, key.shape[-2] - 1), expected)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args, remaining = parser.parse_known_args()
    tensor_path = args.attention_output.with_suffix(".safetensors")
    if args.attention_output.exists() or tensor_path.exists():
        parser.error("Attention evidence must not be overwritten")
    import mlx.core as mx
    import numpy as np
    from vllm_metal.metal import get_ops

    ops = get_ops()
    original = ops.paged_attention_primitive
    calls = 0
    captured = {}
    records = []

    def capture(*arguments, **options):
        nonlocal calls
        index = calls
        calls += 1
        output = original(*arguments, **options)
        if index >= 72 or index % 36 not in (0, 17, 35):
            return output
        query, keys, values = arguments[:3]
        end = int(arguments[7].item())
        if (arguments[3] != 8 or query.shape[1:] != (32, 128)
                or keys.shape[1:] != (16, 8, 128)
                or arguments[6].tolist() != [list(range(keys.shape[0]))]):
            raise ValueError("Attention isolation requires the canonical layout")
        mx.eval(output, keys, values)
        arrays = {"query": query[-1:].transpose(1, 0, 2)[None],
                  "key": keys.reshape(-1, 8, 128)[:end].transpose(1, 0, 2)[None],
                  "value": values.reshape(-1, 8, 128)[:end].transpose(1, 0, 2)[None],
                  "output": output[-1:].transpose(1, 0, 2)[None]}
        captured.update({f"call.{index}.{name}": torch.from_numpy(
            np.array(value.astype(mx.float32))).contiguous()
                         for name, value in arrays.items()})
        records.append({"call": index, "layer": index % 36, "end": end,
                        "query_rows": query.shape[0], "decode": index >= 36})
        return output

    original_arguments = sys.argv
    ops.paged_attention_primitive = capture
    try:
        sys.argv = [original_arguments[0], *remaining,
                    "--execution", "paged", "--prefix-reuse"]
        replay()
    finally:
        ops.paged_attention_primitive = original
        sys.argv = original_arguments
    if len(records) != 6:
        raise ValueError("Expected three layers at prefill and first decode")
    replay_output = Path(remaining[remaining.index("--output") + 1])
    replay_report = json.loads(replay_output.read_text())
    if (replay_report["status"] != "diagnostic-complete"
            or replay_report["planned_turns"] != len(replay_report["turns"])):
        raise ValueError("Attention capture requires a complete group replay")
    contract = {"scope": "Unbound shared-input attention; not policy acceptance",
                "evidence_sha256": replay_report["evidence_sha256"],
                "source_sha256": {**replay_report["source_sha256"],
                                   str(Path(__file__)): file_digest(Path(__file__))},
                "replay_sha256": file_digest(replay_output), "records": records}
    save_file(captured, str(tensor_path), metadata={"contract": json.dumps(contract)})
    with torch.no_grad():
        for row in records:
            values = {name: captured[f"call.{row['call']}.{name}"].to(
                args.device).bfloat16() for name in ("query", "key", "value")}
            expected = captured[f"call.{row['call']}.output"]
            row["operators"] = attention_variants(**values, expected=expected,
                                                  decode=row["decode"])
    report = {**contract, "status": "diagnostic-complete", "updates": 0,
              "device": args.device, "trace_sha256": file_digest(tensor_path),
              "operator_source_sha256": file_digest(Path(s4_operators.__file__))}
    for name, digest in contract["evidence_sha256"].items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Original evidence changed during attention isolation")
    args.attention_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
