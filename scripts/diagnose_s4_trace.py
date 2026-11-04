"""Isolate native RoPE and activation arithmetic on production trace inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from probe_s4_training import require_stopped_server
from safetensors import safe_open
from safetensors.torch import load_file

from nano_areal.archon import model as model_source
from nano_areal.archon import qwen3vl as rope_source
from nano_areal.archon import s4_metal as metal_source
from nano_areal.archon import s4_operators as operator_source
from nano_areal.archon.model import apply_rope
from nano_areal.archon.qwen3vl import InterleavedMRoPE
from nano_areal.archon.s4_metal import metal_rms
from nano_areal.archon.s4_operators import device_rope_frequencies, rounded_swiglu
from nano_areal.sync.policy import PolicyBinding, file_digest


def differences(current, expected):
    if current.shape != expected.shape:
        raise ValueError("Isolated operator tensor shapes differ")
    error = (current.float().cpu() - expected).abs()
    if not bool(torch.isfinite(error).all()):
        raise ValueError("Nonfinite isolated operator difference")
    return {"mae": float(error.mean()), "max": float(error.max()),
            "different_fraction": float((error != 0).float().mean())}


def rms_differences(hidden, weight, expected, epsilon):
    values = hidden.float()
    normalized = values * (values.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    variants = {
        "native_rms": differences(normalized.bfloat16() * weight, expected),
        "single_round": differences((normalized * weight.float()).bfloat16(), expected),
    }
    if hidden.device.type == "mps" and not hidden.requires_grad:
        variants["metal_reduction"] = differences(
            metal_rms(hidden, weight, epsilon), expected)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Diagnostic evidence must not be overwritten")
    require_stopped_server()
    binding = PolicyBinding.read(args.policy)
    binding.verify_files()
    with safe_open(str(args.trace), framework="pt") as source:
        contract = json.loads(source.metadata()["contract"])
    if (not 1 <= len(contract["schedule"]) <= 32
            or contract["evidence"].get(str(args.policy)) != file_digest(args.policy)):
        raise ValueError("Trace policy or schedule differs from the supplied evidence")
    for name, digest in contract["evidence"].items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Original trace evidence changed")
    trace_hash = file_digest(args.trace)
    tensors = load_file(str(args.trace))
    config = json.loads((Path(binding.manifest["training_base"]) / "config.json"
                         ).read_text())["text_config"]
    norm_weights = {}
    for path in binding.manifest["inference_files"]:
        if not path.endswith(".safetensors"):
            continue
        with safe_open(path, framework="pt") as source:
            for name in source.keys():
                if name.startswith("language_model.model.") and name.endswith((
                    "layernorm.weight", "q_norm.weight", "k_norm.weight",
                )):
                    norm_weights[name.removeprefix("language_model.model.")] = (
                        source.get_tensor(name).to(args.device))
    rope = InterleavedMRoPE(config).to(args.device)
    device_rope = InterleavedMRoPE(config).to(args.device)
    device_rope.inv_freq.copy_(device_rope_frequencies(config, args.device))
    rows = []
    with torch.no_grad():
        for turn, phase, start, end in contract["schedule"]:
            prefix = f"turn.{turn}.{phase}"
            positions = torch.full((3, 1, 1), end - 1, device=args.device)
            cos, sin = rope(positions, torch.bfloat16)
            cos32, sin32 = rope(positions, torch.float32)
            device_cos, device_sin = device_rope(positions, torch.bfloat16)
            hidden = tensors[f"{prefix}.embedding"].to(args.device).bfloat16()
            for index in range(config["num_hidden_layers"]):
                stage = f"{prefix}.layers.{index}"
                gate = tensors[f"{stage}.mlp.gate_proj"].to(args.device).bfloat16()
                up = tensors[f"{stage}.mlp.up_proj"].to(args.device).bfloat16()
                target = tensors[f"{stage}.mlp.down_proj.input"]
                activation = {
                    "native_silu": torch.nn.functional.silu(gate) * up,
                    "rounded_sigmoid": (gate * torch.sigmoid(gate)) * up,
                    "single_round": (torch.nn.functional.silu(gate.float())
                                     * up.float()).bfloat16(),
                    "metal_sigmoid": rounded_swiglu(gate, up),
                }
                query, key = (tensors[f"{stage}.attn.{name}_norm"].to(
                    args.device).bfloat16().unsqueeze(1) for name in ("q", "k"))
                variants = {
                    "native_rope": apply_rope(query, key, cos, sin),
                    "device_init_rope": apply_rope(query, key, device_cos, device_sin),
                    "fp32_products": apply_rope(query.float(), key.float(), cos, sin),
                    "fp32_products_trig": apply_rope(
                        query.float(), key.float(), cos32, sin32),
                }
                row = {"stage": stage, "prefill_rows": end - start,
                       "activation": {name: differences(value, target)
                                      for name, value in activation.items()},
                       "rope": {}, "rms": {}}
                for name, values in (
                    ("input_layernorm", hidden),
                    ("post_attention_layernorm", hidden + tensors[
                        f"{stage}.attn.o_proj"].to(args.device).bfloat16()),
                ):
                    row["rms"][name] = rms_differences(
                        values, norm_weights[f"layers.{index}.{name}.weight"],
                        tensors[f"{stage}.{name}"], config["rms_norm_eps"])
                for name in ("q", "k"):
                    expected = tensors[f"{stage}.attn.{name}_norm"]
                    values = tensors[f"{stage}.attn.{name}_proj"].to(
                        args.device).bfloat16().reshape(expected.shape)
                    row["rms"][name + "_norm"] = rms_differences(
                        values, norm_weights[
                            f"layers.{index}.self_attn.{name}_norm.weight"],
                        expected, config["rms_norm_eps"])
                hidden = (hidden + tensors[f"{stage}.attn.o_proj"].to(
                    args.device).bfloat16()) + tensors[f"{stage}.mlp"].to(
                        args.device).bfloat16()
                for name, values in variants.items():
                    row["rope"][name] = {
                        side: differences(value.bfloat16().squeeze(1),
                                          tensors[f"{stage}.attn.rope_{side}"])
                        for side, value in zip(("query", "key"), values)
                    }
                rows.append(row)
    binding.verify_files()
    if trace_hash != file_digest(args.trace):
        raise ValueError("Trace changed during isolated evaluation")
    if contract["evidence"] != {name: file_digest(Path(name))
                                for name in contract["evidence"]}:
        raise ValueError("Original trace evidence changed during isolated evaluation")
    report = {"scope": "Unbound shared-input operator isolation; not policy acceptance",
              "status": "diagnostic-complete", "updates": 0,
              "device": args.device, "trace_sha256": trace_hash,
              "evidence_sha256": contract["evidence"],
              "source_sha256": {str(path): file_digest(path) for path in (
                  Path(__file__), Path(model_source.__file__),
                  Path(rope_source.__file__), Path(operator_source.__file__),
                  Path(metal_source.__file__),
              )},
              "rows": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
