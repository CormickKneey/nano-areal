"""Unbound native operator diagnostics against untouched captured probabilities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from align_s4_base import hf_key
from probe_s4_training import complete_group, require_stopped_server, restore
from safetensors import safe_open

from nano_areal.archon import model as native_model
from nano_areal.archon.model import RMSNorm
from nano_areal.archon.qwen3vl_lora import NativeQwen3VLLoRATrainer
from nano_areal.archon.s4_operators import affine_decode
from nano_areal.archon.s4_update import trajectory_from_turn
from nano_areal.archon.state_dict import qwen3vl_hf_key
from nano_areal.sync.policy import PolicyBinding, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fused-rms", action="store_true")
    parser.add_argument("--dense-projections", action="store_true")
    parser.add_argument("--fp32-rope", action="store_true")
    parser.add_argument("--fused-swiglu", action="store_true")
    parser.add_argument("--dense-head", action="store_true")
    parser.add_argument("--prefill-head", action="store_true")
    parser.add_argument("--fp32-attention", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Diagnostic evidence must not be overwritten")
    require_stopped_server()
    binding = PolicyBinding.read(args.policy)
    binding.verify_files()
    episodes = complete_group(args.group, binding)
    trainer = NativeQwen3VLLoRATrainer(Path(binding.manifest["training_base"]))
    restore(trainer, Path(binding.manifest["adapter"]))
    trainer.model.eval()
    quantized = {}
    source = {Path(name) for name in binding.manifest["inference_files"]
              if name.endswith(".safetensors")}
    for shard in sorted(source):
        with safe_open(str(shard), framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if (not key.startswith("language_model.model.")
                        or not key.endswith(".scales")):
                    continue
                prefix = key.removesuffix(".scales")
                mapped = qwen3vl_hf_key(hf_key(prefix + ".weight"))
                quantized[mapped.removesuffix(".weight")] = tuple(
                    weights.get_tensor(prefix + suffix)
                    for suffix in (".weight", ".scales", ".biases")
                )
    context = {}
    handles = []
    original_rope = native_model.apply_rope
    original_attention = torch.nn.functional.scaled_dot_product_attention
    if args.fp32_attention:
        def attention(query, key, value, *arguments, **options):
            output = original_attention(
                query.float(), key.float(), value.float(), *arguments, **options
            )
            return output.to(query.dtype)

        torch.nn.functional.scaled_dot_product_attention = attention
    if args.fp32_rope:
        def apply_rope(query, key, cos, sin):
            rotated = original_rope(query.float(), key.float(), cos, sin)
            return tuple(value.to(query.dtype) for value in rotated)

        native_model.apply_rope = apply_rope
        handles.append(trainer.model.language_model.rope.register_forward_hook(
            lambda module, inputs, output: module.forward(inputs[0], torch.float32)
        ))

    def projection(name):
        def hook(module, inputs, output):
            hidden = inputs[0]
            start = context["decode_start"]
            if start >= hidden.shape[-2]:
                return output
            tensors = [value.to(hidden.device) for value in quantized[name]]
            replacement = affine_decode(hidden[..., start:, :], *tensors)
            return torch.cat((output[..., :start, :], replacement), dim=-2)
        return hook

    def fused_rms(module, inputs, output):
        hidden = inputs[0].float()
        norm = (hidden.square().mean(-1, keepdim=True) + module.eps).rsqrt()
        return (hidden * norm * module.weight.float()).to(inputs[0].dtype)

    for name, module in trainer.model.named_modules():
        mapped = name.removesuffix(".base")
        if (isinstance(module, torch.nn.Linear) and mapped in quantized
                and not args.dense_projections):
            handles.append(module.register_forward_hook(projection(mapped)))
        if args.fused_rms and isinstance(module, RMSNorm):
            handles.append(module.register_forward_hook(fused_rms))

    def capture_head(module, inputs, output):
        context["hidden"] = output

    def capture_activation(state, name):
        def hook(module, inputs, output):
            state[name] = output
        return hook

    def fused_activation(state):
        def hook(module, inputs):
            gate, up = state.pop("gate"), state.pop("up")
            hidden = torch.nn.functional.silu(gate.float()) * up.float()
            return (hidden.to(gate.dtype),)
        return hook

    if args.fused_swiglu:
        for layer in trainer.model.language_model.layers:
            state = {}
            for name in ("gate", "up"):
                handles.append(getattr(layer.mlp, name + "_proj").register_forward_hook(
                    capture_activation(state, name)
                ))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                fused_activation(state)
            ))

    handles.append(trainer.model.language_model.norm.register_forward_hook(
        capture_head
    ))
    evidence = [args.policy, args.group / "report.json", *[
        args.group / f"episode-{index}.json" for index in range(len(episodes))
    ]]
    evidence_hashes = {str(path): file_digest(path) for path in evidence}
    report = {"scope": "Unbound counterfactual; not policy acceptance or an update",
              "status": "running", "evidence_sha256": evidence_hashes,
              "source_sha256": {
                  str(path): file_digest(path) for path in (
                      Path(__file__), Path(native_model.__file__).with_name(
                          "s4_operators.py"
                      ),
                  )
              },
              "planned_turns": sum(len(item["turns"]) for item in episodes),
              "policy": binding.model, "group": str(args.group), "updates": 0,
              "operator": "fp32 affine weights plus bf16 four-value bias sum rounding",
              "fused_rms": args.fused_rms,
              "dense_projections": args.dense_projections,
              "fp32_rope": args.fp32_rope,
              "fused_swiglu": args.fused_swiglu,
              "dense_head": args.dense_head,
              "prefill_head": args.prefill_head,
              "fp32_attention": args.fp32_attention,
              "turns": []}
    started = time.monotonic()
    try:
        for episode in episodes:
            for item in episode["turns"]:
                trajectory, old = trajectory_from_turn(item)
                positions = torch.nonzero(
                    trajectory.response_mask[1:], as_tuple=True
                )[0]
                context["decode_start"] = int(positions[0]) + 1
                with torch.no_grad():
                    dense = trainer.response_logprobs(trajectory).cpu()[0]
                    hidden = context.pop("hidden")[:, positions.to("mps")]
                    tensors = [value.to("mps") for value in quantized[
                        "language_model.embed_tokens"]]
                    logits = affine_decode(hidden, *tensors).float()
                    labels = trajectory.input_ids[1:][positions].to("mps")
                    current = logits.log_softmax(-1).gather(
                        -1, labels[None, :, None]
                    ).flatten().cpu()
                    if args.prefill_head:
                        current = torch.cat((dense[:1], current[1:]))
                    candidate = dense if args.dense_head else current.cpu()
                    error = (candidate - old[trajectory.response_mask]).abs()
                    if not bool(torch.isfinite(error).all()):
                        raise ValueError("Nonfinite diagnostic logprob difference")
                report["turns"].append({"tokens": error.numel(),
                                        "mae": float(error.mean()),
                                        "max": float(error.max()),
                                        "first_error": float(error[0]),
                                        "max_error_position": int(error.argmax())})
                del hidden, tensors, logits, current, dense, candidate
                report["elapsed_seconds"] = time.monotonic() - started
                report["mae"] = sum(row["mae"] * row["tokens"]
                                    for row in report["turns"]) / sum(
                                        row["tokens"] for row in report["turns"])
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report["turns"][-1]), flush=True)
        binding.verify_files()
        if evidence_hashes != {str(path): file_digest(path) for path in evidence}:
            raise ValueError("Captured evidence changed during the diagnostic")
        report["status"] = "diagnostic-complete"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        native_model.apply_rope = original_rope
        torch.nn.functional.scaled_dot_product_attention = original_attention
        for handle in handles:
            handle.remove()
    print(json.dumps({key: value for key, value in report.items() if key != "turns"}))


if __name__ == "__main__":
    main()
