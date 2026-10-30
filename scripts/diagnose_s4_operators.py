"""Unbound native operator diagnostics against untouched captured probabilities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import torch
from align_s4_base import hf_key
from probe_s4_training import complete_group, require_stopped_server, restore
from safetensors import safe_open

from nano_areal.archon import model as native_model
from nano_areal.archon import s4_metal as metal_source
from nano_areal.archon import s4_trace as trace_source
from nano_areal.archon.model import RMSNorm
from nano_areal.archon.qwen3vl_lora import NativeQwen3VLLoRATrainer
from nano_areal.archon.s4_metal import metal_rms
from nano_areal.archon.s4_operators import (
    affine_decode,
    affine_decode_weights,
    affine_weights,
    device_rope_frequencies,
    rounded_swiglu,
    splitk_prefill,
    splitkv_attention,
)
from nano_areal.archon.s4_replay import cached_text_logprobs
from nano_areal.archon.s4_trace import LayerTrace, native_trace_hooks
from nano_areal.archon.s4_update import trajectory_from_turn
from nano_areal.archon.state_dict import qwen3vl_hf_key
from nano_areal.sync.policy import PolicyBinding, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fused-rms", action="store_true")
    parser.add_argument("--metal-rms", action="store_true")
    parser.add_argument("--dense-projections", action="store_true")
    parser.add_argument("--splitk-prefill", action="store_true")
    parser.add_argument("--fp32-rope", action="store_true")
    parser.add_argument("--device-rope-init", action="store_true")
    parser.add_argument("--fused-swiglu", action="store_true")
    parser.add_argument("--rounded-swiglu", action="store_true")
    parser.add_argument("--dense-head", action="store_true")
    parser.add_argument("--prefill-head", action="store_true")
    parser.add_argument("--fp32-attention", action="store_true")
    parser.add_argument("--paged-attention", action="store_true")
    parser.add_argument("--resident-affine", action="store_true")
    parser.add_argument("--cached-execution", action="store_true")
    parser.add_argument("--prefix-reuse", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-reference", type=Path)
    args = parser.parse_args()
    if args.prefix_reuse and not args.cached_execution:
        parser.error("Prefix reuse requires cached execution")
    if args.splitk_prefill and not args.cached_execution:
        parser.error("Split-K prefill requires cached execution")
    if args.rounded_swiglu and args.fused_swiglu:
        parser.error("Choose exactly one SwiGLU diagnostic")
    if args.paged_attention and args.fp32_attention:
        parser.error("Choose exactly one attention diagnostic")
    if args.metal_rms and args.fused_rms:
        parser.error("Choose exactly one RMS diagnostic")
    if args.resident_affine and (not args.cached_execution or args.dense_projections
                                 or args.dense_head or args.prefill_head):
        parser.error("Resident affine requires fully affine cached execution")
    if not 0 < args.max_seconds <= 600:
        parser.error("Diagnostic deadline must be in (0, 600] seconds")
    if args.output.exists():
        parser.error("Diagnostic evidence must not be overwritten")
    if (args.trace_output or args.trace_reference) and not args.cached_execution:
        parser.error("Tracing requires cached execution")
    if args.trace_output and args.trace_output.exists():
        parser.error("Trace evidence must not be overwritten")
    trace_reference_hash = (file_digest(args.trace_reference)
                            if args.trace_reference else None)
    require_stopped_server()
    binding = PolicyBinding.read(args.policy)
    binding.verify_files()
    episodes = complete_group(args.group, binding)
    trainer = NativeQwen3VLLoRATrainer(Path(binding.manifest["training_base"]))
    restore(trainer, Path(binding.manifest["adapter"]))
    trainer.model.eval()
    if args.device_rope_init:
        rope = trainer.model.language_model.rope
        rope.inv_freq.copy_(device_rope_frequencies(
            trainer.model.config["text_config"], rope.inv_freq.device))
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
    device_quantized = {}

    def quantized_on_device(name, device):
        if name not in device_quantized:
            device_quantized[name] = tuple(
                value.to(device) for value in quantized[name])
        return device_quantized[name]

    handles = []
    resident_forwards = []
    original_rope = native_model.apply_rope
    original_attention = torch.nn.functional.scaled_dot_product_attention
    if args.paged_attention:
        def paged_attention(query, key, value, *arguments, **options):
            start = context["decode_start"]
            if start >= query.shape[-2]:
                return original_attention(
                    query.float(), key.float(), value.float(), *arguments, **options
                ).to(query.dtype)
            if args.cached_execution:
                return splitkv_attention(query, key, value, key.shape[-2] - 1)
            prefill = original_attention(
                query[..., :start, :].float(), key[..., :start, :].float(),
                value[..., :start, :].float(), is_causal=True,
            ).to(query.dtype)
            decode = splitkv_attention(query[..., start:, :], key, value, start)
            return torch.cat((prefill, decode), dim=-2)

        torch.nn.functional.scaled_dot_product_attention = paged_attention
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
                if args.splitk_prefill and hidden.shape[-2] >= 16:
                    return splitk_prefill(hidden, module.weight)
                return output
            if args.dense_projections:
                return output
            tensors = quantized_on_device(name, hidden.device)
            replacement = affine_decode(hidden[..., start:, :], *tensors)
            return torch.cat((output[..., :start, :], replacement), dim=-2)
        return hook

    def fused_rms(module, inputs, output):
        hidden = inputs[0].float()
        norm = (hidden.square().mean(-1, keepdim=True) + module.eps).rsqrt()
        return (hidden * norm * module.weight.float()).to(inputs[0].dtype)

    for name, module in trainer.model.named_modules():
        mapped = name.removesuffix(".base")
        if (args.resident_affine and isinstance(module, torch.nn.Linear)
                and mapped in quantized):
            packed, scales, biases = (tensor.to(module.weight.device)
                                       for tensor in quantized[mapped])
            module.weight.data = affine_weights(packed, scales, biases)
            resident_forwards.append((module, module.forward))

            def resident_forward(module, hidden, biases=biases):
                if context["decode_start"] >= hidden.shape[-2]:
                    weight = module.weight.bfloat16()
                    if args.splitk_prefill and hidden.shape[-2] >= 16:
                        return splitk_prefill(hidden, weight)
                    return torch.nn.functional.linear(hidden, weight)
                return affine_decode_weights(hidden, module.weight, biases)

            module.forward = MethodType(resident_forward, module)
        if (isinstance(module, torch.nn.Linear) and mapped in quantized
                and (not args.dense_projections or args.splitk_prefill)
                and not args.resident_affine):
            handles.append(module.register_forward_hook(projection(mapped)))
        if args.fused_rms and isinstance(module, RMSNorm):
            handles.append(module.register_forward_hook(fused_rms))
        if args.metal_rms and isinstance(module, RMSNorm):
            handles.append(module.register_forward_hook(
                lambda module, inputs, output: metal_rms(
                    inputs[0], module.weight, module.eps)))

    def capture_head(module, inputs, output):
        context["hidden"] = output

    def capture_activation(state, name):
        def hook(module, inputs, output):
            state[name] = output
        return hook

    def fused_activation(state):
        def hook(module, inputs):
            gate, up = state.pop("gate"), state.pop("up")
            if args.rounded_swiglu:
                return (rounded_swiglu(gate, up),)
            hidden = torch.nn.functional.silu(gate.float()) * up.float()
            return (hidden.to(gate.dtype),)
        return hook

    if args.fused_swiglu or args.rounded_swiglu:
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
    resident_head = None
    if args.resident_affine:
        packed, scales, biases = quantized_on_device(
            "language_model.embed_tokens", "mps")
        resident_head = affine_weights(packed, scales, biases), biases
        device_quantized.clear()
        del packed, scales, biases
        torch.mps.synchronize()
        torch.mps.empty_cache()
    evidence = [args.policy, args.group / "report.json", *[
        args.group / f"episode-{index}.json" for index in range(len(episodes))
    ]]
    evidence_hashes = {str(path): file_digest(path) for path in evidence}
    trace = (LayerTrace(evidence_hashes)
             if (args.trace_output or args.trace_reference) else None)
    if trace is not None:
        handles.extend(native_trace_hooks(trainer.model.language_model, trace))
    report = {"scope": "Unbound counterfactual; not policy acceptance or an update",
              "status": "running", "evidence_sha256": evidence_hashes,
              "source_sha256": {
                  str(path): file_digest(path) for path in (
                      Path(__file__), Path(native_model.__file__).with_name(
                          "s4_operators.py"
                      ), Path(native_model.__file__).with_name("s4_replay.py"),
                      Path(trace_source.__file__),
                      Path(metal_source.__file__),
                  )
              },
              "planned_turns": sum(len(item["turns"]) for item in episodes),
              "policy": binding.model, "group": str(args.group), "updates": 0,
              "operator": "fp32 affine weights plus bf16 four-value bias sum rounding",
              "fused_rms": args.fused_rms,
              "metal_rms": args.metal_rms,
              "dense_projections": args.dense_projections,
              "splitk_prefill": args.splitk_prefill,
              "fp32_rope": args.fp32_rope,
              "device_rope_init": args.device_rope_init,
              "fused_swiglu": args.fused_swiglu,
              "rounded_swiglu": args.rounded_swiglu,
              "dense_head": args.dense_head,
              "prefill_head": args.prefill_head,
              "fp32_attention": args.fp32_attention,
              "paged_attention": args.paged_attention,
              "resident_affine": args.resident_affine,
              "cached_execution": args.cached_execution,
              "prefix_reuse": args.prefix_reuse,
              "max_seconds": args.max_seconds,
              "turns": []}
    started = time.monotonic()
    cache_state = {} if args.prefix_reuse else None
    try:
        for episode in episodes:
            for item in episode["turns"]:
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("Diagnostic exceeded its forward deadline")
                trajectory, old = trajectory_from_turn(item)
                positions = torch.nonzero(
                    trajectory.response_mask[1:], as_tuple=True
                )[0]
                context["decode_start"] = int(positions[0]) + 1
                with torch.no_grad():
                    if args.cached_execution:
                        prompt = int(positions[0]) + 1

                        def before_segment(start, end):
                            if trace is not None:
                                trace.segment(len(report["turns"]), start, end, prompt)
                            if time.monotonic() - started >= args.max_seconds:
                                raise TimeoutError(
                                    "Diagnostic exceeded its forward deadline")
                            context["decode_start"] = end if end == prompt else 0

                        def project_head(hidden, start, end):
                            if resident_head is not None:
                                return affine_decode_weights(hidden, *resident_head)
                            if args.dense_head or (args.prefill_head and end == prompt):
                                return torch.nn.functional.linear(
                                    hidden,
                                    trainer.model.language_model.embed_tokens.weight)
                            tensors = quantized_on_device(
                                "language_model.embed_tokens", hidden.device)
                            return affine_decode(hidden, *tensors)

                        candidate = cached_text_logprobs(
                            trainer.model, trajectory, before_segment, project_head,
                            state=cache_state,
                            observe=trace.capture if trace is not None else None,
                        ).cpu()[0]
                    else:
                        dense = trainer.response_logprobs(trajectory).cpu()[0]
                        hidden = context.pop("hidden")[:, positions.to("mps")]
                        tensors = quantized_on_device(
                            "language_model.embed_tokens", "mps")
                        logits = affine_decode(hidden, *tensors).float()
                        labels = trajectory.input_ids[1:][positions].to("mps")
                        current = logits.log_softmax(-1).gather(
                            -1, labels[None, :, None]
                        ).flatten().cpu()
                        if args.prefill_head:
                            current = torch.cat((dense[:1], current[1:]))
                        candidate = dense if args.dense_head else current.cpu()
                        del hidden, tensors, logits, current, dense
                    error = (candidate - old[trajectory.response_mask]).abs()
                    if not bool(torch.isfinite(error).all()):
                        raise ValueError("Nonfinite diagnostic logprob difference")
                report["turns"].append({"tokens": error.numel(),
                                        "mae": float(error.mean()),
                                        "max": float(error.max()),
                                        "first_error": float(error[0]),
                                        "reused_tokens": (cache_state["reused_tokens"]
                                                          if cache_state else 0),
                                        "max_error_position": int(error.argmax())})
                context.pop("hidden", None)
                del candidate
                report["elapsed_seconds"] = time.monotonic() - started
                report["mae"] = sum(row["mae"] * row["tokens"]
                                    for row in report["turns"]) / sum(
                                        row["tokens"] for row in report["turns"])
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report["turns"][-1]), flush=True)
        binding.verify_files()
        if evidence_hashes != {str(path): file_digest(path) for path in evidence}:
            raise ValueError("Captured evidence changed during the diagnostic")
        if trace is not None:
            if args.trace_output:
                trace.save(args.trace_output)
                report["trace_sha256"] = file_digest(args.trace_output)
            if args.trace_reference:
                if file_digest(args.trace_reference) != trace_reference_hash:
                    raise ValueError("Trace reference changed during the diagnostic")
                report["layerwise"] = trace.compare(args.trace_reference)
                report["trace_reference_sha256"] = trace_reference_hash
        report["status"] = "diagnostic-complete"
        report["max_logprob_mae"] = 0.01
        report["within_bound"] = report["mae"] <= 0.01
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        native_model.apply_rope = original_rope
        torch.nn.functional.scaled_dot_product_attention = original_attention
        for handle in handles:
            handle.remove()
        for module, forward in resident_forwards:
            module.forward = forward
            module.weight.data = module.weight.data.bfloat16()
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("turns", "layerwise")}))


if __name__ == "__main__":
    main()
