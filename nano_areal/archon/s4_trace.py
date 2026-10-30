"""Bounded observational layer traces for unbound S4 cache diagnostics."""

from __future__ import annotations

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class LayerTrace:
    def __init__(self, evidence):
        self.evidence = evidence
        self.schedule = []
        self.tensors = {}
        self.shapes = {}
        self.prefix = None

    def segment(self, turn, start, end, prompt):
        self.prefix = None
        if end == prompt or start == prompt:
            if len(self.schedule) >= 32:
                raise ValueError("Trace exceeds the 32-segment diagnostic budget")
            phase = "prefill" if end == prompt else "decode"
            self.prefix = f"turn.{turn}.{phase}"
            self.schedule.append([turn, phase, start, end])

    def capture(self, name, value):
        if self.prefix is None:
            return
        key = f"{self.prefix}.{name}"
        if key in self.tensors:
            raise ValueError(f"Duplicate trace stage: {key}")
        self.shapes[key] = list(value.shape)
        self.tensors[key] = value[:, -1].detach().float().cpu().clone().contiguous()

    def save(self, path):
        if path.exists():
            raise ValueError("Trace evidence must not be overwritten")
        if not self.tensors:
            raise ValueError("Cannot save an empty trace")
        if any(not bool(torch.isfinite(value).all())
               for value in self.tensors.values()):
            raise ValueError("Cannot save a nonfinite trace")
        save_file(self.tensors, str(path), metadata={"contract": json.dumps({
            "scope": "Unbound last-token activations; not policy acceptance",
            "evidence": self.evidence, "schedule": self.schedule,
            "shapes": self.shapes,
        }, sort_keys=True)})

    def compare(self, path):
        with safe_open(str(path), framework="pt") as source:
            contract = json.loads(source.metadata()["contract"])
        expected_shapes = contract["shapes"].copy()
        current_shapes = self.shapes.copy()
        for shapes in (expected_shapes, current_shapes):
            for name, shape in shapes.items():
                if name.endswith(".norm"):
                    shapes[name] = [shape[0], 1, *shape[2:]]
        if (contract["evidence"] != self.evidence
                or contract["schedule"] != self.schedule
                or expected_shapes != current_shapes):
            raise ValueError("Trace evidence, schedule or stage shapes differ")
        reference = load_file(str(path))
        if reference.keys() != self.tensors.keys():
            raise ValueError("Trace stages differ")
        rows = []
        for name, current in self.tensors.items():
            expected = reference[name]
            if current.shape != expected.shape:
                raise ValueError("Trace tensor shapes differ")
            error = (current - expected).abs()
            if not bool(torch.isfinite(error).all()):
                raise ValueError("Nonfinite trace difference")
            rows.append({"stage": name, "mae": float(error.mean()),
                         "max": float(error.max()),
                         "reference_rms": float(expected.square().mean().sqrt()),
                         "different_fraction": float((error != 0).float().mean())})
        return rows


def trace_modules(model, mlx=False):
    modules = {"embedding": model.embed_tokens, "norm": model.norm}
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}"
        for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
            modules[f"{prefix}.{name}"] = getattr(layer, name)
        attention = layer.self_attn._inner if mlx else layer.attn
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
            modules[f"{prefix}.attn.{name}"] = getattr(attention, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            modules[f"{prefix}.mlp.{name}"] = getattr(layer.mlp, name)
    return modules


def native_trace_hooks(model, trace):
    handles = []
    for name, module in trace_modules(model).items():
        def capture(module, inputs, output, name=name):
            trace.capture(name, output)
            if name.endswith(("o_proj", "down_proj")):
                trace.capture(name + ".input", inputs[0])

        handles.append(module.register_forward_hook(capture))
    return handles


def mlx_trace_hooks(model, trace):
    import mlx.core as mx
    import numpy as np
    from vllm_metal.attention.impls import sdpa

    modules = {id(module): name for name, module in
               trace_modules(model, mlx=True).items()}
    classes = {type(module) for module in trace_modules(model, mlx=True).values()}
    originals = {kind: kind.__call__ for kind in classes}
    attention_names = {id(layer.self_attn._inner): f"layers.{index}.attn"
                       for index, layer in enumerate(model.layers)}
    original_prepare = sdpa.prepare_sdpa_qkv

    def capture(name, value):
        if trace.prefix is not None:
            row = torch.from_numpy(np.array(value[:, -1:].astype(mx.float32)))
            trace.capture(name, row)
            trace.shapes[f"{trace.prefix}.{name}"] = list(value.shape)

    def wrapper(original):
        def call(module, *arguments, **options):
            output = original(module, *arguments, **options)
            name = modules.get(id(module))
            if name is not None:
                capture(name, output)
                if name.endswith(("o_proj", "down_proj")):
                    capture(name + ".input", arguments[0])
            return output
        return call

    for kind, original in originals.items():
        kind.__call__ = wrapper(original)

    def prepare(inner, *arguments, **options):
        output = original_prepare(inner, *arguments, **options)
        name = attention_names.get(id(inner))
        if name is not None:
            for suffix, value in zip(("rope_query", "rope_key"), output[:2]):
                capture(f"{name}.{suffix}", value.transpose(0, 2, 1, 3))
        return output

    sdpa.prepare_sdpa_qkv = prepare

    def restore():
        for kind, original in originals.items():
            kind.__call__ = original
        sdpa.prepare_sdpa_qkv = original_prepare

    return restore
