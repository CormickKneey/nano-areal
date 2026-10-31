"""Explicitly bound, text-only S4 arithmetic with differentiable cache replay."""

from __future__ import annotations

import time
from pathlib import Path
from types import MethodType

import torch
from safetensors import safe_open

from nano_areal.archon import model, qwen3vl, qwen3vl_lora, s4_metal, s4_operators
from nano_areal.archon import s4_replay as replay_source
from nano_areal.archon import s4_update as update_source
from nano_areal.archon.model import RMSNorm
from nano_areal.archon.s4_metal import differentiable_metal_rms
from nano_areal.archon.s4_operators import (
    affine_decode_weights,
    affine_weights,
    device_rope_frequencies,
    rounded_swiglu,
    splitk_prefill,
    splitkv_attention,
)
from nano_areal.archon.s4_replay import cached_text_logprobs
from nano_areal.archon.s4_update import trajectory_from_turn
from nano_areal.sync.policy import file_digest


def load_affine_projections(binding):
    quantized = {}
    for path in binding.manifest["inference_files"]:
        if not path.endswith(".safetensors"):
            continue
        with safe_open(path, framework="pt") as source:
            for name in source.keys():
                if (not name.startswith("language_model.model.")
                        or not name.endswith(".scales")):
                    continue
                prefix = name.removesuffix(".scales")
                mapped = prefix.replace("language_model.model.", "language_model.")
                mapped = mapped.replace(".self_attn.", ".attn.")
                if mapped in quantized:
                    raise ValueError("Duplicate bound affine projection")
                quantized[mapped] = tuple(
                    source.get_tensor(prefix + suffix)
                    for suffix in (".weight", ".scales", ".biases"))
    return quantized


class AlignedS4Runtime:
    def __init__(self, trainer, quantized, episodes):
        self.trainer = trainer
        self.history = [trajectory_from_turn(turn)[0]
                        for episode in episodes for turn in episode["turns"]]
        if not self.history:
            raise ValueError("Aligned runtime requires a declared complete group")
        self.index = None
        self.started = time.monotonic()
        self.prefill = True
        self.states = {}
        self.group_replay = False
        self.handles = []
        self.forwards = []
        self.activation_states = []
        self.original_response = trainer.response_logprobs
        self.original_frequency = trainer.model.language_model.rope.inv_freq.clone()
        self.source_sha256 = {str(path.resolve()): file_digest(path) for path in (
            Path(__file__), Path(model.__file__), Path(qwen3vl.__file__),
            Path(qwen3vl_lora.__file__), Path(s4_metal.__file__),
            Path(s4_operators.__file__), Path(replay_source.__file__),
            Path(update_source.__file__),
        )}
        try:
            self.install(quantized)
        except BaseException:
            self.close()
            raise

    def install(self, quantized):
        model = self.trainer.model
        expected = {name.removesuffix(".base") for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and name.startswith("language_model.layers.")}
        if set(quantized) != expected | {"language_model.embed_tokens"}:
            raise ValueError("Bound affine projection coverage differs from the model")
        for name, module in model.named_modules():
            mapped = name.removesuffix(".base")
            if isinstance(module, torch.nn.Linear) and mapped in quantized:
                if module.weight.requires_grad or module.bias is not None:
                    raise ValueError("Aligned projections must be frozen and bias-free")
                packed, scales, biases = [value.to(module.weight.device)
                                           for value in quantized[mapped]]
                precise = affine_weights(packed, scales, biases)
                if not torch.equal(precise.bfloat16(), module.weight):
                    raise ValueError("Bound affine weights differ from the native base")
                self.forwards.append((module, module.forward))
                module.weight.data = precise

                def forward(module, hidden, biases=biases):
                    if self.prefill:
                        weight = module.weight.bfloat16()
                        if hidden.shape[-2] >= 16:
                            return splitk_prefill(hidden, weight)
                        raise ValueError("Unqualified short prefill below 16 rows")
                    return affine_decode_weights(hidden, module.weight, biases)

                module.forward = MethodType(forward, module)
            if isinstance(module, RMSNorm):
                self.handles.append(module.register_forward_hook(
                    lambda module, inputs, output: differentiable_metal_rms(
                        inputs[0], module.weight, module.eps)))
        for layer in model.language_model.layers:
            state = {}
            self.activation_states.append(state)

            def capture(name, state):
                def hook(module, inputs, output):
                    state[name] = output
                return hook

            def activate(module, inputs, state=state):
                return (rounded_swiglu(state.pop("gate"), state.pop("up")),)

            for name in ("gate", "up"):
                projection = getattr(layer.mlp, name + "_proj")
                self.handles.append(projection.register_forward_hook(
                    capture(name, state)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(activate))
        device = model.language_model.embed_tokens.weight.device
        packed, scales, biases = [value.to(device)
                                  for value in quantized["language_model.embed_tokens"]]
        self.head = affine_weights(packed, scales, biases), biases.float()
        if not torch.equal(self.head[0].bfloat16(),
                           model.language_model.embed_tokens.weight):
            raise ValueError("Bound affine head differs from the native embedding")
        model.language_model.rope.inv_freq.copy_(device_rope_frequencies(
            model.config["text_config"], device))
        self.trainer.response_logprobs = self.response_logprobs
        self.trainer.select_turn = self.select_turn
        self.trainer.response_group_logprobs = self.response_group_logprobs
        torch.mps.synchronize()
        torch.mps.empty_cache()

    def select_turn(self, index):
        if not 0 <= index < len(self.history):
            raise ValueError("Turn is outside the declared bound group")
        self.index = index

    def start_update(self):
        if not any(
            index == len(self.history) - 1 for index, state in self.states.values()
        ):
            raise ValueError(
                "Complete the full preflight before starting the update budget"
            )
        self.states.clear()
        self.started = time.monotonic()

    def activate_segment(self, start, end, prompt):
        if time.monotonic() - self.started > 600:
            raise TimeoutError("Aligned replay exceeded its 600-second runtime budget")
        self.prefill = end == prompt

    def attention(self, query, key, value, **options):
        if self.prefill:
            return torch.nn.functional.scaled_dot_product_attention(
                query.float(), key.float(), value.float(), **options).to(query.dtype)
        return splitkv_attention(query, key, value, key.shape[-2] - 1)

    def project_head(self, hidden, start, end):
        return affine_decode_weights(hidden, *self.head)

    def response_group_logprobs(self, trajectories):
        if self.group_replay or len(trajectories) != len(self.history):
            raise ValueError("Group replay requires the complete declared "
                             "turn sequence")
        self.states.clear()
        self.group_replay = True
        try:
            results = []
            for index, trajectory in enumerate(trajectories):
                self.select_turn(index)
                results.append(self.response_logprobs(trajectory))
            return results
        finally:
            self.group_replay = False
            self.states.clear()

    def response_logprobs(self, trajectory):
        if self.index is None:
            raise ValueError("Select a bound turn before evaluating aligned arithmetic")
        expected = self.history[self.index]
        if (trajectory.pixel_values is not None
                or not torch.equal(trajectory.input_ids.cpu(), expected.input_ids)
                or not torch.equal(trajectory.response_mask.cpu(),
                                   expected.response_mask)):
            raise ValueError("Trajectory differs from the selected bound turn")
        enabled = tuple(module.enabled for module in self.trainer.model.modules()
                        if isinstance(module, qwen3vl.LoRALinear))
        differentiating = torch.is_grad_enabled()
        previous, state = self.states.get(enabled, (-1, {}))
        if (differentiating and not self.group_replay) or previous != self.index - 1:
            previous, state = -1, {}
        result = None
        for index in range(previous + 1, self.index + 1):
            result = cached_text_logprobs(
                self.trainer.model, self.history[index], project_head=self.project_head,
                state=state, attention_kernel=self.attention,
                activate_segment=self.activate_segment, checkpoint_layers=True,
                score_response=index == self.index, decode_chunk_size=16,
                checkpoint_head=True,
            )
        if differentiating and not self.group_replay:
            self.states.pop(enabled, None)
        else:
            self.states[enabled] = self.index, state
        return result

    def contract(self):
        return {"name": "native-s4-affine-cache-r5",
                "source_sha256": self.source_sha256,
                "prefix_block_size": 16, "decode_partition_size": 512,
                "decode_checkpoint_chunk_size": 16,
                "storage": "Frozen fp32 affine base and shared head biases; "
                           "bf16 activations; fp32 LoRA",
                "gradients": "One whole-group backward through shared history; "
                             "no detached prefix KV; checkpointed token head; "
                             "persistent KV blocks, concatenate inside non-reentrant "
                             "layer checkpoints; first-order Metal RMS",
                "scope": "Text-only declared group, prefill suffix >=16 rows",
                "max_runtime_seconds_per_phase": 600}

    def verify_sources(self):
        if self.source_sha256 != {name: file_digest(Path(name))
                                  for name in self.source_sha256}:
            raise ValueError("Bound training arithmetic source changed")

    def close(self):
        self.trainer.response_logprobs = self.original_response
        if hasattr(self.trainer, "select_turn"):
            del self.trainer.select_turn
        if hasattr(self.trainer, "response_group_logprobs"):
            del self.trainer.response_group_logprobs
        self.states.clear()
        for handle in self.handles:
            handle.remove()
        for module, forward in self.forwards:
            module.forward = forward
            module.weight.data = module.weight.data.bfloat16()
        for state in self.activation_states:
            state.clear()
        self.trainer.model.language_model.rope.inv_freq.copy_(self.original_frequency)
