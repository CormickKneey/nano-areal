"""Opt-in text-only cached forward reference, never installed by the trainer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from nano_areal.archon import model as native_model


@dataclass(frozen=True)
class CacheBlock:
    key: torch.Tensor
    value: torch.Tensor
    previous: CacheBlock | None = None

    @property
    def length(self):
        length = 0
        block = self
        while block is not None:
            length += block.key.shape[-2]
            block = block.previous
        return length

    def prefix(self, length):
        block = self
        end = self.length
        while block is not None:
            start = end - block.key.shape[-2]
            if length > start:
                if length == end:
                    return block
                return CacheBlock(block.key[..., :length - start, :],
                                  block.value[..., :length - start, :],
                                  block.previous)
            block, end = block.previous, start
        return None

    def tensors(self):
        blocks = []
        block = self
        while block is not None:
            blocks.append(block)
            block = block.previous
        return ([block.key for block in reversed(blocks)],
                [block.value for block in reversed(blocks)])


def cached_text_logprobs(
    model, trajectory, before_segment=None, project_head=None, state=None,
    observe=None,
    attention_kernel=None, activate_segment=None, checkpoint_layers=False,
    score_response=True, decode_chunk_size=1, checkpoint_head=False,
    retain_decode_mlp=False, batch_decode=False,
    retained_mlp_projections=("gate_proj", "up_proj", "down_proj"),
    retain_decode_attention=False,
):
    if not 1 <= decode_chunk_size <= 16:
        raise ValueError("Decode checkpoint chunks must contain 1 to 16 tokens")
    if not set(retained_mlp_projections) <= {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("Unknown retained MLP projection")
    if trajectory.pixel_values is not None:
        raise ValueError("Cached diagnostic supports text only")
    ids = trajectory.input_ids
    mask = trajectory.response_mask.bool()
    positions = torch.nonzero(mask, as_tuple=True)[0]
    if (ids.ndim != 1 or mask.shape != ids.shape or positions.numel() == 0
            or int(positions[0]) == 0 or not bool(mask[int(positions[0]):].all())):
        raise ValueError("Cached diagnostic requires a contiguous response suffix")
    prompt = int(positions[0])
    device = model.language_model.embed_tokens.weight.device
    previous = state.get("input_ids", []) if state is not None else []
    matched = 0
    for left, right in zip(previous[:prompt - 1], ids[:prompt - 1].tolist()):
        if left != right:
            break
        matched += 1
    reused = matched // 16 * 16
    cache = ([block.prefix(reused) for block in state["cache"]] if reused else
             [None] * len(model.language_model.layers))
    ids = ids.to(device).unsqueeze(0)
    scores = []
    segments = [(reused, prompt), *[
        (index, min(index + decode_chunk_size, ids.shape[1] - 1))
        for index in range(prompt, ids.shape[1] - 1, decode_chunk_size)]]
    for start, end in segments:
        if before_segment is not None:
            before_segment(start, end)
        hidden = model.language_model.embed_tokens(ids[:, start:end])
        position_ids = torch.arange(start, end, device=device)[None, None].expand(
            3, 1, -1)
        cos, sin = model.language_model.rope(position_ids, hidden.dtype)
        for index, layer in enumerate(model.language_model.layers):
            def forward_layer(hidden, history, layer=layer, index=index,
                              start=start, end=end, cos=cos, sin=sin):
                old_key, old_value = None, None
                if history is not None:
                    keys, values = history.tensors()
                    old_key = torch.cat(keys, dim=-2)
                    old_value = torch.cat(values, dim=-2)
                outputs, new_keys, new_values = [], [], []
                fragments = ([(0, end - start)] if end == prompt or batch_decode else
                             [(offset, offset + 1) for offset in range(end - start)])
                for offset, stop in fragments:
                    fragment_start, fragment_end = start + offset, start + stop
                    if activate_segment is not None:
                        activate_segment(fragment_start, fragment_end, prompt)
                    fragment = hidden[:, offset:stop]
                    length = stop - offset
                    attention = layer.attn
                    normalized = layer.input_layernorm(fragment)
                    query = attention.q_norm(attention.q_proj(normalized).view(
                        1, length, attention.n_heads, attention.head_dim))
                    key = attention.k_norm(attention.k_proj(normalized).view(
                        1, length, attention.n_kv_heads, attention.head_dim))
                    value = attention.v_proj(normalized).view(
                        1, length, attention.n_kv_heads, attention.head_dim)
                    query, key = native_model.apply_rope(
                        query, key, cos[:, offset:stop], sin[:, offset:stop])
                    if observe is not None:
                        observe(f"layers.{index}.attn.rope_query", query)
                        observe(f"layers.{index}.attn.rope_key", key)
                    key, value = key.transpose(1, 2), value.transpose(1, 2)
                    new_keys.append(key)
                    new_values.append(value)
                    if old_key is not None:
                        key = torch.cat((old_key, key), dim=-2)
                        value = torch.cat((old_value, value), dim=-2)
                    attention_mask = None
                    if fragment_start and length > 1:
                        attention_mask = (
                            torch.arange(fragment_end, device=device)[None, :]
                            <= torch.arange(fragment_start, fragment_end,
                                            device=device)[:, None])
                    kernel = attention_kernel or F.scaled_dot_product_attention
                    attended = kernel(
                        query.transpose(1, 2),
                        key.repeat_interleave(attention.n_rep, dim=1),
                        value.repeat_interleave(attention.n_rep, dim=1),
                        attn_mask=attention_mask, is_causal=fragment_start == 0,
                    ).transpose(1, 2).contiguous().view(1, length, -1)
                    fragment = fragment + attention.o_proj(attended)
                    fragment = fragment + layer.mlp(
                        layer.post_attention_layernorm(fragment))
                    outputs.append(fragment)
                    old_key, old_value = key, value
                return (torch.cat(outputs, dim=1), torch.cat(new_keys, dim=-2),
                        torch.cat(new_values, dim=-2))

            history = cache[index]
            if checkpoint_layers and model.training and torch.is_grad_enabled():
                options = {}
                if (retain_decode_mlp or retain_decode_attention) and end != prompt:
                    retained_weights = ({
                        getattr(layer.mlp, name).weight.untyped_storage().data_ptr()
                        for name in retained_mlp_projections}
                        if retain_decode_mlp else set())

                    def policy(context, operation, *arguments,
                               retained_weights=retained_weights, **keywords):
                        if (retain_decode_attention and operation ==
                                torch.ops.nano_areal.tokenwise_splitkv_forward.default):
                            return CheckpointPolicy.MUST_SAVE
                        if (operation in (torch.ops.aten.mm.default,
                                          torch.ops.aten.linear.default)
                                and arguments[0].numel() == arguments[0].shape[-1]
                                and arguments[1].untyped_storage().data_ptr()
                                in retained_weights):
                            return CheckpointPolicy.MUST_SAVE
                        return CheckpointPolicy.PREFER_RECOMPUTE

                    options["context_fn"] = partial(
                        create_selective_checkpoint_contexts, policy)
                hidden, key, value = checkpoint(
                    forward_layer, hidden, history, use_reentrant=False, **options)
            else:
                hidden, key, value = forward_layer(hidden, history)
            cache[index] = CacheBlock(key, value, history)
        if not score_response:
            continue
        offsets = [end - start - 1] if end == prompt else range(end - start)
        for offset in offsets:
            def score(hidden, target, position=start + offset):
                final_hidden = model.language_model.norm(hidden)
                if project_head is None:
                    weight = (model.language_model.embed_tokens.weight if model.tied
                              else model.lm_head.weight)
                    logits = F.linear(final_hidden, weight)
                else:
                    logits = project_head(final_hidden, position, position + 1)
                return logits.float().log_softmax(-1).gather(-1, target).squeeze(-1)

            arguments = (hidden[:, offset:offset + 1],
                         ids[:, start + offset + 1:start + offset + 2, None])
            if checkpoint_head and model.training and torch.is_grad_enabled():
                scores.append(checkpoint(score, *arguments, use_reentrant=False))
            else:
                scores.append(score(*arguments))
    if state is not None:
        state.update(cache=cache, input_ids=trajectory.input_ids[:-1].tolist(),
                     reused_tokens=reused)
    return torch.cat(scores, dim=-1) if score_response else None
