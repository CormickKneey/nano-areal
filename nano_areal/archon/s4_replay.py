"""Opt-in text-only cached forward reference, never installed by the trainer."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nano_areal.archon import model as native_model


def cached_text_logprobs(
    model, trajectory, before_segment=None, project_head=None, state=None,
    observe=None,
    attention_kernel=None, activate_segment=None, checkpoint_layers=False,
    score_response=True,
):
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
    cache = ([tuple(value[..., :reused, :] for value in pair)
              for pair in state["cache"]] if reused else
             [None] * len(model.language_model.layers))
    ids = ids.to(device).unsqueeze(0)
    scores = []
    segments = [(reused, prompt), *[(index, index + 1)
                               for index in range(prompt, ids.shape[1] - 1)]]
    for start, end in segments:
        if before_segment is not None:
            before_segment(start, end)
        hidden = model.language_model.embed_tokens(ids[:, start:end])
        position_ids = torch.arange(start, end, device=device)[None, None].expand(
            3, 1, -1)
        cos, sin = model.language_model.rope(position_ids, hidden.dtype)
        length = end - start
        attention_mask = None
        if start and length > 1:
            attention_mask = (torch.arange(end, device=device)[None, :]
                              <= torch.arange(start, end, device=device)[:, None])
        for index, layer in enumerate(model.language_model.layers):
            def forward_layer(hidden, old_key, old_value, layer=layer, index=index,
                              start=start, end=end, length=length, cos=cos, sin=sin,
                              attention_mask=attention_mask):
                if activate_segment is not None:
                    activate_segment(start, end, prompt)
                attention = layer.attn
                normalized = layer.input_layernorm(hidden)
                query = attention.q_norm(attention.q_proj(normalized).view(
                    1, length, attention.n_heads, attention.head_dim))
                key = attention.k_norm(attention.k_proj(normalized).view(
                    1, length, attention.n_kv_heads, attention.head_dim))
                value = attention.v_proj(normalized).view(
                    1, length, attention.n_kv_heads, attention.head_dim)
                query, key = native_model.apply_rope(query, key, cos, sin)
                if observe is not None:
                    observe(f"layers.{index}.attn.rope_query", query)
                    observe(f"layers.{index}.attn.rope_key", key)
                key, value = key.transpose(1, 2), value.transpose(1, 2)
                if old_key is not None:
                    key = torch.cat((old_key, key), dim=-2)
                    value = torch.cat((old_value, value), dim=-2)
                kernel = attention_kernel or F.scaled_dot_product_attention
                attended = kernel(
                    query.transpose(1, 2),
                    key.repeat_interleave(attention.n_rep, dim=1),
                    value.repeat_interleave(attention.n_rep, dim=1),
                    attn_mask=attention_mask, is_causal=start == 0,
                ).transpose(1, 2).contiguous().view(1, length, -1)
                hidden = hidden + attention.o_proj(attended)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                return hidden, key, value

            old_key, old_value = cache[index] if cache[index] else (None, None)
            if checkpoint_layers and model.training and torch.is_grad_enabled():
                hidden, key, value = checkpoint(
                    forward_layer, hidden, old_key, old_value, use_reentrant=False)
            else:
                hidden, key, value = forward_layer(hidden, old_key, old_value)
            cache[index] = key, value
        if not score_response:
            continue
        hidden = model.language_model.norm(hidden[:, -1:])
        if project_head is None:
            weight = (model.language_model.embed_tokens.weight if model.tied
                      else model.lm_head.weight)
            logits = F.linear(hidden, weight)
        else:
            logits = project_head(hidden, start, end)
        scores.append(logits.float().log_softmax(-1).gather(
            -1, ids[:, end:end + 1, None]).squeeze(-1))
    if state is not None:
        state.update(cache=cache, input_ids=trajectory.input_ids[:-1].tolist(),
                     reused_tokens=reused)
    return torch.cat(scores, dim=-1) if score_response else None
