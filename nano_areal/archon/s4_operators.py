"""Diagnostic reference for MLX affine decode arithmetic, not a policy loader."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def device_rope_frequencies(config, device):
    head_dim = config["head_dim"]
    rope = config.get("rope_parameters", config.get("rope_scaling", {}))
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Only default rotary frequencies are supported")
    theta = rope.get("rope_theta", config.get("rope_theta", 5000000))
    return 1 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)


def splitkv_attention(query, key, value, query_offset, partition_size=512):
    if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
            or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]
            or query.dtype != torch.bfloat16 or key.dtype != query.dtype
            or value.dtype != query.dtype or partition_size != 512
            or query_offset < 0 or query_offset + query.shape[-2] > key.shape[-2]):
        raise ValueError("Split-KV requires compatible bf16 causal attention tensors")
    query_positions = torch.arange(query_offset, query_offset + query.shape[-2],
                                   device=query.device)[:, None]
    partials, maxima, sums = [], [], []
    for start in range(0, key.shape[-2], partition_size):
        end = min(start + partition_size, key.shape[-2])
        scores = query.float() @ key[..., start:end, :].float().transpose(-2, -1)
        scores = (scores * query.shape[-1] ** -0.5) * 1.4426950408889634
        allowed = torch.arange(start, end, device=query.device)[None] <= query_positions
        scores = scores.masked_fill(~allowed, -torch.inf)
        maximum = scores.amax(-1, keepdim=True)
        safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0)
        weights = torch.exp2(scores - safe_maximum)
        denominator = weights.sum(-1, keepdim=True)
        partial = weights @ value[..., start:end, :].float()
        partials.append((partial / (denominator + 1e-6)).bfloat16().float())
        maxima.append(maximum)
        sums.append(denominator)
    global_maximum = torch.stack(maxima).amax(0)
    rescaled = [total * torch.exp2(maximum - global_maximum)
                for total, maximum in zip(sums, maxima)]
    inverse = 1 / (sum(rescaled) + 1e-6)
    output = partials[0] * rescaled[0] * inverse
    for partial, total in zip(partials[1:], rescaled[1:]):
        output = output + partial * total * inverse
    return output.to(query.dtype)


def affine_weights(packed, scales, biases):
    if packed.ndim != 2 or scales.ndim != 2 or scales.shape != biases.shape:
        raise ValueError("Expected matrix affine quantization tensors")
    if (packed.shape[0] != scales.shape[0]
            or packed.shape[1] * 8 != scales.shape[1] * 64):
        raise ValueError("Only 4-bit affine groups of 64 are supported")
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    values = ((packed.to(torch.int32)[..., None] >> shifts) & 15).float()
    values = values.reshape(scales.shape[0], scales.shape[1], 64)
    return (values * scales.float()[..., None] + biases.float()[..., None]).flatten(1)


def affine_decode(hidden, packed, scales, biases):
    if hidden.dtype != torch.bfloat16 or hidden.shape[-1] != packed.shape[-1] * 8:
        raise ValueError("Expected bf16 inputs matching the packed affine weight")
    weights = affine_weights(packed, scales, biases)
    return affine_decode_weights(hidden, weights, biases)


class _TokenwiseFrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(context, hidden, weight):
        if hidden.ndim != 3 or hidden.shape[0] != 1 or weight.requires_grad:
            raise ValueError("Tokenwise linear requires batch-one inputs "
                             "and frozen weights")
        context.save_for_backward(weight)
        return torch.cat([F.linear(hidden[:, index:index + 1], weight)
                          for index in range(hidden.shape[1])], dim=1)

    @staticmethod
    @once_differentiable
    def backward(context, gradient):
        weight, = context.saved_tensors
        return F.linear(gradient, weight.T), None


def tokenwise_frozen_linear(hidden, weight):
    return _TokenwiseFrozenLinear.apply(hidden, weight)


def affine_decode_weights(hidden, weights, biases, *, linear=F.linear):
    if (hidden.dtype != torch.bfloat16 or weights.dtype != torch.float32
            or hidden.shape[-1] != weights.shape[-1]
            or biases.shape != (weights.shape[0], weights.shape[1] // 64)):
        raise ValueError("Expected bf16 inputs and matching fp32 affine weights")
    grouped = hidden.reshape(*hidden.shape[:-1], -1, 16, 4)
    rounded = ((grouped[..., 0] + grouped[..., 1]) + grouped[..., 2]) + grouped[..., 3]
    correction = (rounded.float() - grouped.float().sum(-1)).sum(-1)
    result = linear(hidden.float(), weights) + linear(correction, biases.float())
    return result.to(hidden.dtype)


def prefill_split_count(rows, inputs, outputs):
    if min(rows, inputs, outputs) <= 0 or inputs % 64:
        raise ValueError("Split-K requires positive shapes and 64-aligned inputs")
    groups = ((outputs + 31) // 32) * ((rows + 31) // 32)
    partitions = min(max(1, 512 // groups), inputs // 64)
    while partitions > 1 and inputs % (partitions * 64):
        partitions -= 1
    return partitions


def splitk_prefill(hidden, weight):
    if (hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or hidden.shape[-1] != weight.shape[-1] or hidden.ndim != 3
            or hidden.shape[0] != 1 or hidden.shape[1] < 16):
        raise ValueError("Split-K diagnostic requires batch-one bf16 prefill >=16 rows")
    partitions = prefill_split_count(hidden.shape[1], weight.shape[1], weight.shape[0])
    if partitions >= 32:
        raise ValueError("Only small-column split reductions below 32 are supported")
    width = weight.shape[1] // partitions
    partials = [F.linear(hidden[..., start:start + width],
                         weight[:, start:start + width])
                for start in range(0, weight.shape[1], width)]
    lanes = []
    for lane in range(min(8, partitions)):
        total = partials[lane]
        for index in range(lane + 8, partitions, 8):
            total = total + partials[index]
        lanes.append(total)
    result = lanes[0]
    for total in lanes[1:]:
        result = result + total
    return result


def rounded_swiglu(gate, up):
    if gate.dtype != torch.bfloat16 or up.dtype != gate.dtype or up.shape != gate.shape:
        raise ValueError("Rounded SwiGLU requires matching bf16 gate and up tensors")
    tail = 1 / (1 + torch.exp(gate.abs()))
    sigmoid = torch.where(gate < 0, tail, 1 - tail)
    return (gate * sigmoid) * up
