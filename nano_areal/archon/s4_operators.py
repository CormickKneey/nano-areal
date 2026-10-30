"""Diagnostic reference for MLX affine decode arithmetic, not a policy loader."""

from __future__ import annotations

import torch
from torch.nn import functional as F


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
    grouped = hidden.reshape(*hidden.shape[:-1], -1, 16, 4)
    rounded = ((grouped[..., 0] + grouped[..., 1]) + grouped[..., 2]) + grouped[..., 3]
    correction = (rounded.float() - grouped.float().sum(-1)).sum(-1)
    result = F.linear(hidden.float(), weights) + F.linear(correction, biases.float())
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
