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
