"""Opt-in native Metal RMS arithmetic and its first-order autograd contract."""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

RMS_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void rms_reference(
    device const bfloat* hidden,
    device const bfloat* weight,
    device bfloat* output,
    device float* inverses,
    constant uint& width,
    constant float& epsilon,
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float totals[32];
    threadgroup float inverse;
    float values[4];
    float total = 0;
    for (uint index = 0; index < 4; ++index) {
        uint column = thread_index * 4 + index;
        values[index] = column < width ? float(hidden[row * width + column]) : 0;
        total = fma(values[index], values[index], total);
    }
    total = simd_sum(total);
    if (group == 0) totals[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) totals[group] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
        float combined = simd_sum(totals[lane]);
        if (lane == 0) {
            inverse = precise::rsqrt(combined / width + epsilon);
            inverses[row] = inverse;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = 0; index < 4; ++index) {
        uint column = thread_index * 4 + index;
        if (column < width) output[row * width + column] =
            weight[column] * bfloat(values[index] * inverse);
    }
}
"""


@lru_cache(maxsize=1)
def rms_library():
    return torch.mps.compile_shader(RMS_SOURCE)


def _metal_rms_forward(hidden, weight, epsilon):
    width = hidden.shape[-1]
    if (hidden.device.type != "mps" or weight.device != hidden.device
            or hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype
            or weight.shape != (width,) or not 128 <= width <= 4096 or width % 128
            or hidden.numel() == 0 or epsilon <= 0):
        raise ValueError("Metal RMS requires measured bf16 MPS shapes and epsilon > 0")
    threads = (width + 127) // 128 * 32
    output = torch.empty_like(hidden.contiguous())
    inverses = torch.empty((*hidden.shape[:-1], 1), device=hidden.device)
    rms_library().rms_reference(
        hidden.contiguous(), weight.contiguous(), output, inverses, width, epsilon,
        threads=hidden.numel() // width * threads, group_size=threads,
    )
    return output, inverses


def metal_rms(hidden, weight, epsilon):
    if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
        raise ValueError("Metal RMS is a forward-only diagnostic, not a training op")
    output, _ = _metal_rms_forward(hidden, weight, epsilon)
    return output


class _DifferentiableMetalRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, epsilon):
        output, inverses = _metal_rms_forward(hidden, weight, epsilon)
        ctx.save_for_backward(hidden, weight, inverses)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        hidden, weight, inverse = ctx.saved_tensors
        values = hidden.float()
        normalized_gradient = (gradient * weight).float()
        projection = (normalized_gradient * values).mean(-1, keepdim=True)
        hidden_gradient = (normalized_gradient * inverse
                           - values * inverse.pow(3) * projection).to(hidden.dtype)
        weight_gradient = None
        if ctx.needs_input_grad[1]:
            normalized = (values * inverse).to(hidden.dtype)
            weight_gradient = (gradient * normalized).sum(tuple(range(hidden.ndim - 1)))
        return hidden_gradient, weight_gradient, None


def differentiable_metal_rms(hidden, weight, epsilon):
    return _DifferentiableMetalRMS.apply(hidden, weight, epsilon)
