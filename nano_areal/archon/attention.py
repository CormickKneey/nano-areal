"""
Full-attention layer shared by Qwen3 (dense) and Qwen3.5 (hybrid).

Qwen3.5 differences, all switchable via args:
  * `attn_output_gate`: q_proj emits 2x heads; the second half is a sigmoid
    gate applied to the attention output before o_proj.
  * partial RoPE (rotary on the first `rotary_dim` dims only) — handled by
    `apply_rope`, which reads the rotary width from cos/sin.
  * zero-centred RMSNorm (weight stored as w-1) — see `RMSNorm`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nano_areal.archon.mrope import apply_rope


class RMSNorm(nn.Module):
    """RMSNorm. `zero_centered=True` stores weight as (w - 1), like Qwen3.5."""

    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = False):
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered else torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float()
        if self.zero_centered:
            w = 1.0 + w
        return (h * w).type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        output_gate: bool = False,
        eps: float = 1e-6,
        zero_centered_norm: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.output_gate = output_gate

        q_out = n_heads * head_dim * (2 if output_gate else 1)
        self.q_proj = nn.Linear(dim, q_out, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)
        self.q_norm = RMSNorm(head_dim, eps=eps, zero_centered=zero_centered_norm)
        self.k_norm = RMSNorm(head_dim, eps=eps, zero_centered=zero_centered_norm)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        B, T, _ = x.shape
        q = self.q_proj(x)
        gate = None
        if self.output_gate:
            q, gate = q.view(B, T, self.n_heads, 2 * self.head_dim).chunk(2, dim=-1)
            gate = gate.reshape(B, T, -1)
        q = self.q_norm(q.reshape(B, T, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        return q, k, v, gate

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None, causal: bool) -> Tensor:
        # q/k/v: [B, T, H, D] -> SDPA [B, H, T, D]
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, is_causal=causal and attn_mask is None,
        )
        return out.transpose(1, 2)

    def forward(
        self,
        x: Tensor,                     # [B, T, D]
        cos: Tensor,                   # [B, T, 1, rotary_dim]
        sin: Tensor,
        attention_mask: Tensor | None = None,   # additive/bool SDPA mask or None (causal)
    ) -> Tensor:
        B, T, _ = x.shape
        q, k, v, gate = self._qkv(x)
        q, k = apply_rope(q, k, cos, sin)
        out = self._attend(q, k, v, attention_mask, causal=True).reshape(B, T, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    @torch.no_grad()
    def forward_step(
        self, x: Tensor, cos: Tensor, sin: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode: x [B, 1, D]; caches [B, S, Hkv, D]. Returns out and grown caches."""
        B = x.shape[0]
        q, k, v, gate = self._qkv(x)
        q, k = apply_rope(q, k, cos, sin)
        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)
        out = self._attend(q, k_cache, v_cache, None, causal=False).reshape(B, 1, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out), k_cache, v_cache
