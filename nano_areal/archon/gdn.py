"""
GatedDeltaNet linear-attention layer (Qwen3.5 / Qwen3-Next family).

Pure PyTorch: runs on CPU / MPS / CUDA. When the `fla` package
(flash-linear-attention) is importable and the tensors live on CUDA, the
chunked kernel is used instead of the torch reference implementation.

Layout of one layer (mirrors HF `Qwen3_5GatedDeltaNet` parameter names so
weights map 1:1):

  in_proj_qkv : D -> 2*key_dim + value_dim
  in_proj_z   : D -> value_dim          (output gate, SiLU-gated RMSNorm)
  in_proj_b   : D -> num_v_heads        (beta, sigmoid -> write strength)
  in_proj_a   : D -> num_v_heads        (decay input, softplus(a + dt_bias))
  conv1d      : depthwise causal conv over q/k/v, kernel 4, SiLU
  A_log       : per-head log decay scale
  dt_bias     : per-head decay bias
  norm        : RMSNormGated(head_v_dim)
  out_proj    : value_dim -> D
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # optional CUDA fast path
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
except Exception:  # pragma: no cover - fla is optional
    _fla_chunk = None


class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating: norm(x) * w * silu(gate). Matches FLA."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dtype)
        h = h * F.silu(gate.float())
        return h.to(dtype)


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Depthwise causal conv. x: [B, C, T]; weight: [C, K]. Returns SiLU(conv)."""
    _, C, T = x.shape
    K = weight.shape[-1]
    out = F.conv1d(x.to(weight.dtype), weight.unsqueeze(1), bias, padding=K - 1, groups=C)
    return F.silu(out[:, :, :T]).to(x.dtype)


def chunk_gated_delta_rule(
    q: Tensor,          # [B, T, H, Dk]
    k: Tensor,          # [B, T, H, Dk]
    v: Tensor,          # [B, T, H, Dv]
    g: Tensor,          # [B, T, H]   log decay (<= 0)
    beta: Tensor,       # [B, T, H]   in (0, 1)
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """
    Torch reference of the chunked gated delta rule (Yang et al., 2024).
    q/k are L2-normalised inside (matches `use_qk_l2norm_in_kernel=True`).
    Returns output [B, T, H, Dv] and the final recurrent state [B, H, Dk, Dv].
    """
    if _fla_chunk is not None and q.is_cuda:
        out, state = _fla_chunk(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
        )
        return out, state

    dtype_in = q.dtype
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v, beta, g = [
        x.transpose(1, 2).to(torch.float32).contiguous() for x in (q, k, v, beta, g)
    ]  # [B, H, T, ...]
    q = l2norm(q)
    k = l2norm(k)
    q = q * Dk ** -0.5

    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    Tp = T + pad
    n_chunks = Tp // chunk_size

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [
        x.reshape(B, H, n_chunks, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)
    ]
    g = g.reshape(B, H, n_chunks, chunk_size)

    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum_g = g.cumsum(-1)
    pair = (cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()

    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    k_beta_decayed = k_beta * cum_g.exp().unsqueeze(-1)

    u = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(ut, k_beta_decayed, upper=False, unitriangular=True)

    state = (
        torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=q.device)
        if initial_state is None else initial_state.to(torch.float32)
    )
    out = torch.zeros_like(u)
    q = q * cum_g.exp().unsqueeze(-1)
    k = k * (cum_g[..., -1:] - cum_g).exp().unsqueeze(-1)
    chunk_decay = cum_g[..., -1].exp()[..., None, None]

    for i in range(n_chunks):
        v_new = u[:, :, i] - w[:, :, i] @ state
        out[:, :, i] = q[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(B, H, Tp, Dv)[:, :, :T].transpose(1, 2).to(dtype_in).contiguous()
    return out, (state if output_final_state else None)


def recurrent_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    state: Tensor,  # [B, H, Dk, Dv]
) -> tuple[Tensor, Tensor]:
    """Single-step recurrence used during decoding. q/k/v: [B, 1, H, D]."""
    dtype_in = q.dtype
    q = l2norm(q[:, 0].float()) * q.shape[-1] ** -0.5   # [B, H, Dk]
    k = l2norm(k[:, 0].float())
    v = v[:, 0].float()
    g = g[:, 0].float().exp()[..., None, None]           # [B, H, 1, 1]
    beta = beta[:, 0].float()[..., None]                 # [B, H, 1]
    state = state.float() * g
    kv_mem = (state * k.unsqueeze(-1)).sum(-2)           # [B, H, Dv]
    delta = (v - kv_mem) * beta
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (state * q.unsqueeze(-1)).sum(-2)              # [B, H, Dv]
    return out.unsqueeze(1).to(dtype_in), state


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = nn.Linear(dim, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(dim, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(dim, num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, kernel_size=conv_kernel, groups=conv_dim,
            bias=False, padding=conv_kernel - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.empty(num_v_heads).uniform_(0.01, 16).log())
        self.norm = RMSNormGated(head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, dim, bias=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, T, _ = x.shape
        z = self.in_proj_z(x).view(B, T, self.num_v_heads, self.head_v_dim)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        mixed = self.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
        return mixed, z, beta, g, B, T

    def _split_heads(self, mixed: Tensor, B: int, T: int) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = torch.split(
            mixed.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        return q, k, v

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """x: [B, T, D]; padding_mask: [B, T] 1 for real tokens (zeroes pads)."""
        if padding_mask is not None:
            x = (x * padding_mask[:, :, None]).to(x.dtype)
        mixed, z, beta, g, B, T = self._project(x)
        mixed = causal_conv1d(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias)
        q, k, v = self._split_heads(mixed, B, T)
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(B, T, -1))

    @torch.no_grad()
    def forward_step(
        self, x: Tensor, conv_state: Tensor, rec_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decode one token. x: [B, 1, D]; conv_state: [B, C, K-1]; rec_state: [B, H, Dk, Dv].
        Returns (out [B, 1, D], new conv_state, new rec_state).
        """
        mixed, z, beta, g, B, T = self._project(x)
        w = self.conv1d.weight.squeeze(1)                     # [C, K]
        window = torch.cat([conv_state, mixed], dim=-1)       # [B, C, K]
        conv_state = window[:, :, 1:]
        conv_out = F.silu((window.to(w.dtype) * w).sum(-1, keepdim=True)).to(mixed.dtype)
        q, k, v = self._split_heads(conv_out, B, 1)
        out, rec_state = recurrent_gated_delta_rule(q, k, v, g, beta, rec_state)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(B, 1, -1)), conv_state, rec_state

    def init_state(self, B: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        C = self.conv1d.in_channels
        K = self.conv1d.kernel_size[0]
        conv_state = torch.zeros(B, C, K - 1, device=device, dtype=dtype)
        rec_state = torch.zeros(
            B, self.num_v_heads, self.head_k_dim, self.head_v_dim, device=device, dtype=torch.float32
        )
        return conv_state, rec_state
