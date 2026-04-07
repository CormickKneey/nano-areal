"""
nano-archon: PyTorch-native Qwen3 implementation.

Supports Qwen3 dense models (Qwen3-0.6B, 1.7B, 4B, 8B, ...).
Inspired by torchtitan (https://github.com/pytorch/torchtitan).
No dependency on HuggingFace forward pass — pure torch, torch.compile-friendly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Model arguments
# ---------------------------------------------------------------------------

@dataclass
class Qwen35Args:
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    head_dim: int = 128            # explicit per-head dim (Qwen3: 128)
    attention_bias: bool = False   # Qwen3 has no Q/K/V bias

    @classmethod
    def from_hf_config(cls, config_path: str | Path) -> "Qwen35Args":
        with open(Path(config_path) / "config.json") as f:
            cfg = json.load(f)
        # Qwen3.5 VLM wraps text params under "text_config"; Qwen3 dense is flat
        tc = cfg.get("text_config", cfg)
        hidden = tc["hidden_size"]
        return cls(
            dim=hidden,
            n_layers=tc["num_hidden_layers"],
            n_heads=tc["num_attention_heads"],
            n_kv_heads=tc.get("num_key_value_heads", tc["num_attention_heads"]),
            vocab_size=tc["vocab_size"],
            intermediate_size=tc["intermediate_size"],
            max_position_embeddings=tc.get("max_position_embeddings", 40960),
            rope_theta=tc.get("rope_theta", 1000000.0),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            head_dim=tc.get("head_dim", hidden // tc["num_attention_heads"]),
            attention_bias=tc.get("attention_bias", False),
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    def __init__(self, args: Qwen35Args):
        super().__init__()
        head_dim = args.head_dim
        inv_freq = 1.0 / (
            args.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._seq_len_cached:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(self, positions: Tensor, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        self._update_cache(seq_len, positions.device, dtype)
        cos = self._cos_cached[positions]   # [B, T, head_dim]
        sin = self._sin_cached[positions]
        return cos.unsqueeze(2), sin.unsqueeze(2)  # [B, T, 1, head_dim]


class Attention(nn.Module):
    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = args.n_heads // args.n_kv_heads  # GQA repeat factor

        bias = args.attention_bias
        self.q_proj = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=bias)
        self.k_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=bias)
        self.v_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=bias)
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        # Qwen3: per-head QK norm (stabilises training with large head_dim)
        self.q_norm = RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(args.head_dim, eps=args.rms_norm_eps)

    def forward(
        self,
        x: Tensor,           # [B, T, D]
        cos: Tensor,         # [B, T, 1, head_dim]
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Per-head QK norm (Qwen3 specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        # Repeat KV heads for GQA
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # [B, heads, T, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )  # [B, heads, T, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward."""
    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.gate_proj = nn.Linear(args.dim, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.dim, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.input_layernorm(x), cos, sin, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Qwen35Model(nn.Module):
    """
    Qwen3.5 language model (decoder-only transformer).
    Loadable from HuggingFace checkpoints via Qwen35StateDictAdapter.
    """

    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.rms_norm_eps)
        self.rope = RotaryEmbedding(args)

        if args.tie_word_embeddings:
            self.lm_head = None   # weight-tied; use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,           # [B, T]
        positions: Optional[Tensor] = None,  # [B, T]; defaults to 0..T-1
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:                     # [B, T, vocab_size]
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embed_tokens(input_ids)  # [B, T, D]
        cos, sin = self.rope(positions, seq_len=T, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)

        x = self.norm(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        return logits  # [B, T, vocab_size]

    def compute_log_probs(
        self,
        input_ids: Tensor,       # [B, T]
        response_mask: Tensor,   # [B, T] — 1 for response tokens, 0 for prompt
    ) -> Tensor:                 # [B, T] log probs (0 where mask=0)
        logits = self.forward(input_ids)                         # [B, T, V]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)    # [B, T-1, V]
        target_ids = input_ids[:, 1:]                            # [B, T-1]
        token_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)                                            # [B, T-1]
        mask = response_mask[:, 1:]                              # align with T-1
        return token_log_probs * mask


def build_model(args: Qwen35Args, device: torch.device, dtype: torch.dtype) -> Qwen35Model:
    model = Qwen35Model(args)
    model = model.to(dtype=dtype, device=device)
    return model
