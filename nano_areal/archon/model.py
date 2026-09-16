"""
nano-archon: PyTorch-native Qwen3 / Qwen3.5 implementation.

* Qwen3 dense (Qwen3-0.6B ...): every layer is full attention, 1-D RoPE.
* Qwen3.5 (Qwen3.5-2B, Qwen3.8-27B dense variants): hybrid decoder that
  alternates GatedDeltaNet and gated full attention (`layer_types`), partial
  interleaved M-RoPE, zero-centred RMSNorm, optional ViT for images.

Pure torch — runs on CPU/MPS/CUDA; `fla` kernels are used opportunistically on
CUDA for the GDN layers. Weight names mirror HF (minus the `model.` prefix) so
`state_dict.py` is a thin key translator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from nano_areal.archon.attention import Attention, RMSNorm
from nano_areal.archon.gdn import GatedDeltaNet
from nano_areal.archon.mrope import RotaryEmbedding, get_rope_index
from nano_areal.archon.vision import VisionArgs, VisionModel

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
    head_dim: int = 128
    attention_bias: bool = False

    # Qwen3.5 hybrid extras. Defaults reproduce Qwen3 dense.
    layer_types: list[str] = field(default_factory=list)   # [] → all "full_attention"
    attn_output_gate: bool = False
    partial_rotary_factor: float = 1.0
    mrope_section: list[int] | None = None
    zero_centered_norm: bool = False
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Vision (None → text-only model)
    vision: VisionArgs | None = None
    image_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1

    @property
    def is_hybrid(self) -> bool:
        return any(t == "linear_attention" for t in self.layer_types)

    def layer_type(self, i: int) -> str:
        return self.layer_types[i] if self.layer_types else "full_attention"

    @classmethod
    def from_hf_config(cls, config_path: str | Path, text_only: bool = False) -> "Qwen35Args":
        p = Path(config_path)
        with open(p / "config.json" if p.is_dir() else p) as f:
            cfg = json.load(f)
        return cls.from_hf_dict(cfg, text_only=text_only)

    @classmethod
    def from_hf_dict(cls, cfg: dict, text_only: bool = False) -> "Qwen35Args":
        tc = cfg.get("text_config", cfg)
        hidden = tc["hidden_size"]
        n_heads = tc["num_attention_heads"]
        rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        is_q35 = tc.get("model_type", cfg.get("model_type", "")) in ("qwen3_5_text", "qwen3_5", "qwen3_next")
        layer_types = list(tc.get("layer_types") or [])
        if not layer_types and is_q35:
            interval = tc.get("full_attention_interval", 4)
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(tc["num_hidden_layers"])
            ]
        vision = None
        vc = cfg.get("vision_config")
        if vc and not text_only:
            vision = VisionArgs.from_hf(vc)
        return cls(
            dim=hidden,
            n_layers=tc["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=tc.get("num_key_value_heads", n_heads),
            vocab_size=tc["vocab_size"],
            intermediate_size=tc["intermediate_size"],
            max_position_embeddings=tc.get("max_position_embeddings", 40960),
            rope_theta=float(rp.get("rope_theta", tc.get("rope_theta", 1000000.0))),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", True)),
            head_dim=tc.get("head_dim", hidden // n_heads),
            attention_bias=tc.get("attention_bias", False),
            layer_types=layer_types,
            attn_output_gate=tc.get("attn_output_gate", is_q35),
            partial_rotary_factor=float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))),
            mrope_section=list(rp["mrope_section"]) if rp.get("mrope_section") else ([11, 11, 10] if is_q35 else None),
            zero_centered_norm=is_q35,
            linear_num_key_heads=tc.get("linear_num_key_heads", 16),
            linear_num_value_heads=tc.get("linear_num_value_heads", 16),
            linear_key_head_dim=tc.get("linear_key_head_dim", 128),
            linear_value_head_dim=tc.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=tc.get("linear_conv_kernel_dim", 4),
            vision=vision,
            image_token_id=cfg.get("image_token_id", -1),
            vision_start_token_id=cfg.get("vision_start_token_id", -1),
            vision_end_token_id=cfg.get("vision_end_token_id", -1),
        )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.gate_proj = nn.Linear(args.dim, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.dim, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """One decoder block: (GDN | gated attention) + SwiGLU MLP, pre-norm."""

    def __init__(self, args: Qwen35Args, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_type(layer_idx)
        zc = args.zero_centered_norm
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                dim=args.dim,
                num_k_heads=args.linear_num_key_heads,
                num_v_heads=args.linear_num_value_heads,
                head_k_dim=args.linear_key_head_dim,
                head_v_dim=args.linear_value_head_dim,
                conv_kernel=args.linear_conv_kernel_dim,
                eps=args.rms_norm_eps,
            )
        else:
            self.self_attn = Attention(
                dim=args.dim,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                bias=args.attention_bias,
                output_gate=args.attn_output_gate,
                eps=args.rms_norm_eps,
                zero_centered_norm=zc,
            )
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.dim, eps=args.rms_norm_eps, zero_centered=zc)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.rms_norm_eps, zero_centered=zc)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(h)
        else:
            h = self.self_attn(h, cos, sin, attention_mask)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([DecoderLayer(args, i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.rms_norm_eps, zero_centered=args.zero_centered_norm)
        self.rotary_emb = RotaryEmbedding(
            args.head_dim, args.rope_theta, args.partial_rotary_factor, args.mrope_section
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: Tensor,          # [B, T, D]
        positions: Tensor,              # [B, T] or [3, B, T]
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = inputs_embeds
        cos, sin = self.rotary_emb(positions, dtype=x.dtype)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, attention_mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, attention_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Qwen35Model(nn.Module):
    """
    Qwen3 / Qwen3.5 causal LM with optional vision tower.
    Loadable from HuggingFace checkpoints via `state_dict.load_hf_weights`.
    """

    def __init__(self, args: Qwen35Args):
        super().__init__()
        self.args = args
        self.language_model = TextModel(args)
        self.visual = VisionModel(args.vision) if args.vision is not None else None
        self.lm_head = None if args.tie_word_embeddings else nn.Linear(args.dim, args.vocab_size, bias=False)
        self._init_weights()

    # -- init ---------------------------------------------------------------

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    @property
    def layers(self) -> nn.ModuleList:
        return self.language_model.layers

    def set_gradient_checkpointing(self, enabled: bool):
        self.language_model.gradient_checkpointing = enabled

    # -- forward ------------------------------------------------------------

    def embed(
        self,
        input_ids: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None and pixel_values.numel() > 0:
            if self.visual is None:
                raise ValueError("pixel_values given but model has no vision tower")
            img = self.visual(pixel_values.to(x.device), image_grid_thw.to(x.device)).to(x.dtype)
            mask = (input_ids == self.args.image_token_id).unsqueeze(-1).expand_as(x)
            n_tok = int(mask[..., 0].sum().item())
            if n_tok != img.shape[0]:
                raise ValueError(f"{n_tok} image tokens but {img.shape[0]} image embeddings")
            x = x.masked_scatter(mask, img)
        return x

    def positions_for(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T = input_ids.shape
        if image_grid_thw is None or image_grid_thw.numel() == 0 or self.args.image_token_id < 0:
            return torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        return get_rope_index(
            input_ids, self.args.image_token_id, image_grid_thw,
            self.args.vision.spatial_merge_size if self.args.vision else 2,
            attention_mask,
        )

    def forward(
        self,
        input_ids: Tensor,                        # [B, T]
        positions: Optional[Tensor] = None,       # [B, T] | [3, B, T]
        attention_mask: Optional[Tensor] = None,  # SDPA mask (None → causal)
        pixel_values: Optional[Tensor] = None,    # [N_patches, C*tp*ps*ps]
        image_grid_thw: Optional[Tensor] = None,  # [N_img, 3]
    ) -> Tensor:                                  # [B, T, vocab]
        if positions is None:
            positions = self.positions_for(input_ids, image_grid_thw)
        x = self.embed(input_ids, pixel_values, image_grid_thw)
        h = self.language_model(x, positions, attention_mask)
        return self.logits(h)

    def logits(self, h: Tensor) -> Tensor:
        if self.lm_head is not None:
            return self.lm_head(h)
        return F.linear(h, self.language_model.embed_tokens.weight)

    def compute_log_probs(
        self,
        input_ids: Tensor,         # [B, T]
        response_mask: Tensor,     # [B, T]
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:                   # [B, T-1] log P(token[t+1] | ..), 0 outside mask
        logits = self.forward(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tok = logp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return tok * response_mask[:, 1:]


def build_model(args: Qwen35Args, device: torch.device, dtype: torch.dtype) -> Qwen35Model:
    return Qwen35Model(args).to(dtype=dtype, device=device)
