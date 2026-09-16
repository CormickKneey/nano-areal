"""
Qwen3.5 vision encoder (SigLIP-style ViT + 2x2 patch merger).

Input follows the HF image processor: `pixel_values` [N_patches, C*tp*ps*ps]
with patches already in spatial-merge-block order, plus `grid_thw` [N_img, 3].
Output: [N_patches / merge^2, out_hidden_size] image embeddings that replace
`<|image_pad|>` token embeddings in the language model.

Qwen3.5 checkpoints ship `deepstack_visual_indexes: []`, so there is no
DeepStack injection here; the merged tokens enter the LLM only at the embedding.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nano_areal.archon.mrope import rotate_half


@dataclass
class VisionArgs:
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    rope_theta: float = 10000.0
    hidden_act: str = "gelu_pytorch_tanh"

    @classmethod
    def from_hf(cls, vc: dict) -> "VisionArgs":
        rp = vc.get("rope_parameters") or {}
        return cls(
            depth=vc["depth"],
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_heads=vc["num_heads"],
            in_channels=vc.get("in_channels", 3),
            patch_size=vc["patch_size"],
            spatial_merge_size=vc.get("spatial_merge_size", 2),
            temporal_patch_size=vc.get("temporal_patch_size", 2),
            out_hidden_size=vc["out_hidden_size"],
            num_position_embeddings=vc.get("num_position_embeddings", 2304),
            rope_theta=rp.get("rope_theta", 10000.0),
            hidden_act=vc.get("hidden_act", "gelu_pytorch_tanh"),
        )


def _act(name: str):
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"unsupported vision activation {name}")


class PatchEmbed(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.a = a
        ks = (a.temporal_patch_size, a.patch_size, a.patch_size)
        self.proj = nn.Conv3d(a.in_channels, a.hidden_size, kernel_size=ks, stride=ks, bias=True)

    def forward(self, pixel_values: Tensor) -> Tensor:
        a = self.a
        x = pixel_values.view(-1, a.in_channels, a.temporal_patch_size, a.patch_size, a.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, a.hidden_size)


class VisionAttention(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.num_heads = a.num_heads
        self.head_dim = a.hidden_size // a.num_heads
        self.qkv = nn.Linear(a.hidden_size, 3 * a.hidden_size, bias=True)
        self.proj = nn.Linear(a.hidden_size, a.hidden_size, bias=True)

    def forward(self, x: Tensor, cu_seqlens: list[int], cos: Tensor, sin: Tensor) -> Tensor:
        N = x.shape[0]
        q, k, v = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        # rotary in fp32, full head dim
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        qf, kf = q.float(), k.float()
        q = (qf * c + rotate_half(qf) * s).to(q.dtype)
        k = (kf * c + rotate_half(kf) * s).to(k.dtype)
        outs = []
        for i in range(len(cu_seqlens) - 1):
            lo, hi = cu_seqlens[i], cu_seqlens[i + 1]
            o = F.scaled_dot_product_attention(
                q[lo:hi].transpose(0, 1).unsqueeze(0),
                k[lo:hi].transpose(0, 1).unsqueeze(0),
                v[lo:hi].transpose(0, 1).unsqueeze(0),
            )  # [1, H, L, D]
            outs.append(o[0].transpose(0, 1))
        out = torch.cat(outs, dim=0).reshape(N, -1)
        return self.proj(out)


class VisionBlock(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.norm1 = nn.LayerNorm(a.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(a.hidden_size, eps=1e-6)
        self.attn = VisionAttention(a)
        self.mlp = VisionMLP(a)

    def forward(self, x: Tensor, cu_seqlens: list[int], cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionMLP(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.linear_fc1 = nn.Linear(a.hidden_size, a.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(a.intermediate_size, a.hidden_size, bias=True)
        self.act = _act(a.hidden_act)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class PatchMerger(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.in_dim = a.hidden_size
        self.merged = a.hidden_size * a.spatial_merge_size ** 2
        self.norm = nn.LayerNorm(a.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged, self.merged)
        self.linear_fc2 = nn.Linear(self.merged, a.out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x).view(-1, self.merged)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.a = a
        self.merge = a.spatial_merge_size
        self.patch_embed = PatchEmbed(a)
        self.pos_embed = nn.Embedding(a.num_position_embeddings, a.hidden_size)
        self.side = int(a.num_position_embeddings ** 0.5)
        head_dim = a.hidden_size // a.num_heads
        spatial = head_dim // 2
        inv_freq = 1.0 / (a.rope_theta ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.blocks = nn.ModuleList([VisionBlock(a) for _ in range(a.depth)])
        self.merger = PatchMerger(a)

    # -- position helpers -------------------------------------------------

    def _block_order(self, h: int, w: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """(row, col) indices of patches in spatial-merge-block order."""
        m = self.merge
        hp, wp = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        shape = (h // m, m, w // m, m)
        return (
            hp.reshape(shape).transpose(1, 2).flatten(),
            wp.reshape(shape).transpose(1, 2).flatten(),
        )

    def _pos_embed(self, grid_thw: list[list[int]], device: torch.device) -> Tensor:
        table = self.pos_embed.weight.view(1, self.side, self.side, -1).permute(0, 3, 1, 2)
        outs = []
        for t, h, w in grid_thw:
            grid = F.interpolate(
                table.float(), size=(h, w), mode="bilinear", align_corners=True
            )[0].permute(1, 2, 0)  # [h, w, D]
            r, c = self._block_order(h, w, device)
            pe = grid[r, c]  # [h*w, D]
            outs.append(pe.repeat(t, 1))
        return torch.cat(outs, 0)

    def _rotary(self, grid_thw: list[list[int]], device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        pos = []
        for t, h, w in grid_thw:
            r, c = self._block_order(h, w, device)
            pos.append(torch.stack([r, c], dim=-1).repeat(t, 1))
        pos = torch.cat(pos, 0).float()                         # [N, 2]
        freqs = pos[..., None] * self.inv_freq                  # [N, 2, S/2]
        hw = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)      # [N, S]
        emb = torch.cat([hw, hw], dim=-1)                       # [N, head_dim]
        return emb.cos(), emb.sin()

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        """pixel_values [N, C*tp*ps*ps]; grid_thw [N_img, 3] -> [N / merge^2, out_hidden]."""
        thw = [[int(v) for v in row] for row in grid_thw.tolist()]
        dev = pixel_values.device
        x = self.patch_embed(pixel_values)
        x = x + self._pos_embed(thw, dev).to(x.dtype)
        cos, sin = self._rotary(thw, dev, x.dtype)
        cu = [0]
        for t, h, w in thw:
            for _ in range(t):
                cu.append(cu[-1] + h * w)
        for blk in self.blocks:
            x = blk(x, cu, cos, sin)
        return self.merger(x)
