"""
Rotary embeddings for nano-archon.

* Plain 1-D RoPE (Qwen3 dense): positions [B, T].
* Interleaved M-RoPE (Qwen3.5 VLM): positions [3, B, T] = (t, h, w). The
  rotary dim is split by `mrope_section`; frequency index i uses the temporal
  position for i % 3 == 0, height for i % 3 == 1, width for i % 3 == 2 (only up
  to 3 * section[axis]). Text tokens have t == h == w so the result equals 1-D
  RoPE — text-only inputs never need to build 3-D positions.

Also hosts `get_rope_index`, which converts (input_ids, image_grid_thw) into
3-D positions the same way the HF processor/model do.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """
    Partial RoPE: rotate the first `cos.shape[-1]` dims of q/k, pass the rest through.
    q/k: [B, T, H, D]; cos/sin: [B, T, 1, R] with R <= D.
    """
    r = cos.shape[-1]
    q_rot, q_pass = q[..., :r], q[..., r:]
    k_rot, k_pass = k[..., :r], k[..., r:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], -1), torch.cat([k_rot, k_pass], -1)


class RotaryEmbedding(nn.Module):
    """1-D RoPE or interleaved M-RoPE depending on `mrope_section`."""

    def __init__(
        self,
        head_dim: int,
        theta: float,
        partial_rotary_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = mrope_section

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """
        positions: [B, T] (1-D) or [3, B, T] (M-RoPE).
        Returns cos, sin: [B, T, 1, rotary_dim].
        """
        if positions.dim() == 2:
            freqs = positions[..., None].float() * self.inv_freq          # [B, T, R/2]
        else:
            freqs3 = positions[..., None].float() * self.inv_freq         # [3, B, T, R/2]
            freqs = freqs3[0].clone()
            if self.mrope_section is not None:
                for axis, offset in ((1, 1), (2, 2)):
                    length = self.mrope_section[axis] * 3
                    idx = slice(offset, length, 3)
                    freqs[..., idx] = freqs3[axis, ..., idx]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype).unsqueeze(2), emb.sin().to(dtype).unsqueeze(2)


def get_rope_index(
    input_ids: Tensor,                 # [B, T]
    image_token_id: int,
    image_grid_thw: Tensor | None,     # [N_img, 3] (t, h, w) in patches
    spatial_merge_size: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """
    3-D (t, h, w) positions for text + image tokens, [3, B, T].
    Text runs advance all three axes together; an image block occupies a
    (t, h/m, w/m) grid anchored at the current position, after which the text
    position jumps by max(h, w)/m. Images are consumed in order across the batch.
    """
    B, T = input_ids.shape
    dev = input_ids.device
    pos = torch.zeros(3, B, T, dtype=torch.long, device=dev)
    img_iter = iter(image_grid_thw.tolist()) if image_grid_thw is not None else iter(())
    for b in range(B):
        ids = input_ids[b]
        keep = attention_mask[b].bool() if attention_mask is not None else torch.ones(T, dtype=torch.bool, device=dev)
        ids_kept = ids[keep]
        is_img = (ids_kept == image_token_id).tolist()
        chunks: list[Tensor] = []
        cur = 0
        i = 0
        n = len(is_img)
        while i < n:
            j = i
            while j < n and is_img[j] == is_img[i]:
                j += 1
            if not is_img[i]:
                span = torch.arange(j - i, device=dev).view(1, -1).expand(3, -1) + cur
                chunks.append(span)
                cur += j - i
            else:
                t, h, w = next(img_iter)
                gt, gh, gw = t, h // spatial_merge_size, w // spatial_merge_size
                tt, hh, ww = torch.meshgrid(
                    torch.arange(gt, device=dev), torch.arange(gh, device=dev),
                    torch.arange(gw, device=dev), indexing="ij",
                )
                grid = torch.stack([tt, hh, ww]).reshape(3, -1) + cur
                if grid.shape[1] != j - i:
                    raise ValueError(
                        f"image token count {j - i} != grid size {grid.shape[1]} for thw={(t, h, w)}"
                    )
                chunks.append(grid)
                cur += max(gh, gw)
            i = j
        if chunks:
            pos[:, b, keep] = torch.cat(chunks, dim=1)
    return pos
