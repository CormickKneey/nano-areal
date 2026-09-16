"""
Minimal LoRA for nano-archon.

`apply_lora(model, cfg)` swaps selected `nn.Linear` modules for `LoRALinear`
(frozen base + trainable A/B), freezes everything else, and optionally keeps
the vision tower frozen. `merge_lora_state_dict` produces a plain state dict
(W + BA*scale) that the inference engine can load without LoRA support —
this is what weight sync ships to vLLM.

Default targets cover both attention flavours and the MLP:
  self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_qkv,in_proj_z,out_proj},
  mlp.{gate,up,down}_proj
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGETS)
    freeze_vision: bool = True
    train_embeddings: bool = False


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)

    def merged_weight(self) -> Tensor:
        return self.base.weight + (self.lora_B.float() @ self.lora_A.float() * self.scale).to(self.base.weight.dtype)


def _is_target(name: str, targets: tuple[str, ...]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in targets and not name.startswith("visual.")


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """In-place: wrap target Linear layers, freeze the rest. Returns the model."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _is_target(full, cfg.targets):
                setattr_module(module, child_name, LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout))
    if cfg.train_embeddings:
        for name, p in model.named_parameters():
            if "embed_tokens" in name:
                p.requires_grad_(True)
    if not cfg.freeze_vision:
        for name, p in model.named_parameters():
            if name.startswith("visual."):
                p.requires_grad_(True)
    return model


def setattr_module(parent: nn.Module, name: str, new: nn.Module) -> None:
    parent._modules[name] = new


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the adapter weights (for lightweight checkpoints)."""
    return {k: v for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def merge_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """
    Full state dict with LoRA folded into the base weights and `.base.` removed
    from names, so it matches a LoRA-free `Qwen35Model` exactly.
    """
    out: dict[str, Tensor] = {}
    lora_modules = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for k, v in model.state_dict().items():
        if ".lora_A" in k or ".lora_B" in k:
            continue
        owner = k.rsplit(".", 2)[0] if k.endswith(".base.weight") or k.endswith(".base.bias") else None
        if owner is not None and owner in lora_modules:
            if k.endswith(".base.weight"):
                out[owner + ".weight"] = lora_modules[owner].merged_weight().detach()
            else:
                out[owner + ".bias"] = v
        else:
            out[k] = v
    return out


def count_trainable(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
