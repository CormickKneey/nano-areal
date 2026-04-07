"""
Weight conversion between HuggingFace Qwen3 and nano-archon.

HF key layout (Qwen3 dense):
  model.embed_tokens.weight
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight  (no bias in Qwen3)
  model.layers.{i}.self_attn.{q,k}_norm.weight       (per-head QK norm)
  model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.post_attention_layernorm.weight
  model.norm.weight
  lm_head.weight

nano-archon key layout (mirrors the module tree in model.py):
  embed_tokens.weight
  layers.{i}.attn.{q,k,v,o}_proj.weight
  layers.{i}.attn.{q,k}_norm.weight
  layers.{i}.mlp.{gate,up,down}_proj.weight
  layers.{i}.input_layernorm.weight
  layers.{i}.post_attention_layernorm.weight
  norm.weight
  lm_head.weight  (absent when tie_word_embeddings=True)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import torch


# ---------------------------------------------------------------------------
# Key translation tables
# ---------------------------------------------------------------------------

_HF_TO_ARCHON: list[tuple[re.Pattern, str]] = [
    # embeddings / head
    (re.compile(r"^model\.embed_tokens\.weight$"),       "embed_tokens.weight"),
    (re.compile(r"^model\.norm\.weight$"),               "norm.weight"),
    (re.compile(r"^lm_head\.weight$"),                   "lm_head.weight"),
    # per-layer attention projections (weight; bias optional)
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"),  r"layers.\1.attn.q_proj.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$"),    r"layers.\1.attn.q_proj.bias"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"),  r"layers.\1.attn.k_proj.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$"),    r"layers.\1.attn.k_proj.bias"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"),  r"layers.\1.attn.v_proj.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$"),    r"layers.\1.attn.v_proj.bias"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),  r"layers.\1.attn.o_proj.weight"),
    # per-head QK norms (Qwen3)
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$"),  r"layers.\1.attn.q_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$"),  r"layers.\1.attn.k_norm.weight"),
    # per-layer MLP
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),     r"layers.\1.mlp.gate_proj.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"),       r"layers.\1.mlp.up_proj.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),     r"layers.\1.mlp.down_proj.weight"),
    # per-layer norms
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"),         r"layers.\1.input_layernorm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"), r"layers.\1.post_attention_layernorm.weight"),
]


def hf_key_to_archon(hf_key: str) -> str | None:
    """Translate a single HF state-dict key to nano-archon key. Returns None if unknown."""
    for pattern, replacement in _HF_TO_ARCHON:
        if pattern.search(hf_key):
            return pattern.sub(replacement, hf_key)
    return None


def archon_key_to_hf(archon_key: str) -> str | None:
    """Reverse translation: nano-archon key → HF key."""
    # Build reverse map lazily
    for pattern, replacement in _HF_TO_ARCHON:
        # Attempt reverse via regex groups
        # For simple 1-1 mappings we can just scan both directions
        pass

    # Direct reverse table (manually maintained)
    rev: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^embed_tokens\.weight$"),  "model.embed_tokens.weight"),
        (re.compile(r"^norm\.weight$"),          "model.norm.weight"),
        (re.compile(r"^lm_head\.weight$"),       "lm_head.weight"),
        (re.compile(r"^layers\.(\d+)\.attn\.q_proj\.weight$"),  r"model.layers.\1.self_attn.q_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.attn\.q_proj\.bias$"),    r"model.layers.\1.self_attn.q_proj.bias"),
        (re.compile(r"^layers\.(\d+)\.attn\.k_proj\.weight$"),  r"model.layers.\1.self_attn.k_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.attn\.k_proj\.bias$"),    r"model.layers.\1.self_attn.k_proj.bias"),
        (re.compile(r"^layers\.(\d+)\.attn\.v_proj\.weight$"),  r"model.layers.\1.self_attn.v_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.attn\.v_proj\.bias$"),    r"model.layers.\1.self_attn.v_proj.bias"),
        (re.compile(r"^layers\.(\d+)\.attn\.o_proj\.weight$"),      r"model.layers.\1.self_attn.o_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.attn\.q_norm\.weight$"),     r"model.layers.\1.self_attn.q_norm.weight"),
        (re.compile(r"^layers\.(\d+)\.attn\.k_norm\.weight$"),     r"model.layers.\1.self_attn.k_norm.weight"),
        (re.compile(r"^layers\.(\d+)\.mlp\.gate_proj\.weight$"), r"model.layers.\1.mlp.gate_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.mlp\.up_proj\.weight$"),   r"model.layers.\1.mlp.up_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.mlp\.down_proj\.weight$"), r"model.layers.\1.mlp.down_proj.weight"),
        (re.compile(r"^layers\.(\d+)\.input_layernorm\.weight$"),          r"model.layers.\1.input_layernorm.weight"),
        (re.compile(r"^layers\.(\d+)\.post_attention_layernorm\.weight$"), r"model.layers.\1.post_attention_layernorm.weight"),
    ]
    for pattern, replacement in rev:
        if pattern.search(archon_key):
            return pattern.sub(replacement, archon_key)
    return None


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class Qwen35StateDictAdapter:
    """Load HuggingFace weights into nano-archon and export back."""

    def hf_to_archon(self, hf_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        archon_state: Dict[str, torch.Tensor] = {}
        skipped = []
        for hf_key, tensor in hf_state.items():
            archon_key = hf_key_to_archon(hf_key)
            if archon_key is not None:
                archon_state[archon_key] = tensor
            else:
                skipped.append(hf_key)
        if skipped:
            import warnings
            warnings.warn(f"Skipped {len(skipped)} HF keys (not mapped): {skipped[:5]}...")
        return archon_state

    def archon_to_hf(self, archon_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hf_state: Dict[str, torch.Tensor] = {}
        for archon_key, tensor in archon_state.items():
            hf_key = archon_key_to_hf(archon_key)
            if hf_key is not None:
                hf_state[hf_key] = tensor
        return hf_state


# ---------------------------------------------------------------------------
# Convenience: load HF checkpoint directly into a Qwen35Model
# ---------------------------------------------------------------------------

def load_hf_weights(model, checkpoint_path: str | Path, device="cpu"):
    """
    Load Qwen3.5 HuggingFace weights into a nano-archon Qwen35Model.
    Supports both single safetensors/bin files and sharded checkpoints.
    """
    from pathlib import Path as P

    path = P(checkpoint_path)
    adapter = Qwen35StateDictAdapter()

    # Try safetensors first (exclude macOS ._* metadata files)
    shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
    if not shard_files:
        shard_files = sorted(f for f in path.glob("pytorch_model*.bin") if not f.name.startswith("._"))

    if not shard_files:
        raise FileNotFoundError(f"No weight files found in {checkpoint_path}")

    merged: Dict[str, torch.Tensor] = {}
    for shard in shard_files:
        if shard.suffix == ".safetensors":
            try:
                from safetensors import safe_open
                with safe_open(str(shard), framework="pt", device=str(device)) as f:
                    merged.update({k: f.get_tensor(k) for k in f.keys()})
            except ImportError:
                raise ImportError("pip install safetensors")
        else:
            merged.update(torch.load(str(shard), map_location=device, weights_only=True))

    archon_state = adapter.hf_to_archon(merged)
    missing, unexpected = model.load_state_dict(archon_state, strict=False)

    if missing:
        import warnings
        warnings.warn(f"Missing keys when loading weights: {missing[:5]}")

    return model
