"""
Weight conversion between HuggingFace and nano-archon.

nano-archon parameter names mirror HF `Qwen3_5ForConditionalGeneration` with
the leading `model.` removed:

  HF                                          archon
  model.language_model.embed_tokens.weight    language_model.embed_tokens.weight
  model.language_model.layers.N.*             language_model.layers.N.*
  model.visual.*                              visual.*
  lm_head.weight                              lm_head.weight
  mtp.*                                       (dropped — MTP head is not trained)

Qwen3 dense checkpoints (`Qwen3ForCausalLM`) put the text stack directly under
`model.`; those keys are mapped to `language_model.*` too, so one adapter covers
both families.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import torch

_TEXT_PREFIXES = ("embed_tokens.", "layers.", "norm.")


def hf_key_to_archon(hf_key: str) -> str | None:
    """Translate a single HF state-dict key. Returns None if the key is dropped."""
    if hf_key.startswith("mtp.") or hf_key.startswith("model.mtp."):
        return None
    if hf_key == "lm_head.weight":
        return hf_key
    if hf_key.startswith("model.language_model."):
        return hf_key[len("model."):]
    if hf_key.startswith("model.visual."):
        return hf_key[len("model."):]
    if hf_key.startswith("model."):
        rest = hf_key[len("model."):]
        if rest.startswith(_TEXT_PREFIXES):
            return "language_model." + rest
    if hf_key.startswith(("language_model.", "visual.")):
        return hf_key
    return None


def archon_key_to_hf(archon_key: str, dense_layout: bool = False) -> str | None:
    """
    Reverse translation. `dense_layout=True` emits Qwen3ForCausalLM names
    (`model.layers.*`), otherwise Qwen3_5ForConditionalGeneration names.
    """
    if archon_key == "lm_head.weight":
        return archon_key
    if archon_key.startswith("language_model."):
        if dense_layout:
            return "model." + archon_key[len("language_model."):]
        return "model." + archon_key
    if archon_key.startswith("visual."):
        return "model." + archon_key
    return None


class Qwen35StateDictAdapter:
    """Load HuggingFace weights into nano-archon and export back."""

    def __init__(self, dense_layout: bool = False):
        self.dense_layout = dense_layout

    def hf_to_archon(self, hf_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        skipped = []
        for k, v in hf_state.items():
            ak = hf_key_to_archon(k)
            if ak is None:
                skipped.append(k)
            else:
                out[ak] = v
        real_skips = [k for k in skipped if not k.startswith(("mtp.", "model.mtp."))]
        if real_skips:
            warnings.warn(f"Skipped {len(real_skips)} HF keys (not mapped): {real_skips[:5]}...")
        return out

    def archon_to_hf(self, archon_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in archon_state.items():
            hk = archon_key_to_hf(k, self.dense_layout)
            if hk is not None:
                out[hk] = v
        return out


def read_hf_state_dict(checkpoint_path: str | Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    shards = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
    if not shards:
        shards = sorted(f for f in path.glob("pytorch_model*.bin") if not f.name.startswith("._"))
    if not shards:
        raise FileNotFoundError(f"No weight files found in {checkpoint_path}")
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        if shard.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(shard), framework="pt", device=str(device)) as f:
                for k in f.keys():
                    merged[k] = f.get_tensor(k)
        else:
            merged.update(torch.load(str(shard), map_location=device, weights_only=True))
    return merged


def load_hf_weights(model, checkpoint_path: str | Path, device="cpu", strict_text: bool = True):
    """
    Load HF weights into a nano-archon `Qwen35Model` (single or sharded
    safetensors/bin). Vision weights are skipped when the model is text-only.
    """
    hf_state = read_hf_state_dict(checkpoint_path, device)
    archon_state = Qwen35StateDictAdapter().hf_to_archon(hf_state)
    if model.visual is None:
        archon_state = {k: v for k, v in archon_state.items() if not k.startswith("visual.")}
    if model.lm_head is None:
        archon_state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(archon_state, strict=False)
    if unexpected:
        warnings.warn(f"Unexpected keys when loading weights: {unexpected[:5]}")
    if missing:
        msg = f"Missing keys when loading weights: {missing[:5]} (+{max(0, len(missing) - 5)} more)"
        if strict_text and any(k.startswith("language_model.") for k in missing):
            raise RuntimeError(msg)
        warnings.warn(msg)
    return model
