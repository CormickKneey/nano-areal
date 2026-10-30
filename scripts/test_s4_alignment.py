"""Offline contracts for served-base conversion."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from align_s4_base import hf_key
from probe_s4 import training_eligible
from probe_s4_training import complete_group

from nano_areal.archon.losses import group_advantages
from nano_areal.sync.policy import PolicyBinding


@pytest.mark.parametrize("source,expected", [
    ("language_model.model.embed_tokens.weight",
     "model.language_model.embed_tokens.weight"),
    ("language_model.model.layers.0.self_attn.q_proj.weight",
     "model.language_model.layers.0.self_attn.q_proj.weight"),
    ("vision_tower.patch_embed.proj.weight", "model.visual.patch_embed.proj.weight"),
])
def test_alignment_key_mapping(source, expected):
    assert hf_key(source) == expected


def test_alignment_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unsupported MLX checkpoint key"):
        hf_key("unknown.weight")


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_training_admits_complete_external_failures_without_relabeling(reward):
    row = {
        "passed": reward == 1.0, "reconciliation": {"matched": True},
        "session_logs": 1, "failed_proxy_requests": 0, "failed_dsh_attempts": 0,
        "cancelled_requests": 0, "tool_turns": 3, "exported_interactions": 1,
        "result": {"finish_reason": "completed", "verified_reward": reward,
                   "verification": {"runner": "pytest", "total": 1,
                                    "pass_rate": reward, "outcomes": [{
                                        "status": "passed" if reward else "failed",
                                    }]}},
    }
    trajectories = [SimpleNamespace(truncated=False)]
    assert training_eligible(row, trajectories)
    for field, value in (("failed_proxy_requests", 1), ("cancelled_requests", 1),
                         ("session_logs", 0), ("failed_dsh_attempts", 1)):
        assert not training_eligible({**row, field: value}, trajectories)
    assert not training_eligible(row, [SimpleNamespace(truncated=True)])
    row["result"]["verification"]["outcomes"][0]["status"] = "timeout"
    assert not training_eligible(row, trajectories)


def test_complete_group_preserves_external_episode_rewards(tmp_path):
    binding = PolicyBinding({"version": 0})
    rows = []
    for index, reward in enumerate([0.0, 1.0]):
        rows.append({"index": index, "passed": bool(reward),
                     "training_eligible": True,
                     "result": {"verified_reward": reward,
                                "verification": {"pass_rate": reward}}})
        episode = {"policy": {"model": binding.model, "version": 0},
                   "episode": index, "reward": reward,
                   "turns": [{"truncated": False, "discounted_reward": 0.123}]}
        (tmp_path / f"episode-{index}.json").write_text(json.dumps(episode))
    summary = {"planned": 2, "attempted": 2, "episodes": rows}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(summary))
    assert [item["reward"] for item in complete_group(tmp_path, binding)] == [0, 1]
    summary["attempted"] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete declared"):
        complete_group(tmp_path, binding)
    summary["attempted"] = 2
    rows[0]["training_eligible"] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="unreconciled"):
        complete_group(tmp_path, binding)


def test_all_success_advantages_stay_zero():
    import torch

    advantages, _ = group_advantages(torch.tensor([1.0, 1.0]), 2)
    assert advantages.tolist() == [0.0, 0.0]


def test_mlx_alignment_roundtrip(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists():
        pytest.skip("Repository-local MLX environment is unavailable")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    program = '''
import json
from pathlib import Path
import mlx.core as mx
from align_s4_base import convert
source, output = Path(SOURCE), Path(OUTPUT)
quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
(source / "config.json").write_text(json.dumps({"quantization": quantization}))
weight = mx.arange(128, dtype=mx.float32).reshape(2, 64).astype(mx.bfloat16)
packed, scales, biases = mx.quantize(weight, **quantization)
patch = mx.arange(48, dtype=mx.float32).reshape(2, 2, 2, 2, 3)
prefix = "language_model.model.embed_tokens"
mx.save_safetensors(str(source / "model.safetensors"), {
    prefix + ".weight": packed, prefix + ".scales": scales,
    prefix + ".biases": biases, "vision_tower.patch_embed.proj.weight": patch,
})
convert(source, output)
converted = {}
for shard in output.glob("*.safetensors"):
    converted.update(mx.load(str(shard)))
assert mx.array_equal(converted["model.language_model.embed_tokens.weight"],
                      mx.dequantize(packed, scales, biases, **quantization)).item()
assert mx.array_equal(converted["model.visual.patch_embed.proj.weight"],
                      patch.transpose(0, 4, 1, 2, 3)).item()
assert "quantization" not in json.loads((output / "config.json").read_text())
try:
    convert(source, output)
except FileExistsError:
    pass
else:
    raise AssertionError("Existing aligned snapshot must not be overwritten")
'''
    subprocess.run(
        [str(executable), "-c", program.replace("SOURCE", repr(str(source)))
         .replace("OUTPUT", repr(str(output)))],
        env={**os.environ, "PYTHONPATH": ".:scripts"},
        check=True, capture_output=True, text=True, timeout=60,
    )
    report = json.loads((output / "alignment.json").read_text())
    assert report["dequantized_modules"] == 1
    assert report["tensor_count"] == 2
