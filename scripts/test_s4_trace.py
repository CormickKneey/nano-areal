"""Contracts for observational S4 layer traces and strict comparisons."""

import pytest
import torch
from diagnose_s4_trace import differences, rms_differences
from test_archon_parity import tiny_config

from nano_areal.archon.model import RMSNorm
from nano_areal.archon.qwen3vl import Qwen3VLModel
from nano_areal.archon.s4_replay import cached_text_logprobs
from nano_areal.archon.s4_trace import LayerTrace, native_trace_hooks
from nano_areal.archon.s4_update import trajectory_from_turn


def test_trace_observes_last_rows_without_changing_cached_scores(tmp_path):
    model = Qwen3VLModel(tiny_config().to_dict()).float().eval()
    trajectory, _ = trajectory_from_turn({
        "input_ids": list(range(1, 36)), "response_mask": [0] * 30 + [1] * 5,
        "old_log_probs": [0] * 35, "truncated": False,
    })
    with torch.no_grad():
        expected = cached_text_logprobs(model, trajectory)
        trace = LayerTrace({"group": "example"})
        handles = native_trace_hooks(model.language_model, trace)
        try:
            actual = cached_text_logprobs(
                model, trajectory,
                before_segment=lambda start, end: trace.segment(0, start, end, 30),
                observe=trace.capture,
            )
        finally:
            for handle in handles:
                handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert trace.schedule == [[0, "prefill", 0, 30], [0, "decode", 30, 31]]
    assert trace.shapes["turn.0.prefill.embedding"] == [1, 30, 32]
    assert trace.tensors["turn.0.prefill.embedding"].shape == (1, 32)
    assert "turn.0.prefill.layers.0.attn.rope_query" in trace.tensors
    path = tmp_path / "trace.safetensors"
    trace.save(path)
    assert all(row["mae"] == 0 for row in trace.compare(path))
    with pytest.raises(ValueError, match="overwritten"):
        trace.save(path)


@pytest.mark.parametrize("variant", ["evidence", "schedule", "shapes", "stages"])
def test_trace_rejects_mismatched_provenance_or_schedule(tmp_path, variant):
    trace = LayerTrace({"group": "original"})
    trace.segment(0, 0, 4, 4)
    trace.capture("embedding", torch.ones(1, 4, 8))
    path = tmp_path / "trace.safetensors"
    trace.save(path)
    if variant == "evidence":
        trace.evidence = {"group": "changed"}
    elif variant == "schedule":
        trace.schedule[0][2] = 1
    elif variant == "shapes":
        trace.shapes["turn.0.prefill.embedding"] = [1, 3, 8]
    else:
        trace.tensors.pop("turn.0.prefill.embedding")
    with pytest.raises(ValueError, match="differ"):
        trace.compare(path)


def test_trace_rejects_duplicate_nonfinite_and_unbounded_capture(tmp_path):
    trace = LayerTrace({})
    with pytest.raises(ValueError, match="empty"):
        trace.save(tmp_path / "empty.safetensors")
    trace.segment(0, 0, 4, 4)
    trace.capture("norm", torch.full((1, 4, 8), float("nan")))
    with pytest.raises(ValueError, match="Duplicate"):
        trace.capture("norm", torch.ones(1, 4, 8))
    with pytest.raises(ValueError, match="nonfinite"):
        trace.save(tmp_path / "nonfinite.safetensors")
    for turn in range(1, 32):
        trace.segment(turn, 0, 4, 4)
    with pytest.raises(ValueError, match="budget"):
        trace.segment(32, 0, 4, 4)


def test_final_norm_allows_rowwise_equivalent_execution_shape(tmp_path):
    trace = LayerTrace({})
    trace.segment(0, 0, 4, 4)
    trace.capture("norm", torch.ones(1, 4, 8))
    path = tmp_path / "trace.safetensors"
    trace.save(path)
    trace.shapes["turn.0.prefill.norm"] = [1, 1, 8]
    assert trace.compare(path)[0]["mae"] == 0


def test_isolated_comparison_rejects_broadcasting_and_nonfinite_values():
    with pytest.raises(ValueError, match="shapes differ"):
        differences(torch.ones(1, 4), torch.ones(4))
    with pytest.raises(ValueError, match="Nonfinite"):
        differences(torch.tensor([float("nan")]), torch.zeros(1))


def test_shared_input_rms_preserves_the_native_reference():
    module = RMSNorm(8).bfloat16()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.1, 1.3, 8))
    hidden = torch.linspace(-1.1, 2.1, 8).reshape(1, 8).bfloat16()
    with torch.no_grad():
        expected = module(hidden).float()
        result = rms_differences(hidden, module.weight, expected, module.eps)
    assert result["native_rms"]["max"] == 0
