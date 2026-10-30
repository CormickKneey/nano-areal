"""Read-only replay guards and block-aligned prefix accounting."""

import sys

import pytest
import torch
from diagnose_s4_replay import common_prefix, require_zero_adapter, response_span
from safetensors.torch import save_file
from test_archon_parity import tiny_config

from nano_areal.archon.qwen3vl import Qwen3VLModel, add_lora
from nano_areal.archon.s4_replay import cached_text_logprobs
from nano_areal.archon.s4_update import trajectory_from_turn


def test_response_span_requires_complete_contiguous_suffix():
    turn = {"input_ids": [1, 2, 3, 4], "response_mask": [0, 0, 1, 1],
            "old_log_probs": [0, 0, -0.2, -0.3], "truncated": False}
    prompt, captured = response_span(turn)
    assert prompt == 2
    torch.testing.assert_close(torch.tensor(captured), torch.tensor([-0.2, -0.3]))
    with pytest.raises(ValueError, match="contiguous response suffix"):
        response_span({**turn, "response_mask": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="Malformed or truncated"):
        response_span({**turn, "truncated": True})


def test_prefix_only_reuses_matching_complete_blocks_before_last_prompt_token():
    previous = list(range(100))
    assert common_prefix(previous, previous, 95, 16) == 80
    assert common_prefix(previous, previous, 96, 16) == 96
    assert common_prefix(previous, [*range(35), -1], 99, 16) == 32
    assert common_prefix([], previous, 99, 16) == 0


@pytest.mark.parametrize("variant", ["zero", "changed", "nonfinite", "missing"])
def test_base_only_replay_never_ignores_changed_adapter(tmp_path, variant):
    weights = {"projection.lora_A.weight": torch.ones(2, 4),
               "projection.lora_B.weight": torch.zeros(4, 2)}
    if variant == "changed":
        weights["projection.lora_B.weight"][0, 0] = 0.001
    elif variant == "nonfinite":
        weights["projection.lora_A.weight"][0, 0] = float("nan")
    elif variant == "missing":
        weights.pop("projection.lora_B.weight")
    save_file(weights, str(tmp_path / "adapter_model.safetensors"))
    if variant == "zero":
        require_zero_adapter(tmp_path)
    else:
        with pytest.raises(ValueError, match="exactly zero-B"):
            require_zero_adapter(tmp_path)


def test_cached_native_matches_full_causal_forward_and_gradients():
    torch.manual_seed(42)
    model = Qwen3VLModel(tiny_config().to_dict()).float().eval()
    add_lora(model)
    ids = list(range(1, 36))
    trajectory, _ = trajectory_from_turn({
        "input_ids": ids, "response_mask": [0] * 30 + [1] * 5,
        "old_log_probs": [0] * 35, "truncated": False,
    })
    positions = torch.arange(29, 34)
    logits = model(torch.tensor([ids]), logits_to_keep=positions)
    full = logits.log_softmax(-1).gather(-1, torch.tensor([ids[30:]])[..., None]
                                        ).squeeze(-1)
    state = {}
    segments = []
    cached = cached_text_logprobs(
        model, trajectory,
        before_segment=lambda start, end: segments.append((start, end)),
        state=state,
    )
    assert segments == [(0, 30), (30, 31), (31, 32), (32, 33), (33, 34)]
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)
    assert state["reused_tokens"] == 0
    reused = cached_text_logprobs(model, trajectory, state=state)
    assert state["reused_tokens"] == 16
    torch.testing.assert_close(reused, full, rtol=1e-5, atol=1e-5)
    cached.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients)
    assert any(bool(value.any()) for value in gradients)
    expected_gradients = {name: parameter.grad.clone() for name, parameter in
                          model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    full.sum().backward()
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, expected_gradients[name],
                                       rtol=1e-4, atol=1e-5)
    changed, _ = trajectory_from_turn({
        "input_ids": [99, *ids[1:]], "response_mask": [0] * 30 + [1] * 5,
        "old_log_probs": [0] * 35, "truncated": False,
    })
    with torch.no_grad():
        cached_text_logprobs(model, changed, state=state)
    assert state["reused_tokens"] == 0


def test_cached_native_rejects_multimodal_and_noncontiguous_turns():
    model = Qwen3VLModel(tiny_config().to_dict()).eval()
    trajectory, _ = trajectory_from_turn({
        "input_ids": [1, 2, 3, 4], "response_mask": [0, 1, 0, 1],
        "old_log_probs": [0, -0.2, 0, -0.3], "truncated": False,
    })
    with pytest.raises(ValueError, match="contiguous response suffix"):
        cached_text_logprobs(model, trajectory)
    trajectory.pixel_values = torch.ones(1)
    with pytest.raises(ValueError, match="text only"):
        cached_text_logprobs(model, trajectory)


@pytest.mark.parametrize("seconds", ["0", "601", "nan", "inf"])
def test_native_diagnostic_rejects_unbounded_deadlines(monkeypatch, tmp_path, seconds):
    from diagnose_s4_operators import main

    monkeypatch.setattr(sys, "argv", [
        "diagnose_s4_operators.py", "--policy", "missing-policy.json",
        "--group", "missing-group", "--output", str(tmp_path / "new.json"),
        "--max-seconds", seconds,
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "new.json").exists()


@pytest.mark.parametrize("seconds", ["0", "601", "nan", "inf"])
def test_mlx_replay_rejects_unbounded_deadlines(monkeypatch, tmp_path, seconds):
    from diagnose_s4_replay import main

    monkeypatch.setattr(sys, "argv", [
        "diagnose_s4_replay.py", "--policy", "missing-policy.json",
        "--group", "missing-group", "--output", str(tmp_path / "new.json"),
        "--execution", "paged", "--max-seconds", seconds,
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "new.json").exists()
