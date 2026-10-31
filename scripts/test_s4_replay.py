"""Read-only replay guards and block-aligned prefix accounting."""

import sys

import pytest
import torch
from diagnose_s4_replay import common_prefix, require_zero_adapter, response_span
from safetensors.torch import save_file
from test_archon_parity import tiny_config

from nano_areal.archon.qwen3vl import Qwen3VLModel, add_lora
from nano_areal.archon.s4_replay import CacheBlock, cached_text_logprobs
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


@pytest.mark.parametrize("chunk_size", [1, 4, 16])
def test_checkpoint_chunks_preserve_cross_turn_gradients_and_linear_storage(chunk_size):
    torch.manual_seed(61)
    model = Qwen3VLModel(tiny_config().to_dict()).float().train()
    add_lora(model)
    state = {}
    trajectories = [trajectory_from_turn({
        "input_ids": list(range(1, prompt + 17)),
        "response_mask": [0] * prompt + [1] * 16,
        "old_log_probs": [0] * (prompt + 16), "truncated": False,
    })[0] for prompt in (32, 64)]
    for index, trajectory in enumerate(trajectories):
        current = cached_text_logprobs(
            model, trajectory, state=state, checkpoint_layers=True,
            score_response=index == 1, decode_chunk_size=chunk_size)
        if index == 0:
            historical_value = state["cache"][0].prefix(32).value
            historical_value.retain_grad()
    assert state["reused_tokens"] == 32
    for block in state["cache"]:
        keys, values = block.tensors()
        assert sum(key.shape[-2] for key in keys) == 79
        assert [key.shape[-2] for key in keys] == [
            32, 32, *([chunk_size] * (15 // chunk_size)),
            *([15 % chunk_size] if 15 % chunk_size else [])]
        assert sum(value.numel() for value in values) == sum(
            key.numel() for key in keys)
    current.sum().backward()
    assert historical_value.grad is not None
    assert bool(historical_value.grad.abs().any())
    gradients = {name: parameter.grad.clone() for name, parameter in
                 model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    trajectory = trajectories[-1]
    ids = trajectory.input_ids.unsqueeze(0)
    logits = model(ids, logits_to_keep=torch.arange(63, 79))
    expected = logits.log_softmax(-1).gather(-1, ids[:, 64:, None]).squeeze(-1)
    torch.testing.assert_close(current, expected, rtol=1e-5, atol=1e-5)
    expected.sum().backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(gradients[name], parameter.grad,
                                       rtol=1e-4, atol=1e-5)


def test_cache_block_prefix_keeps_original_autograd_and_deep_history():
    key = torch.randn(1, 2, 16, 4, requires_grad=True)
    block = CacheBlock(key, key * 2)
    for _ in range(1100):
        block = CacheBlock(key[..., :1, :], key[..., :1, :] * 2, block)
    assert block.length == 1116
    assert block.prefix(0) is None
    prefix = block.prefix(17)
    assert prefix.length == 17 and prefix.previous.key is key
    partial = block.prefix(8)
    partial.key.sum().backward()
    torch.testing.assert_close(key.grad[..., :8, :], torch.ones_like(key[..., :8, :]))
    assert not bool(key.grad[..., 8:, :].any())


def test_shared_group_backward_matches_independent_complete_history_in_fp32():
    torch.manual_seed(53)
    model = Qwen3VLModel(tiny_config().to_dict()).float().train()
    add_lora(model)
    trajectories = [trajectory_from_turn({
        "input_ids": list(range(1, prompt + 17)),
        "response_mask": [0] * prompt + [1] * 16,
        "old_log_probs": [0] * (prompt + 16), "truncated": False,
    })[0] for prompt in (32, 64, 80)]
    expected = []
    for index in range(len(trajectories)):
        state = {}
        for previous in range(index + 1):
            current = cached_text_logprobs(
                model, trajectories[previous], state=state,
                score_response=previous == index, decode_chunk_size=16)
        expected.append(current.detach())
        (current.sum() * (index + 1)).backward()
    gradients = {name: parameter.grad.clone() for name, parameter in
                 model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    state = {}
    currents = []
    for trajectory in trajectories:
        currents.append(cached_text_logprobs(
            model, trajectory, state=state, checkpoint_layers=True,
            checkpoint_head=True, decode_chunk_size=16))
    for actual, reference in zip(currents, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    sum(current.sum() * (index + 1)
        for index, current in enumerate(currents)).backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name],
                                       rtol=1e-4, atol=1e-5)


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
