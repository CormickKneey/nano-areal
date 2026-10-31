"""Bound native cache replay, provenance checks and checkpointed LoRA gradients."""

from types import SimpleNamespace

import pytest
import torch
from test_archon_parity import tiny_config
from torch.utils.checkpoint import CheckpointPolicy

from nano_areal.archon import s4_replay
from nano_areal.archon.qwen3vl import LoRALinear, Qwen3VLModel, add_lora
from nano_areal.archon.s4_native import AlignedS4Runtime
from nano_areal.archon.s4_operators import affine_decode_weights, affine_weights
from nano_areal.archon.s4_replay import cached_text_logprobs


def aligned_fixture():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")
    torch.manual_seed(47)
    config = tiny_config().to_dict()
    config["text_config"].update(hidden_size=128, head_dim=128, num_attention_heads=1,
                                  num_key_value_heads=1, intermediate_size=256,
                                  num_hidden_layers=2, vocab_size=256)
    config["tie_word_embeddings"] = True
    config["vision_config"]["out_hidden_size"] = 128
    model = Qwen3VLModel(config).to(device="mps", dtype=torch.bfloat16)
    add_lora(model)
    quantized = {}
    modules = {name.removesuffix(".base"): module
               for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)
               and name.startswith("language_model.layers.")}
    modules["language_model.embed_tokens"] = model.language_model.embed_tokens
    with torch.no_grad():
        for name, module in modules.items():
            rows, columns = module.weight.shape
            packed = torch.randint(0, 2 ** 32, (rows, columns // 8), dtype=torch.int64)
            scales = torch.full((rows, columns // 64), 0.015625, dtype=torch.bfloat16)
            biases = torch.randn(rows, columns // 64).bfloat16() * 0.02 - 0.1
            quantized[name] = packed, scales, biases
            module.weight.copy_(
                affine_weights(packed, scales, biases).bfloat16().to("mps"))
    episodes = [{"turns": [{"input_ids": list(range(1, prompt + 4)),
                             "response_mask": [0] * prompt + [1] * 3,
                             "old_log_probs": [0] * (prompt + 3), "truncated": False}
                            for prompt in (32, 64)]}]
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    trainer = SimpleNamespace(model=model, parameters=parameters,
                              response_logprobs=lambda trajectory: None)
    return trainer, quantized, episodes


def test_bound_native_checkpoint_cache_has_complete_history_gradients():
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        with torch.no_grad():
            for index in range(2):
                runtime.select_turn(index)
                observed = trainer.response_logprobs(runtime.history[index])
        with torch.no_grad(), trainer.model.disable_adapter():
            runtime.select_turn(0)
            trainer.response_logprobs(runtime.history[0])
        reference_states = {enabled: state for enabled, state in runtime.states.items()
                            if not any(enabled)}
        assert len(reference_states) == 1
        trainer.model.train()
        runtime.select_turn(1)
        current = trainer.response_logprobs(runtime.history[1])
        assert runtime.states == reference_states
        torch.testing.assert_close(current, observed, rtol=0, atol=0)
        current.sum().backward()
        checkpoint_gradients = {name: parameter.grad.clone()
                                for name, parameter in trainer.parameters.items()}
        for parameter in trainer.parameters.values():
            parameter.grad = None
        state = {}
        for index, trajectory in enumerate(runtime.history):
            expected = cached_text_logprobs(
                trainer.model, trajectory, project_head=runtime.project_head,
                state=state, attention_kernel=runtime.attention,
                activate_segment=runtime.activate_segment, score_response=index == 1,
                decode_chunk_size=16, batch_decode=True)
        expected.sum().backward()
        for name, parameter in trainer.parameters.items():
            assert bool(torch.isfinite(checkpoint_gradients[name]).all())
            torch.testing.assert_close(checkpoint_gradients[name], parameter.grad,
                                       rtol=1 / 128, atol=1e-5)
        assert any(bool(value.abs().any()) for value in checkpoint_gradients.values())
        assert runtime.contract()["decode_partition_size"] == 512
        runtime.verify_sources()
    finally:
        runtime.close()
    assert not hasattr(trainer, "select_turn")
    assert all(module.weight.dtype == torch.bfloat16
               for module in trainer.model.modules()
               if isinstance(module, torch.nn.Linear))


def test_bound_runtime_rejects_incomplete_projection_coverage():
    trainer, quantized, episodes = aligned_fixture()
    quantized.pop("language_model.layers.0.attn.k_proj")
    with pytest.raises(ValueError, match="coverage differs"):
        AlignedS4Runtime(trainer, quantized, episodes)
    assert not hasattr(trainer, "select_turn")


def test_group_schedule_matches_complete_history_replay_gradients():
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        trainer.model.train()
        expected_values = []
        for index, trajectory in enumerate(runtime.history):
            runtime.select_turn(index)
            current = trainer.response_logprobs(trajectory)
            expected_values.append(current.detach())
            (current.sum() * (index + 1)).backward()
        expected_gradients = {name: parameter.grad.clone()
                              for name, parameter in trainer.parameters.items()}
        for parameter in trainer.parameters.values():
            parameter.grad = None
        saved_shapes = []

        def pack(tensor):
            saved_shapes.append(tensor.shape)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            currents = trainer.response_group_logprobs(runtime.history)
        assert not runtime.states and not runtime.group_replay
        vocab_size = trainer.model.config["text_config"]["vocab_size"]
        assert not any(shape == (1, 1, vocab_size)
                       for shape in saved_shapes)
        for actual, expected in zip(currents, expected_values, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sum(current.sum() * (index + 1)
            for index, current in enumerate(currents)).backward()
        actual_gradient = torch.cat([parameter.grad.flatten()
                                     for parameter in trainer.parameters.values()])
        expected_gradient = torch.cat([value.flatten()
                                       for value in expected_gradients.values()])
        assert bool(torch.isfinite(actual_gradient).all())
        relative_error = (actual_gradient - expected_gradient).norm()
        assert float(relative_error / expected_gradient.norm()) <= 1 / 128
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1 / 128,
            atol=float(expected_gradient.abs().max()) / 128)
        assert any(bool(parameter.grad.abs().any())
                   for parameter in trainer.parameters.values())
        with pytest.raises(ValueError, match="complete declared"):
            trainer.response_group_logprobs(runtime.history[:-1])
    finally:
        runtime.close()
    assert not hasattr(trainer, "response_group_logprobs")


def test_bound_head_shares_fp32_bias_storage_across_differentiable_tokens():
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        weights, biases = runtime.head
        assert biases.dtype == torch.float32 and not biases.requires_grad
        hidden = torch.randn(1, 1, weights.shape[-1], device="mps",
                             dtype=torch.bfloat16, requires_grad=True)
        saved_biases = []

        def pack(tensor):
            if tensor.shape in (biases.shape, biases.T.shape):
                saved_biases.append(tensor.untyped_storage().data_ptr())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            current = sum(runtime.project_head(hidden, index, index + 1)
                          for index in range(3))
        assert len(saved_biases) == 3
        assert set(saved_biases) == {biases.untyped_storage().data_ptr()}
        current.sum().backward()
        gradient = hidden.grad.clone()
        hidden.grad = None
        original = quantized["language_model.embed_tokens"][2].to("mps")
        expected = sum(affine_decode_weights(hidden, weights, original)
                       for _ in range(3))
        torch.testing.assert_close(current, expected, rtol=0, atol=0)
        expected.sum().backward()
        torch.testing.assert_close(gradient, hidden.grad, rtol=0, atol=0)
    finally:
        runtime.close()


def test_bound_runtime_rejects_unselected_and_mismatched_turns():
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        with pytest.raises(ValueError, match="Select a bound turn"):
            trainer.response_logprobs(runtime.history[0])
        runtime.select_turn(1)
        with pytest.raises(ValueError, match="differs from the selected"):
            trainer.response_logprobs(runtime.history[0])
        with pytest.raises(ValueError, match="outside"):
            runtime.select_turn(2)
    finally:
        runtime.close()


@pytest.mark.parametrize("projections,count,storage", [
    (("gate_proj", "up_proj", "down_proj"), 24, 20480),
    (("gate_proj", "down_proj"), 16, 12288),
])
def test_selective_replay_retains_only_bounded_decode_mlp_products(
    monkeypatch, projections, count, storage,
):
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    original = s4_replay.create_selective_checkpoint_contexts
    retained = []

    def contexts(policy):
        def record(context, operation, *arguments, **keywords):
            decision = policy(context, operation, *arguments, **keywords)
            if not context.is_recompute and decision == CheckpointPolicy.MUST_SAVE:
                left, right = arguments
                outputs = (right.shape[0] if operation == torch.ops.aten.linear.default
                           else right.shape[1])
                retained.append(outputs * left.element_size())
            return decision

        return original(record)

    monkeypatch.setattr(s4_replay, "create_selective_checkpoint_contexts", contexts)
    try:
        trainer.model.train()
        state = {}
        currents = [cached_text_logprobs(
            trainer.model, trajectory, state=state,
            project_head=runtime.project_head, attention_kernel=runtime.attention,
            activate_segment=runtime.activate_segment, checkpoint_layers=True,
            checkpoint_head=True, decode_chunk_size=16, retain_decode_mlp=True,
            retained_mlp_projections=projections)
            for trajectory in runtime.history]
        assert len(retained) == count
        assert sum(retained) == storage
        sum(current.sum() for current in currents).backward()
        assert len(retained) == count
    finally:
        runtime.close()


@pytest.mark.parametrize("changed", [False, True])
def test_training_group_reference_identity_requires_exact_zero_adapter(changed):
    trainer, quantized, episodes = aligned_fixture()
    if changed:
        with torch.no_grad():
            for module in trainer.model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_b.fill_(0.01)
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        trainer.model.train()
        stages = []
        currents, references, mode = trainer.training_group_logprobs(
            runtime.history, stages.append)
        assert all(current.requires_grad for current in currents)
        assert all(not reference.requires_grad for reference in references)
        assert stages == (["forward"] * 2 + ["reference"] * 2 if changed
                          else ["forward"] * 2)
        assert ("exact-zero" in mode) != changed
        with torch.no_grad(), trainer.model.disable_adapter():
            expected = trainer.response_group_logprobs(runtime.history)
        for reference, actual in zip(references, expected, strict=True):
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)
        sum(current.sum() for current in currents).backward()
        assert any(bool(parameter.grad.abs().any())
                   for parameter in trainer.parameters.values())
    finally:
        runtime.close()
    assert not hasattr(trainer, "training_group_logprobs")


def test_bound_runtime_has_a_hard_segment_deadline():
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        runtime.select_turn(0)
        runtime.started -= 601
        with torch.no_grad(), pytest.raises(TimeoutError, match="600-second"):
            trainer.response_logprobs(runtime.history[0])
    finally:
        runtime.close()


def test_bound_runtime_renews_update_budget_only_after_full_preflight():
    trainer, quantized, episodes = aligned_fixture()
    runtime = AlignedS4Runtime(trainer, quantized, episodes)
    try:
        with pytest.raises(ValueError, match="full preflight"):
            runtime.start_update()
        with torch.no_grad():
            for index in range(len(runtime.history)):
                runtime.select_turn(index)
                trainer.response_logprobs(runtime.history[index])
        previous = runtime.started
        runtime.start_update()
        assert runtime.started > previous and not runtime.states
    finally:
        runtime.close()
