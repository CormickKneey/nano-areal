"""Bound native cache replay, provenance checks and checkpointed LoRA gradients."""

from types import SimpleNamespace

import pytest
import torch
from test_archon_parity import tiny_config

from nano_areal.archon.qwen3vl import Qwen3VLModel, add_lora
from nano_areal.archon.s4_native import AlignedS4Runtime
from nano_areal.archon.s4_operators import affine_weights
from nano_areal.archon.s4_replay import cached_text_logprobs


def aligned_fixture():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")
    torch.manual_seed(47)
    config = tiny_config().to_dict()
    config["text_config"].update(hidden_size=128, head_dim=128, num_attention_heads=1,
                                  num_key_value_heads=1, intermediate_size=256,
                                  num_hidden_layers=2)
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
        trainer.model.train()
        runtime.select_turn(1)
        current = trainer.response_logprobs(runtime.history[1])
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
                activate_segment=runtime.activate_segment, score_response=index == 1)
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
