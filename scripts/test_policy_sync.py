"""Fail-closed identity and native PEFT export regressions."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from fastapi import HTTPException
from openai.types.chat import ChatCompletion
from safetensors.torch import load_file

from nano_areal.archon.qwen3vl import LoRALinear
from nano_areal.archon.qwen3vl_lora import NativeQwen3VLLoRATrainer
from nano_areal.engine.proxy import OpenAIProxyServer, ProxySession
from nano_areal.sync.policy import PolicyBinding, checkpoint_files, file_digest


def test_binding_rejects_drift(tmp_path):
    weight = tmp_path / "weights"
    weight.write_bytes(b"original")
    files = {str(weight): file_digest(weight)}
    binding = PolicyBinding({
        "version": 3, "adapter": str(tmp_path), "base_model": "base",
        "inference_files": files, "training_files": files, "adapter_files": files,
    })
    binding.verify_files()
    binding.verify_response(binding.model, 3)
    with pytest.raises(ValueError, match="rollout version"):
        binding.verify_response(binding.model, 2)
    with pytest.raises(ValueError, match="Backend model"):
        binding.verify_response("base", 3)
    catalog = {"data": [{
        "id": binding.model, "root": str(tmp_path), "parent": "base",
    }, {"id": "base", "root": str(tmp_path)}]}
    binding.verify_catalog(catalog)
    catalog["data"][1]["root"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="base path"):
        binding.verify_catalog(catalog)
    catalog["data"][1]["root"] = str(tmp_path)
    catalog["data"][0]["parent"] = "other"
    with pytest.raises(ValueError, match="parent"):
        binding.verify_catalog(catalog)
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Policy file changed"):
        binding.verify_files()


def test_manifest_identity_cannot_be_renamed(tmp_path):
    path = tmp_path / "policy.json"
    PolicyBinding({"version": 0}).write(path)
    payload = json.loads(path.read_text())
    payload["version"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        PolicyBinding.read(path)


def test_checkpoint_identity_includes_loader_metadata(tmp_path):
    names = (
        "model.safetensors", "config.json", "model.safetensors.index.json",
        "generation_config.json", "chat_template.jinja",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"example")
    assert set(checkpoint_files(tmp_path)) == {
        str((tmp_path / name).resolve()) for name in names
    }


def test_native_peft_export_preserves_weights(tmp_path):
    attention = SimpleNamespace(**{
        projection: LoRALinear(torch.nn.Linear(6, 4), rank=2, alpha=4)
        for projection in ("q_proj", "v_proj")
    })
    trainer = NativeQwen3VLLoRATrainer.__new__(NativeQwen3VLLoRATrainer)
    trainer.model = SimpleNamespace(language_model=SimpleNamespace(
        layers=[SimpleNamespace(attn=attention)]
    ))
    target = tmp_path / "adapter"
    trainer.export_peft_adapter(target)
    weights = load_file(str(target / "adapter_model.safetensors"))
    assert len(weights) == 4
    for projection in ("q_proj", "v_proj"):
        for side in ("a", "b"):
            key = (
                "base_model.model.model.language_model.layers.0."
                f"self_attn.{projection}.lora_{side.upper()}.weight"
            )
            assert torch.equal(weights[key], getattr(
                getattr(attention, projection), f"lora_{side}"
            ))
    assert json.loads((target / "adapter_config.json").read_text())["lora_alpha"] == 4
    with pytest.raises(FileExistsError):
        trainer.export_peft_adapter(target)


@pytest.mark.parametrize("model,version,accepted", [
    ("bound", 0, True), ("base", 0, False), ("bound", 1, False),
])
def test_proxy_binds_native_token_response(model, version, accepted):
    async def run():
        binding = PolicyBinding({"version": 0})
        server = OpenAIProxyServer("http://localhost/v1", None)
        server.model_name = binding.model
        server.policy_binding = binding
        server.expected_policy_version = version
        server._tokenize_messages = lambda *args: [1, 2]
        server._tokenize_text = lambda *args: [3]
        response = ChatCompletion.model_validate({
            "id": "test", "object": "chat.completion", "created": 0,
            "model": binding.model if model == "bound" else model,
            "prompt_token_ids": [1, 2],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "one"},
                "token_ids": [3], "finish_reason": "stop",
                "logprobs": {"content": [{
                    "token": "one", "logprob": -0.5, "top_logprobs": [],
                }]},
            }],
        })
        server._vllm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(return_value=response)
        )))
        session = ProxySession("session", "key")
        if accepted:
            await server._handle_chat_completion({"messages": []}, session)
            assert session.last_interaction().input_ids == [1, 2, 3]
            exported = session.last_interaction().to_trajectory()
            assert exported.policy_model == binding.model
            assert exported.policy_version == 0
            server.policy_binding = PolicyBinding({"version": 1})
            with pytest.raises(HTTPException, match="active session"):
                await server._handle_chat_completion({"messages": []}, session)
            server.policy_binding = None
            with pytest.raises(HTTPException, match="active session"):
                await server._handle_chat_completion({"messages": []}, session)
        else:
            with pytest.raises(HTTPException, match="bound policy"):
                await server._handle_chat_completion({"messages": []}, session)
            assert not session.interactions
    asyncio.run(run())
