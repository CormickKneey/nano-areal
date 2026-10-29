"""No-download regression tests for the real-model RL bridge."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import torch
from fastapi import HTTPException
from openai.types.chat import ChatCompletion

from nano_areal.archon.losses import compute_policy_loss, group_advantages
from nano_areal.engine.mlx_server import collect_tokens
from nano_areal.engine.proxy import OpenAIProxyClient, OpenAIProxyServer, ProxySession
from nano_areal.reward.dsh_task import external_test_pass_rate


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_metal_dsh_entry_rejects_unported_architecture(self):
        import train

        with self.assertRaisesRegex(NotImplementedError, "run_agentic_rl.py"):
            await train.main_async(SimpleNamespace(agent="dsh", backend="metal"))

    async def capture(self, exact=None, logprobs=None):
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [1, 2],
            __call__=lambda *args, **kwargs: SimpleNamespace(input_ids=[3]),
        )
        server = OpenAIProxyServer("http://127.0.0.1:8000/v1", tokenizer)
        server._tokenize_text = lambda text: [3]
        payload = {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "one"},
                    "finish_reason": "stop",
                    "logprobs": logprobs,
                }
            ],
        }
        if exact is not None:
            payload["nano_areal"] = exact
        server._vllm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=ChatCompletion.model_validate(payload)
                    ),
                )
            )
        )
        server.expected_policy_version = 0
        session = ProxySession("session", "local-test-key")
        await server._handle_chat_completion(
            {"messages": [{"role": "user", "content": "test"}]},
            session,
        )
        return session.last_interaction()

    async def test_exact_backend_ids_include_stop_without_retokenizing(self):
        record = await self.capture(
            {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 9],
                "token_logprobs": [-0.5, -0.1],
                "policy_version": 0,
            }
        )
        self.assertEqual(record.input_ids, [1, 2, 3, 9])
        self.assertEqual(record.response_mask, [0, 0, 1, 1])
        self.assertEqual(record.token_logprobs, [-0.5, -0.1])

    async def test_missing_logprobs_fail_closed(self):
        with self.assertRaises(HTTPException):
            await self.capture()

    async def test_prompt_parity_and_policy_version_fail_closed(self):
        for prompt, version in (([1, 7], 0), ([1, 2], 1)):
            with self.assertRaises(HTTPException):
                await self.capture(
                    {
                        "prompt_token_ids": prompt,
                        "completion_token_ids": [3],
                        "token_logprobs": [-0.5],
                        "policy_version": version,
                    }
                )

    async def test_nonfinite_and_misaligned_logprobs_rejected(self):
        for values in ([float("nan")], [], [0.1], [-0.5, -0.2]):
            with self.assertRaises(HTTPException):
                await self.capture(
                    {
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3],
                        "token_logprobs": values,
                        "policy_version": 0,
                    }
                )


class ExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_preserves_truncation_and_closes_client(self):
        def handler(request):
            if request.url.path == "/export_trajectories":
                return httpx.Response(200, json=[{
                    "input_ids": [1, 2], "response_mask": [0, 1],
                    "logprobs": [0, -0.5], "reward": 0, "truncated": True,
                }])
            return httpx.Response(200, json={"ok": True})

        client = OpenAIProxyClient("http://127.0.0.1:8001")
        client._http = httpx.AsyncClient(
            base_url=client.proxy_base_url, transport=httpx.MockTransport(handler),
        )
        client.session_id, client.api_key = "test-session", "local-test-key"
        await client.__aexit__(None, None, None)
        self.assertTrue(client.trajectories[0].truncated)
        self.assertTrue(client._http.is_closed)

    async def test_successful_episode_does_not_hide_export_failure(self):
        client = OpenAIProxyClient("http://127.0.0.1:8001")
        client._http = httpx.AsyncClient(
            base_url=client.proxy_base_url,
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        )
        client.session_id, client.api_key = "test-session", "local-test-key"
        with self.assertRaises(httpx.HTTPStatusError):
            await client.__aexit__(None, None, None)
        self.assertTrue(client._http.is_closed)


class LossTests(unittest.TestCase):
    def test_microbatch_gradient_equals_complete_group_gradient(self):
        policy = torch.tensor(
            [[-1.1, -0.1], [-0.9, -0.3], [-0.8, -0.2]], requires_grad=True
        )
        old = policy.detach() + 0.1
        reference = policy.detach() - 0.2
        rewards = torch.tensor([0.0, 1.0, 0.0])
        masks = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        loss, _ = compute_policy_loss(
            "grpo",
            policy,
            reference,
            rewards,
            masks,
            old_log_probs=old,
            group_ids=["task"] * 3,
        )
        expected = torch.autograd.grad(loss, policy)[0]
        advantages, _ = group_advantages(rewards, None, group_ids=["task"] * 3)
        total = 0.0
        for index in range(3):
            local, _ = compute_policy_loss(
                "grpo",
                policy[index : index + 1],
                reference[index : index + 1],
                rewards[index : index + 1],
                masks[index : index + 1],
                old_log_probs=old[index : index + 1],
                advantages=advantages[index : index + 1],
            )
            total = total + local * masks[index].sum() / masks.sum()
        actual = torch.autograd.grad(total, policy)[0]
        torch.testing.assert_close(actual, expected)

    def test_precomputed_advantage_shape_is_checked(self):
        with self.assertRaises(ValueError):
            compute_policy_loss(
                "grpo",
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1),
                torch.ones(1, 2),
                advantages=torch.zeros(2),
            )


class HiddenTests(unittest.TestCase):
    def test_verifier_sees_response_after_episode_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, hidden = root / "task", root / "hidden"
            workspace.mkdir()
            hidden.mkdir()
            (hidden / "test_response.py").write_text(
                "import os\nfrom pathlib import Path\n"
                'assert Path(os.environ["TASK_RESPONSE_PATH"]).read_text() == "yes"\n'
            )
            self.assertEqual(external_test_pass_rate(workspace, hidden, "yes"), 1)
            self.assertEqual(external_test_pass_rate(workspace, hidden, "no"), 0)
            self.assertEqual(list(workspace.iterdir()), [])
            with self.assertRaises(ValueError):
                external_test_pass_rate(workspace, workspace, "yes")


class MLXCollectorTests(unittest.TestCase):
    def test_length_terminal_does_not_duplicate_token(self):
        chunks = [
            SimpleNamespace(
                text="yes",
                token=1,
                logprobs=torch.tensor([-2.0, -0.5]),
                finish_reason=None,
                token_ids=None,
            ),
            SimpleNamespace(
                text="",
                token=1,
                logprobs=torch.tensor([-2.0, -0.5]),
                finish_reason="length",
                token_ids=[1],
                generation_tokens=1,
                prompt_tokens=2,
            ),
        ]
        result = collect_tokens(chunks)
        self.assertEqual(result["tokens"], [1])
        self.assertEqual(result["logprobs"], [-0.5])


@unittest.skipUnless(importlib.util.find_spec("mlx"), "MLX is optional")
class AdapterLayoutTests(unittest.TestCase):
    def test_peft_transpose_and_alpha_match_mlx_lora(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx_vlm.trainer.peft.lora_layers import LoRALinear

        from nano_areal.archon.qwen3vl_lora import Qwen3VLLoRATrainer

        generator = torch.Generator().manual_seed(17)
        adapter_a = torch.randn(8, 16, generator=generator) * 0.1
        adapter_b = torch.randn(12, 8, generator=generator) * 0.1
        inputs = torch.randn(2, 16, generator=generator)
        trainer = Qwen3VLLoRATrainer.__new__(Qwen3VLLoRATrainer)
        prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
        trainer.parameters = {
            prefix + ".lora_A.default.weight": adapter_a,
            prefix + ".lora_B.default.weight": adapter_b,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            trainer.export_mlx_adapter(path)
            weights = mx.load(str(path))
        base = nn.Linear(16, 12, bias=False)
        base.weight = mx.zeros((12, 16))
        layer = LoRALinear.from_base(base, r=8, scale=2.0)
        key = "language_model.model.layers.0.self_attn.q_proj"
        layer.lora_a = weights[key + ".lora_a"]
        layer.lora_b = weights[key + ".lora_b"]
        actual = np.array(layer(mx.array(inputs.numpy())))
        expected = ((inputs @ adapter_a.T) @ adapter_b.T * 2).numpy()
        np.testing.assert_allclose(actual, expected, atol=0.003, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
