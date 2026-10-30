"""Offline image transport and BFCL flow regressions; no model quality claims."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import torch
from fastapi import HTTPException
from openai.types.chat import ChatCompletion
from probe_mac_vlm import red_png

from nano_areal.agent.bfcl import BFCLAgent
from nano_areal.clevr import load_clevr
from nano_areal.dataset import BFCLSample
from nano_areal.engine.messages import anthropic_request_to_openai
from nano_areal.engine.multimodal import prepare_single_image
from nano_areal.engine.proxy import OpenAIProxyClient, OpenAIProxyServer, ProxySession


def image_message():
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What color?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(red_png()).decode()
                },
            },
        ],
    }


def fake_processor():
    return SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 120, 2]]),
            "pixel_values": torch.ones(4, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )


class ImageContracts(unittest.TestCase):
    def test_messages_keeps_image_payload_and_order(self):
        source = {"type": "base64", "media_type": "image/png", "data": "eA=="}
        converted = anthropic_request_to_openai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "before"},
                            {"type": "image", "source": source},
                            {"type": "text", "text": "after"},
                        ],
                    }
                ]
            }
        )
        content = converted["messages"][0]["content"]
        self.assertEqual(
            [part["type"] for part in content], ["text", "image_url", "text"]
        )
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,eA==")

    def test_local_image_preparation_and_remote_rejection(self):
        ids, pixels, grid = prepare_single_image(fake_processor(), [image_message()])
        self.assertEqual(ids, [1, 120, 2])
        self.assertEqual(tuple(pixels.shape), (4, 24))
        self.assertEqual(grid.tolist(), [[1, 2, 2]])
        for url in (
            "https://example.com/image.png",
            "file:///etc/passwd",
            "data:image/png;base64,???",
        ):
            message = image_message()
            message["content"][1]["image_url"]["url"] = url
            with self.assertRaises(ValueError):
                prepare_single_image(fake_processor(), [message])
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            prepare_single_image(fake_processor(), [image_message(), image_message()])

    def test_clevr_local_loader_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "images/train").mkdir(parents=True)
            (root / "images/train/test.png").write_bytes(red_png())
            data = {
                "questions": [
                    {
                        "question": "How many?",
                        "answer": "1",
                        "question_index": 0,
                        "image_filename": "test.png",
                    }
                ]
            }
            question_file = root / "questions/CLEVR_train_questions.json"
            question_file.write_text(json.dumps(data))
            self.assertEqual(load_clevr(root)[0].answer, "1")
            data["questions"][0]["image_filename"] = (
                "../../questions/CLEVR_train_questions.json"
            )
            question_file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_clevr(root)


class ImageProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_images_without_local_processor_fail_closed(self):
        server = OpenAIProxyServer("http://localhost/v1", None)
        with self.assertRaises(HTTPException) as raised:
            await server._handle_chat_completion(
                {"messages": [image_message()]}, ProxySession("no-processor", "local")
            )
        self.assertEqual(raised.exception.status_code, 422)

    async def test_pixels_survive_capture_and_export(self):
        server = OpenAIProxyServer(
            "http://localhost/v1", None, processor=fake_processor()
        )
        payload = {
            "id": "image-call",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "prompt_token_ids": [1, 120, 2],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "red"},
                    "finish_reason": "stop",
                    "token_ids": [3, 9],
                    "logprobs": {
                        "content": [
                            {"token": "red", "logprob": -0.1, "top_logprobs": []},
                            {"token": "stop", "logprob": -0.2, "top_logprobs": []},
                        ]
                    },
                }
            ],
        }
        server._vllm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=ChatCompletion.model_validate(payload)
                    )
                )
            )
        )
        server._tokenize_text = lambda text: []
        session = ProxySession("image-session", "image-key")
        server._sessions[session.session_id] = session
        server._key_to_session[session.api_key] = session.session_id
        await server._handle_chat_completion({"messages": [image_message()]}, session)
        client = OpenAIProxyClient("http://localhost")
        client.session_id, client.api_key = session.session_id, session.api_key
        client._http = httpx.AsyncClient(
            base_url="http://localhost", transport=httpx.ASGITransport(app=server.app)
        )
        await client.__aexit__(None, None, None)
        trajectory = client.trajectories[0]
        self.assertEqual(trajectory.input_ids.tolist(), [1, 120, 2, 3, 9])
        torch.testing.assert_close(trajectory.pixel_values, torch.ones(4, 24))
        self.assertEqual(trajectory.image_grid_thw.tolist(), [[1, 2, 2]])

    async def test_bfcl_two_turn_tool_history_is_unchanged(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            number = len(requests)
            return httpx.Response(
                200,
                json={
                    "id": f"turn-{number}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call-{number}",
                                        "type": "function",
                                        "function": {
                                            "name": "ls",
                                            "arguments": '{"path":"."}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        sample = BFCLSample(
            "test",
            [
                [{"role": "user", "content": "List files"}],
                [{"role": "user", "content": "Again"}],
            ],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "ls",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
            [{"ls": {"path": ["."]}}] * 2,
            {},
            [],
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rewards = await BFCLAgent().run(
                sample,
                base_url="http://localhost/v1",
                api_key="test",
                http_client=client,
            )
        self.assertEqual(rewards, {"turn-1": 1.0, "turn-2": 1.0})
        self.assertEqual(requests[1]["tools"], requests[0]["tools"])
        self.assertEqual(
            [message["role"] for message in requests[1]["messages"]],
            ["system", "user", "assistant", "tool", "user"],
        )


if __name__ == "__main__":
    unittest.main()
