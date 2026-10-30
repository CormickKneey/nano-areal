"""Probe production vLLM HTTP and a real dsh Messages roundtrip."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import tempfile
from pathlib import Path

import httpx
from probe_mac_vlm import memory_snapshot, red_png
from transformers import AutoProcessor, AutoTokenizer

from nano_areal.agent.dsh import DshAgent
from nano_areal.engine.proxy import OpenAIProxyServer, ProxySession


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    report = {
        "backend": "production_vllm_metal_http",
        "status": "failed",
        "memory_before": memory_snapshot(args.server_pid),
        "checks": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proxy = OpenAIProxyServer(
        args.base_url + "/v1",
        tokenizer,
        port=args.proxy_port,
        model_name=args.model,
        processor=(
            AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
            if args.proxy_image
            else None
        ),
    )
    captured_tools = []
    tokenize = proxy._tokenize_messages

    def capture_prompt(messages, tools=None):
        captured_tools.append(tools or [])
        return tokenize(messages, tools)

    proxy._tokenize_messages = capture_prompt
    try:
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            for name in ("text", "image"):
                content = "Reply OK."
                if name == "image":
                    content = [
                        {"type": "text", "text": "Name the color in one word."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(red_png()).decode()
                            },
                        },
                    ]
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 32,
                    "temperature": 1,
                    "top_p": 1,
                    "top_k": -1,
                    "logprobs": True,
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                }
                response = await client.post(
                    args.base_url + "/v1/chat/completions", json=payload
                )
                response.raise_for_status()
                result = response.json()
                choice = result["choices"][0]
                entries = choice["logprobs"]["content"]
                ids = choice["token_ids"]
                assert ids and len(ids) == result["usage"]["completion_tokens"]
                assert [
                    int(entry["token"].removeprefix("token_id:")) for entry in entries
                ] == ids
                assert all(
                    math.isfinite(entry["logprob"]) and entry["logprob"] <= 0
                    for entry in entries
                )
                assert (
                    len(result["prompt_token_ids"]) == result["usage"]["prompt_tokens"]
                )
                tokenized = await client.post(
                    args.base_url + "/tokenize",
                    json={
                        "model": args.model,
                        "messages": payload["messages"],
                        "add_generation_prompt": True,
                    },
                )
                tokenized.raise_for_status()
                assert tokenized.json()["tokens"] == result["prompt_token_ids"]
                report["checks"][name] = result
        await proxy.start()
        if args.proxy_image:
            image_session = ProxySession("image-smoke", "local-image-key")
            proxy._sessions[image_session.session_id] = image_session
            proxy._key_to_session[image_session.api_key] = image_session.session_id
            async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
                response = await client.post(
                    proxy.base_url + "/v1/messages",
                    headers={"x-api-key": image_session.api_key},
                    json={
                        "model": args.model,
                        "max_tokens": 32,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Name the color in one word.",
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": base64.b64encode(
                                                red_png()
                                            ).decode(),
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                )
                response.raise_for_status()
                assert "event: message_stop" in response.text
            image_record = image_session.last_interaction()
            assert image_record.response_content.strip().lower() == "red"
            image_trajectory = image_record.to_trajectory()
            report["image_proxy"] = {
                "response": image_record.response_content,
                "prompt_tokens": image_record.response_mask.count(0),
                "output_tokens": image_record.response_mask.count(1),
                "pixel_values_shape": list(image_trajectory.pixel_values.shape),
                "image_grid_thw": image_trajectory.image_grid_thw.tolist(),
                "behavior_logprobs": image_record.token_logprobs,
            }
            captured_tools.clear()
        session = ProxySession("production-smoke", "local-smoke")
        proxy._sessions[session.session_id] = session
        proxy._key_to_session[session.api_key] = session.session_id
        agent = DshAgent(
            model_name=args.model,
            max_output_tokens=64,
            patches=(str(Path("scripts/dsh_patch.yml").resolve()),),
        )
        with tempfile.TemporaryDirectory(prefix="s1-production-") as directory:
            await agent.run(
                {"prompt": "Reply S1_OK. Do not use tools.", "workspace": directory},
                base_url=proxy.base_url + "/v1",
                api_key=session.api_key,
            )
        record = session.last_interaction()
        assert agent.last_result["finish_reason"] == "completed"
        assert "S1_OK" in agent.last_result["response"]
        assert record is not None and len(session.interactions) == 1
        assert len(captured_tools) == 1 and len(captured_tools[0]) == 3
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            tokenized = await client.post(
                args.base_url + "/tokenize",
                json={
                    "model": args.model,
                    "messages": record.messages,
                    "tools": captured_tools[0],
                    "add_generation_prompt": True,
                },
            )
            tokenized.raise_for_status()
            prompt_count = record.response_mask.count(0)
            assert tokenized.json()["tokens"] == record.input_ids[:prompt_count]
        report["dsh"] = {
            **agent.last_result,
            "prompt_tokens": record.response_mask.count(0),
            "output_tokens": record.response_mask.count(1),
            "behavior_logprobs": record.token_logprobs,
            "interaction_count": len(session.interactions),
            "tool_schema_count": len(captured_tools[0]),
            "backend_tokenize_parity": True,
        }
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        await proxy.stop()
        report["memory_after"] = memory_snapshot(args.server_pid)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    parser.add_argument(
        "--model-path", default=".s1-cache/models/Qwen3-VL-4B-Instruct-4bit"
    )
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--proxy-port", type=int, default=8001)
    parser.add_argument("--proxy-image", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s1/vllm-http.json")
    )
    asyncio.run(run(parser.parse_args()))
