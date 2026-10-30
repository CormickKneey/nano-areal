"""Compare the unchanged two-turn fixture and an explicit-path diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from nano_areal.agent.bfcl import BFCLAgent
from nano_areal.dataset import BFCLSample


async def probe(args):
    report = {"parser": args.parser, "fixtures": {}}
    tool = {
        "type": "function",
        "function": {
            "name": "ls",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
    for label, turns in (
        ("exact", ["List files", "Again"]),
        (
            "explicit-path",
            [
                'List files in directory . using ls(path=".").',
                "Again, call ls for path .",
            ],
        ),
    ):
        responses = []
        requests = []

        async def capture(response):
            await response.aread()
            responses.append(response.json())
            requests.append(json.loads(response.request.content))

        sample = BFCLSample(
            "offline-two-turn-ls",
            [[{"role": "user", "content": text}] for text in turns],
            [tool],
            [{"ls": {"path": ["."]}}] * 2,
            {},
            [],
        )
        async with httpx.AsyncClient(event_hooks={"response": [capture]}) as client:
            rewards = await BFCLAgent(
                temperature=0, max_tokens=128, model_name=args.model
            ).run(
                sample,
                base_url=args.base_url,
                api_key="local",
                http_client=client,
            )
        report["fixtures"][label] = {
            "rewards": list(rewards.values()),
            "responses": responses,
            "requests": requests,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    parser.add_argument("--parser", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args()))
