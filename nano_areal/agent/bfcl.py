"""
BFCL multi-turn function-calling agent (proxy mode).

The agent is completely decoupled from AReaL internals:
  - Uses only the standard `openai` Python SDK
  - Receives base_url + api_key via extra_kwargs from OpenAIProxyWorkflow
  - Returns {response_id: reward} so the proxy can assign rewards per turn

The proxy server handles tokenization and logprob capture transparently.
"""
from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from nano_areal.agent.base import AgentBase
from nano_areal.dataset import BFCLSample, build_system_prompt
from nano_areal.reward.bfcl import BFCLReward, parse_tool_calls


# ---------------------------------------------------------------------------
# Simulated BFCL tool execution environment
# ---------------------------------------------------------------------------

class BFCLEnvironment:
    """
    Lightweight simulation of BFCL stateful backends.
    Returns plausible tool results so the conversation can continue.
    Reward comes from the AST verifier, not from execution correctness.
    """

    def __init__(self, initial_config: dict):
        self.state = dict(initial_config)

    def execute(self, tool_calls: list) -> str:
        results = []
        for tc in tool_calls:
            fn_name = tc.function.name if hasattr(tc, "function") else tc.get("name", "")
            try:
                args = (
                    json.loads(tc.function.arguments)
                    if hasattr(tc, "function")
                    else tc.get("arguments", {})
                )
            except (json.JSONDecodeError, AttributeError):
                args = {}
            results.append({"name": fn_name, "result": self._dispatch(fn_name, args)})
        return json.dumps(results)

    def _dispatch(self, fn_name: str, args: dict) -> Any:
        handler = getattr(self, f"_handle_{fn_name}", None)
        if handler:
            return handler(args)
        return {"status": "success", "output": f"Executed {fn_name}({args})"}

    def _handle_mv(self, args: dict) -> dict:
        return {"status": "success", "moved": args.get("src"), "to": args.get("dst")}

    def _handle_grep(self, args: dict) -> dict:
        return {"status": "success", "matches": [], "pattern": args.get("pattern")}

    def _handle_find(self, args: dict) -> dict:
        return {"status": "success", "results": []}


# ---------------------------------------------------------------------------
# BFCL Agent
# ---------------------------------------------------------------------------

class BFCLAgent(AgentBase):
    """
    Multi-turn BFCL function-calling agent.

    Mirrors the pattern from AReaL's gsm8k_rl_mt.py but adapted for
    function-calling tasks and the proxy mode interface.

    Return value: dict mapping each response_id to its turn reward.
    The proxy applies turn_discount via /export_trajectories.
    """

    def __init__(
        self,
        max_turns: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ):
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reward_fn = BFCLReward()

    async def run(
        self,
        data: BFCLSample,
        **extra_kwargs,
    ) -> dict[str, float]:
        base_url: str = extra_kwargs["base_url"]
        api_key: str = extra_kwargs["api_key"]
        http_client = extra_kwargs.get("http_client")

        # Standard OpenAI SDK — no AReaL dependency
        client_kwargs: dict = dict(base_url=base_url, api_key=api_key, max_retries=0)
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        client = AsyncOpenAI(**client_kwargs)

        env = BFCLEnvironment(data.initial_config)
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(data.involved_classes)}
        ]

        rewards: dict[str, float] = {}
        n_turns = min(len(data.question), self.max_turns)

        for turn_idx in range(n_turns):
            # Append user messages for this turn
            for msg in data.question[turn_idx]:
                messages.append(msg)

            response = await client.chat.completions.create(
                model="default",            # proxy rewrites to actual model name
                messages=messages,
                tools=data.tools or None,
                tool_choice="auto" if data.tools else None,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            assistant_msg = response.choices[0].message
            tool_calls_raw = assistant_msg.tool_calls or []

            # AST reward for this turn
            gt = data.ground_truth[turn_idx] if turn_idx < len(data.ground_truth) else {}
            turn_reward = self.reward_fn(parse_tool_calls(assistant_msg), gt)
            rewards[response.id] = turn_reward

            # Build assistant message dict for history
            assistant_dict: dict = {
                "role": "assistant",
                "content": assistant_msg.content or "",
            }
            if tool_calls_raw:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls_raw
                ]

            messages.append(assistant_dict)

            # Execute tool calls and append tool result
            if tool_calls_raw:
                tool_result = env.execute(tool_calls_raw)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_calls_raw[0].id,
                    "content": tool_result,
                })

        return rewards
        # The proxy applies turn_discount when export_trajectories is called
        # with discount=turn_discount — no need to apply it here.
