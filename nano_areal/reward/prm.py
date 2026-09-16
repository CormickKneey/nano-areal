"""
Process-style shaping rewards (AReaL #1701 "PRM" hook) for agent trajectories.

These are cheap rule-based signals computed per LLM call from the captured
InteractionRecord / TrajectoryTensors, added to the outcome reward with a
small weight so that GRPO sees a denser gradient on long tool-using episodes:

  * tool-call validity   - every tool_call parses as JSON and names a known tool
  * length budget        - penalise responses that hit max_tokens or exceed a
                           soft token budget
  * format               - reasoning present when required, no empty replies
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ProcessRewardConfig:
    weight: float = 0.1
    known_tools: tuple[str, ...] = ()
    soft_token_budget: int = 2048        # response tokens
    penalty_truncated: float = -1.0
    penalty_bad_tool_call: float = -1.0
    bonus_valid_tool_call: float = 0.5
    penalty_empty: float = -0.5
    require_reasoning: bool = False
    # Extra parsers: name -> callable(str) -> float in [-1, 1]
    extra: dict = field(default_factory=dict)


def _tool_call_ok(tc: dict, known: tuple[str, ...]) -> bool:
    name = tc.get("name") or tc.get("function", {}).get("name", "")
    if known and name not in known:
        return False
    args = tc.get("arguments", tc.get("function", {}).get("arguments", "{}"))
    if isinstance(args, str):
        try:
            json.loads(args or "{}")
        except json.JSONDecodeError:
            return False
    return True


def process_reward(
    *,
    response_content: str,
    tool_calls: list[dict],
    n_response_tokens: int,
    stop_reason: str,
    reasoning_content: str = "",
    cfg: ProcessRewardConfig | None = None,
) -> tuple[float, dict]:
    """Return (weighted shaping reward, components)."""
    cfg = cfg or ProcessRewardConfig()
    parts: dict[str, float] = {}

    if stop_reason == "length":
        parts["truncated"] = cfg.penalty_truncated
    elif n_response_tokens > cfg.soft_token_budget:
        parts["over_budget"] = -min(1.0, (n_response_tokens - cfg.soft_token_budget) / cfg.soft_token_budget)

    if tool_calls:
        ok = all(_tool_call_ok(tc, cfg.known_tools) for tc in tool_calls)
        parts["tool_calls"] = cfg.bonus_valid_tool_call if ok else cfg.penalty_bad_tool_call
    elif not response_content.strip():
        parts["empty"] = cfg.penalty_empty

    if cfg.require_reasoning and not reasoning_content.strip():
        parts["no_reasoning"] = -0.5

    for name, fn in cfg.extra.items():
        parts[name] = float(fn(response_content))

    total = cfg.weight * sum(parts.values())
    return total, parts
