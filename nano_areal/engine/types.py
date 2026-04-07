"""
Token-level interaction data structures used by the proxy server and trainer.
Mirrors AReaL's InteractionWithTokenLogpReward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InteractionRecord:
    """
    One LLM call captured by the proxy.
    Stores token-level data needed for GRPO gradient computation.
    """
    response_id: str

    # Full tokenized sequence: [prompt tokens] + [response tokens]
    input_ids: list[int] = field(default_factory=list)

    # 1 for response tokens, 0 for prompt tokens
    response_mask: list[int] = field(default_factory=list)

    # Per-token log probabilities for response tokens (length == sum(response_mask))
    token_logprobs: list[float] = field(default_factory=list)

    # Reward assigned via /rl/set_reward (None until set)
    reward: float | None = None

    # Original messages sent to the model (for visualization)
    messages: list[dict] = field(default_factory=list)

    # Raw response content (for reward computation and replay)
    response_content: str = ""

    # Tool calls in the response (parsed)
    tool_calls: list[dict] = field(default_factory=list)

    # Parent interaction_id for multi-turn reward discount
    parent_id: str | None = None

    def to_trajectory(self) -> "TrajectoryTensors":
        """Convert to tensor format for the trainer."""
        T = len(self.input_ids)
        input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        response_mask = torch.tensor(self.response_mask, dtype=torch.float)

        # Align logprobs: proxy stores logprob[i] = log P(token[i+1] | token[:i+1])
        # trainer expects [T] tensor aligned with input_ids shift
        logprobs_tensor = torch.zeros(T, dtype=torch.float)
        resp_positions = [i for i, m in enumerate(self.response_mask) if m == 1]
        for pos, lp in zip(resp_positions, self.token_logprobs):
            if pos < T:
                logprobs_tensor[pos] = lp

        return TrajectoryTensors(
            input_ids=input_ids,
            response_mask=response_mask,
            logprobs=logprobs_tensor,
            reward=self.reward or 0.0,
        )


@dataclass
class TrajectoryTensors:
    """Ready-for-training tensors from one interaction."""
    input_ids: torch.Tensor      # [T]
    response_mask: torch.Tensor  # [T]  float, 0/1
    logprobs: torch.Tensor       # [T]  log probs (0 at prompt positions)
    reward: float
