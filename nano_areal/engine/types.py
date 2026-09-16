"""
Token-level interaction data structures used by the proxy server and trainer.
Mirrors AReaL's InteractionWithTokenLogpReward, extended with multimodal
inputs (pixel patches + grid) so image trajectories can be re-scored by the
trainer exactly as the inference engine saw them.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Tensor <-> JSON helpers (export_trajectories transport)
# ---------------------------------------------------------------------------

def tensor_to_b64(t: torch.Tensor) -> dict:
    t = t.detach().cpu().contiguous()
    if t.is_floating_point():
        t = t.to(torch.float16)
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "data": base64.b64encode(t.numpy().tobytes()).decode("ascii"),
    }


def b64_to_tensor(d: dict) -> torch.Tensor:
    import numpy as np
    dtype = d["dtype"]
    np_dtype = {"float16": np.float16, "float32": np.float32, "int64": np.int64, "int32": np.int32}[dtype]
    arr = np.frombuffer(base64.b64decode(d["data"]), dtype=np_dtype).reshape(d["shape"])
    t = torch.from_numpy(arr.copy())
    return t.float() if t.is_floating_point() else t.long()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

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

    # Model "thinking" text returned separately by the backend (dsh / Qwen thinking mode)
    reasoning_content: str = ""

    # Tool calls in the response (parsed)
    tool_calls: list[dict] = field(default_factory=list)

    # Parent interaction_id for multi-turn reward discount
    parent_id: str | None = None

    # vllm finish_reason: "stop" | "length" | "tool_calls" | "abort" | ...
    stop_reason: str = "stop"

    # Depth in the conversation tree (0 = first LLM call of the episode)
    turn_id: int = 0

    # Multimodal prompt inputs, exactly as fed to the inference engine
    pixel_values: torch.Tensor | None = None     # [N_patches, C*tp*ps*ps]
    image_grid_thw: torch.Tensor | None = None   # [N_img, 3]
    # Lightweight image metadata (source url / data-uri hash / size) for replay
    images: list[dict] = field(default_factory=list)

    @property
    def is_truncated(self) -> bool:
        return self.stop_reason == "length"

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
            stop_reason=self.stop_reason,
            turn_id=self.turn_id,
            pixel_values=self.pixel_values,
            image_grid_thw=self.image_grid_thw,
            images=list(self.images),
            tool_calls=list(self.tool_calls),
            reasoning_content=self.reasoning_content,
            response_content=self.response_content,
        )


@dataclass
class TrajectoryTensors:
    """Ready-for-training tensors from one interaction."""
    input_ids: torch.Tensor      # [T]
    response_mask: torch.Tensor  # [T]  float, 0/1
    logprobs: torch.Tensor       # [T]  behavior log probs (0 at prompt positions)
    reward: float
    stop_reason: str = "stop"
    turn_id: int = 0
    # Set by the rollout worker: all episodes of one prompt share a group_id
    group_id: str = ""
    # Multimodal inputs (None for text-only)
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    images: list[dict] = field(default_factory=list)
    # Replay metadata
    tool_calls: list[dict] = field(default_factory=list)
    reasoning_content: str = ""
    response_content: str = ""

    @property
    def is_truncated(self) -> bool:
        return self.stop_reason == "length"

    @property
    def has_images(self) -> bool:
        return self.pixel_values is not None and self.pixel_values.numel() > 0

    def to_json(self) -> dict:
        d = {
            "input_ids": self.input_ids.tolist(),
            "response_mask": self.response_mask.tolist(),
            "logprobs": self.logprobs.tolist(),
            "reward": self.reward,
            "stop_reason": self.stop_reason,
            "turn_id": self.turn_id,
            "group_id": self.group_id,
            "images": self.images,
            "tool_calls": self.tool_calls,
            "reasoning_content": self.reasoning_content,
            "response_content": self.response_content,
        }
        if self.has_images:
            d["pixel_values"] = tensor_to_b64(self.pixel_values)
            d["image_grid_thw"] = tensor_to_b64(self.image_grid_thw)
        return d

    @classmethod
    def from_json(cls, t: dict) -> "TrajectoryTensors":
        return cls(
            input_ids=torch.tensor(t["input_ids"], dtype=torch.long),
            response_mask=torch.tensor(t["response_mask"], dtype=torch.float),
            logprobs=torch.tensor(t["logprobs"], dtype=torch.float),
            reward=t["reward"],
            stop_reason=t.get("stop_reason", "stop"),
            turn_id=t.get("turn_id", 0),
            group_id=t.get("group_id", ""),
            pixel_values=b64_to_tensor(t["pixel_values"]) if t.get("pixel_values") else None,
            image_grid_thw=b64_to_tensor(t["image_grid_thw"]) if t.get("image_grid_thw") else None,
            images=t.get("images", []),
            tool_calls=t.get("tool_calls", []),
            reasoning_content=t.get("reasoning_content", ""),
            response_content=t.get("response_content", ""),
        )
