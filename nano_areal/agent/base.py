from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentBase(ABC):
    """
    Base class for nano-areal agents (proxy mode).

    Agents are plain async classes with a single `run()` method.
    They receive `base_url` + `api_key` via extra_kwargs and construct
    a standard AsyncOpenAI client — no AReaL internals are imported.

    Return value of `run()`:
        float              → reward for the last LLM call in the episode
        dict[str, float]   → {response_id: reward} — one entry per LLM call

    The proxy server uses the return value to assign rewards before
    exporting trajectories.
    """

    @abstractmethod
    async def run(self, data: Any, **extra_kwargs) -> float | dict[str, float]:
        """
        Execute one episode.

        Args:
            data:          One dataset sample (e.g. BFCLSample).
            **extra_kwargs:
                base_url   (str)                 — proxy /v1 endpoint
                api_key    (str)                 — per-session auth token
                http_client (httpx.AsyncClient)  — optional shared client

        Returns:
            float or dict mapping response_id → reward.
        """
        ...
