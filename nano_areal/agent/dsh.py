"""Black-box deepseek-harness (dsh) agent.

The Python SDK (``deepseek_harness.DeepSeekHarness``) launches bundled ``dsh
--profile sdk-minimal`` over stdio JSON-RPC. nano-areal does not import the
harness package until a turn runs, and it does not modify dsh.

Verified against deepseek-ai/deepseek-harness master (SDK README +
``llm-deepseek``):

- ``dsh_home`` is required. The SDK never reads ``~/.dsh``.
- ``cwd`` is the task workspace, not this repository checkout.
- ``base_url`` / ``api_key`` become ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_API_KEY``.
- The adapter speaks Anthropic Messages. ``protocol`` is not configurable.
- Auth header on the model request is ``x-api-key``, not ``Authorization``.
- ``RunResult.finish_reason`` is ``completed``, ``max-tokens``, ``error``, or None.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from nano_areal.agent.base import AgentBase
from nano_areal.engine.messages import messages_base_url
from nano_areal.reward.dsh_task import hidden_test_pass_rate


def build_dsh_kwargs(
    *,
    base_url: str,
    api_key: str,
    cwd: str,
    dsh_home: str,
    model: str = "default",
    profile: str = "sdk-minimal",
    provider: str = "deepseek-official",
    max_tokens: int = 1024,
    context_window_tokens: int = 8192,
) -> dict:
    """Keyword arguments for ``DeepSeekHarness``. The key stays out of argv."""
    if not str(dsh_home).strip():
        raise ValueError("dsh requires an explicit dsh_home")
    workspace = str(Path(cwd).expanduser().resolve())
    home = str(Path(dsh_home).expanduser().resolve())
    return {
        "dsh_home": home,
        "cwd": workspace,
        "profile": profile,
        "provider": provider,
        "model": model,
        "base_url": messages_base_url(base_url),
        "api_key": api_key,
        "max_tokens": max_tokens,
        "env": {"DSH_CONTEXT_WINDOW": str(context_window_tokens)},
    }


def episode_reward(finish_reason: str | None, verified: float | None) -> float:
    """``max-tokens`` / ``error`` / missing reason score 0. A clean exit is not a reward."""
    if finish_reason != "completed":
        return 0.0
    if verified is None:
        return 0.0
    return float(verified)


def task_prompt(data: Any) -> str:
    if isinstance(data, dict):
        return str(data.get("prompt") or data.get("instruction") or "")
    question = getattr(data, "question", None)
    if question:
        parts: list[str] = []
        for turn in question:
            for msg in turn:
                content = msg.get("content") if isinstance(msg, dict) else ""
                if isinstance(content, str) and content:
                    parts.append(content)
        return "\n".join(parts)
    return ""


def task_workspace(data: Any) -> Path | None:
    raw = data.get("workspace") if isinstance(data, dict) else None
    if not raw:
        return None
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"dsh task workspace does not exist: {path}")
    return path


class DshAgent(AgentBase):
    """One dsh ``sdk-minimal`` turn against the proxy, then a local verifier."""

    def __init__(
        self,
        model_name: str = "default",
        profile: str = "sdk-minimal",
        provider: str = "deepseek-official",
        context_window_tokens: int = 8192,
        max_output_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.profile = profile
        self.provider = provider
        self.context_window_tokens = context_window_tokens
        self.max_output_tokens = max_output_tokens

    async def run(self, data: Any, **extra_kwargs) -> float:
        base_url = extra_kwargs["base_url"]
        api_key = extra_kwargs["api_key"]
        return await asyncio.to_thread(self._run_sync, data, base_url, api_key)

    def _run_sync(self, data: Any, base_url: str, api_key: str) -> float:
        try:
            from deepseek_harness import DeepSeekHarness
        except ImportError as exc:
            raise ImportError(
                "deepseek-harness-sdk is not installed. "
                "Install deepseek_harness and the matching deepseek-harness-runtime-bin "
                "from the same deepseek-ai/deepseek-harness revision."
            ) from exc

        prompt = task_prompt(data)
        if not prompt.strip():
            raise ValueError("dsh run requires a non-empty prompt")
        given = task_workspace(data)
        workspace_cm = None
        workspace = given
        if workspace is None:
            workspace_cm = tempfile.TemporaryDirectory(prefix="nano-areal-task-")
            workspace = Path(workspace_cm.name)
        home_cm = tempfile.TemporaryDirectory(prefix="nano-areal-dsh-home-")
        try:
            kwargs = build_dsh_kwargs(
                base_url=base_url,
                api_key=api_key,
                cwd=str(workspace),
                dsh_home=home_cm.name,
                model=self.model_name,
                profile=self.profile,
                provider=self.provider,
                max_tokens=self.max_output_tokens,
                context_window_tokens=self.context_window_tokens,
            )
            with DeepSeekHarness(**kwargs) as harness:
                result = harness.run(prompt)
            preset = data.get("reward") if isinstance(data, dict) else None
            verified = float(preset) if preset is not None else hidden_test_pass_rate(Path(kwargs["cwd"]))
            return episode_reward(result.finish_reason, verified)
        finally:
            home_cm.cleanup()
            if workspace_cm is not None:
                workspace_cm.cleanup()
