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
import concurrent.futures
import json
import tempfile
import threading
import uuid
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
        "env": {
            "DSH_CONTEXT_WINDOW": str(context_window_tokens),
            "HOME": home,
            "XDG_CACHE_HOME": str(Path(home) / "cache"),
        },
    }


def episode_reward(finish_reason: str | None, verified: float | None) -> float:
    """Only a completed, independently verified episode earns a reward."""
    if finish_reason != "completed":
        return 0.0
    if verified is None:
        return 0.0
    return float(verified)


def run_with_deadline(harness, prompt: str, timeout_seconds: float, cancelled=None):
    """Close the owned runtime at the activity deadline, then join the watchdog."""
    expired = threading.Event()
    finished = threading.Event()

    def watch_cancellation():
        while not finished.wait(0.05):
            if cancelled.is_set():
                harness.close()
                return

    def expire():
        expired.set()
        harness.close()

    timer = threading.Timer(timeout_seconds, expire)
    timer.daemon = True
    timer.start()
    watcher = None
    if cancelled is not None:
        watcher = threading.Thread(target=watch_cancellation, daemon=True)
        watcher.start()
    try:
        try:
            result = harness.run(prompt)
        except Exception as error:
            if cancelled is not None and cancelled.is_set():
                raise concurrent.futures.CancelledError(
                    "dsh episode cancelled"
                ) from error
            if expired.is_set():
                raise TimeoutError("dsh episode activity deadline exceeded") from error
            raise
        if expired.is_set():
            raise TimeoutError("dsh episode activity deadline exceeded")
        if cancelled is not None and cancelled.is_set():
            raise concurrent.futures.CancelledError("dsh episode cancelled")
        return result
    finally:
        timer.cancel()
        timer.join()
        finished.set()
        if watcher:
            watcher.join()


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
        patches: tuple[str, ...] = (),
        timeout_seconds: float = 600.0,
        session_log_dir: str | None = None,
    ):
        self.model_name = model_name
        self.profile = profile
        self.provider = provider
        self.context_window_tokens = context_window_tokens
        self.max_output_tokens = max_output_tokens
        self.patches = patches
        if timeout_seconds <= 0:
            raise ValueError("dsh timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self.session_log_dir = session_log_dir
        self.last_result: dict = {}

    async def run(self, data: Any, **extra_kwargs) -> float:
        base_url = extra_kwargs["base_url"]
        api_key = extra_kwargs["api_key"]
        cancelled = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(
            self._run_sync, data, base_url, api_key, cancelled
        ))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if not worker.done():
                    raise
            except Exception:
                pass
            self.last_result.update(
                status="cancelled", error_type="CancelledError",
                error="dsh episode cancelled", verified_reward=0.0,
            )
            capture = self.last_result.get("session_log_dir")
            if capture:
                (Path(capture) / "result.json").write_text(
                    json.dumps(self.last_result, indent=2) + "\n"
                )
            raise

    def _run_sync(
        self, data: Any, base_url: str, api_key: str, cancelled=None,
    ) -> float:
        self.last_result = {}
        try:
            from deepseek_harness import DeepSeekHarness
        except ImportError as exc:
            raise ImportError(
                "deepseek-harness-sdk is not installed. "
                "Install deepseek_harness and matching deepseek-harness-runtime-bin "
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
        capture = None
        home = home_cm.name
        if self.session_log_dir:
            capture = Path(self.session_log_dir).resolve() / uuid.uuid4().hex
            capture.mkdir(parents=True)
            home = str(capture / "dsh-home")
        self.last_result = {"session_log_dir": str(capture) if capture else None}
        try:
            kwargs = build_dsh_kwargs(
                base_url=base_url,
                api_key=api_key,
                cwd=str(workspace),
                dsh_home=home,
                model=self.model_name,
                profile=self.profile,
                provider=self.provider,
                max_tokens=self.max_output_tokens,
                context_window_tokens=self.context_window_tokens,
            )
            kwargs.update(
                patches=self.patches,
                request_timeout_seconds=self.timeout_seconds,
            )
            with DeepSeekHarness(**kwargs) as harness:
                result = run_with_deadline(
                    harness, prompt, self.timeout_seconds, cancelled
                )
            if capture:
                (capture / "events.json").write_text(
                    json.dumps(result.events, indent=2) + "\n"
                )
            hidden_dir = data.get("hidden_test_dir") if isinstance(data, dict) else None
            if hidden_dir:
                from nano_areal.reward.dsh_task import external_test_report

                verification = external_test_report(
                    workspace,
                    Path(hidden_dir),
                    result.final_response,
                    runner=data.get("hidden_test_runner", "script"),
                )
                verified = verification["pass_rate"]
                self.last_result["verification"] = verification
            else:
                preset = data.get("reward") if isinstance(data, dict) else None
                verified = (
                    float(preset)
                    if preset is not None
                    else hidden_test_pass_rate(Path(kwargs["cwd"]))
                )
            self.last_result.update(
                {
                    "session_id": result.session_id,
                    "finish_reason": result.finish_reason,
                    "verified_reward": verified,
                    "response": result.final_response,
                    "status": result.finish_reason or "missing-finish-reason",
                    "terminal_reasons": [
                        event["data"].get("reason") for event in result.events
                        if event.get("type") == "turn/end"
                    ],
                }
            )
            return episode_reward(result.finish_reason, verified)
        except Exception as error:
            self.last_result.update(
                {
                    "status": "cancelled"
                    if isinstance(error, concurrent.futures.CancelledError)
                    else "timeout"
                    if "timeout" in type(error).__name__.lower()
                    else "error",
                    "error_type": type(error).__name__,
                    "error": str(error).replace(api_key, "[REDACTED]"),
                }
            )
            raise
        finally:
            if capture:
                (capture / "result.json").write_text(
                    json.dumps(self.last_result, indent=2) + "\n"
                )
            home_cm.cleanup()
            if workspace_cm is not None:
                workspace_cm.cleanup()
