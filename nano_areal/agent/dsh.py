"""
DeepSeek Harness (dsh) agent for nano-SWE tasks.

Two drivers share one workspace/reward contract:

  mode="sdk"      the real `deepseek-harness-sdk` runs the task as a subprocess
                  (profile `sdk-minimal`, chat-completions protocol patch) with
                  base_url/api_key pointed at the nano-areal proxy. Every model
                  call dsh makes is captured by the proxy as one interaction;
                  the outcome reward from the workspace is attached to the last
                  call and discounted backwards by /export_trajectories.

  mode="builtin"  a dependency-free tool loop with the same tool surface dsh's
                  minimal profile exposes (write_file / run_command / finish),
                  streaming Chat Completions (SSE) with `reasoning_content`.
                  Used for CPU tests and when the SDK is not installed.

Workspaces are fresh temp dirs per episode; verification runs in a separate
subprocess (see reward/code_exec.py).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from openai import AsyncOpenAI

from nano_areal.agent.base import AgentBase
from nano_areal.reward.code_exec import run_python
from nano_areal.reward.dsh_task import dsh_task_reward, workspace_files
from nano_areal.tasks.code import CodeSample

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a python file in the workspace, e.g. 'tests/test_solution.py'. Returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the task complete.",
            "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}},
        },
    },
]

_SYSTEM = (
    "You are a coding agent working in a small Python repository. Read the task in "
    "README.md, implement solution.py with write_file, verify with run_command on "
    "tests/test_solution.py, then call finish."
)


class DSHAgent(AgentBase):
    def __init__(
        self,
        mode: str = "auto",              # auto | sdk | builtin
        max_turns: int = 6,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        profile: str = "sdk-minimal",
        patches: tuple[str, ...] = (),
        dsh_home: str | None = None,
        model: str = "default",
        keep_workspaces: bool = False,
    ):
        self.mode = mode
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.profile = profile
        self.patches = patches
        self.dsh_home = dsh_home
        self.model = model
        self.keep_workspaces = keep_workspaces

    # ------------------------------------------------------------------

    def _resolve_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        try:
            import deepseek_harness  # noqa: F401

            return "sdk"
        except ImportError:
            return "builtin"

    async def run(self, data: CodeSample, **extra_kwargs) -> float:
        ws = Path(tempfile.mkdtemp(prefix="nano-areal-dsh-"))
        for rel, content in workspace_files(data).items():
            p = ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        try:
            if self._resolve_mode() == "sdk":
                await self._run_sdk(data, ws, extra_kwargs["base_url"], extra_kwargs["api_key"])
            else:
                await self._run_builtin(data, ws, extra_kwargs)
            reward, _ = await dsh_task_reward(data, ws)
            return reward
        finally:
            if not self.keep_workspaces:
                shutil.rmtree(ws, ignore_errors=True)

    # ------------------------------------------------------------------
    # Real SDK (subprocess, blocking → thread)
    # ------------------------------------------------------------------

    async def _run_sdk(self, data: CodeSample, ws: Path, base_url: str, api_key: str) -> None:
        from deepseek_harness import DeepSeekHarness

        home = self.dsh_home or tempfile.mkdtemp(prefix="nano-areal-dsh-home-")
        kwargs: dict = dict(
            dsh_home=str(Path(home).resolve()),
            cwd=str(ws.resolve()),
            profile=self.profile,
            provider="deepseek-official",
            model=self.model,
            base_url=base_url,
            api_key=api_key,
        )
        if self.patches:
            kwargs["patches"] = tuple(str(Path(p).resolve()) for p in self.patches)
        task = (ws / "README.md").read_text()

        def _blocking():
            with DeepSeekHarness(**kwargs) as harness:
                harness.run(task, session_id=f"nano-areal-{uuid.uuid4().hex[:8]}")

        await asyncio.to_thread(_blocking)

    # ------------------------------------------------------------------
    # Builtin tool loop (streaming)
    # ------------------------------------------------------------------

    async def _run_builtin(self, data: CodeSample, ws: Path, extra_kwargs: dict) -> None:
        client_kwargs: dict = dict(
            base_url=extra_kwargs["base_url"], api_key=extra_kwargs["api_key"], max_retries=0
        )
        if extra_kwargs.get("http_client") is not None:
            client_kwargs["http_client"] = extra_kwargs["http_client"]
        client = AsyncOpenAI(**client_kwargs)

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": (ws / "README.md").read_text()},
        ]
        for _ in range(self.max_turns):
            msg = await self._stream_turn(client, messages)
            messages.append(msg)
            if not msg.get("tool_calls"):
                break
            done = False
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self._tool(name, args, ws)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                done = done or name == "finish"
            if done:
                break

    async def _stream_turn(self, client: AsyncOpenAI, messages: list[dict]) -> dict:
        """Consume an SSE stream into one assistant message (content, reasoning, tool_calls)."""
        stream = await client.chat.completions.create(
            model="default",
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
        )
        content, reasoning = "", ""
        calls: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content += delta.content or ""
            extra = getattr(delta, "reasoning_content", None) or (
                delta.model_extra.get("reasoning_content") if getattr(delta, "model_extra", None) else None
            )
            reasoning += extra or ""
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(
                    tc.index, {"id": tc.id or "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    slot["function"]["name"] += tc.function.name or ""
                    slot["function"]["arguments"] += tc.function.arguments or ""
        msg: dict = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        return msg

    async def _tool(self, name: str, args: dict, ws: Path) -> str:
        if name == "write_file":
            rel = str(args.get("path", "")).lstrip("/")
            if not rel or ".." in Path(rel).parts:
                return "error: invalid path"
            p = ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")))
            return f"wrote {rel} ({len(str(args.get('content', '')))} chars)"
        if name == "run_command":
            rel = str(args.get("path", "")).lstrip("/")
            if not rel or ".." in Path(rel).parts or not (ws / rel).exists():
                return "error: file not found"
            argv = ["-m", "pytest", "-q", "-p", "no:cacheprovider", rel] if rel.startswith("tests") else [rel]
            res, _ = await run_python({}, argv, timeout=30.0, workspace=ws)
            out = (res.stdout + res.stderr)[-2000:]
            return f"exit={res.returncode}{' (timeout)' if res.timed_out else ''}\n{out}"
        if name == "finish":
            return "ok"
        return f"error: unknown tool {name}"
