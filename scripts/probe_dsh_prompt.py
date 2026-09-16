"""
S1 probe: how many prompt tokens does a DeepSeek Harness (dsh) agent send?

Starts a mock OpenAI/Anthropic-compatible server that records every model
request, points the dsh Python SDK at it, runs a small task, and reports
per-call prompt token counts under the given tokenizer. The mock replies
with a short scripted tool call followed by a final answer, so the probe
also measures how fast the context grows per turn.

    pip install deepseek-harness-sdk
    uv run python scripts/probe_dsh_prompt.py \
        --tokenizer Qwen/Qwen3.5-2B --profile sdk-minimal \
        --patch scripts/dsh_probe_patch.yml --dsh-home /tmp/dsh-probe

Without the patch dsh uses its default Anthropic-style Messages protocol;
the mock server accepts both, so `--patch` may be omitted to compare.

The numbers feed docs/v0.2-plan.md §7 (max_model_len for dsh trajectories).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Request normalisation (Chat Completions and Messages → OpenAI-style)
# ---------------------------------------------------------------------------


def _messages_to_openai(body: dict) -> tuple[list[dict], list[dict]]:
    """Anthropic Messages payload → (messages, tools) in OpenAI shape."""
    msgs: list[dict] = []
    system = body.get("system")
    if system:
        text = (
            system
            if isinstance(system, str)
            else "".join(b.get("text", "") for b in system if isinstance(b, dict))
        )
        msgs.append({"role": "system", "content": text})
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": m["role"], "content": content})
            continue
        text_parts, tool_calls = [], []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                text_parts.append(b.get("text", ""))
            elif t == "thinking":
                text_parts.append(b.get("thinking", ""))
            elif t == "tool_use":
                tool_calls.append(
                    {
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    }
                )
            elif t == "tool_result":
                rc = b.get("content")
                rtext = (
                    rc
                    if isinstance(rc, str)
                    else "".join(
                        x.get("text", "") for x in rc or [] if isinstance(x, dict)
                    )
                )
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": rtext,
                    }
                )
            elif t == "image":
                text_parts.append("<image>")
        if text_parts or tool_calls:
            entry: dict = {"role": m["role"], "content": "".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            msgs.append(entry)
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in body.get("tools", [])
    ]
    return msgs, tools


def _chat_to_openai(body: dict) -> tuple[list[dict], list[dict]]:
    msgs = []
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(
                p.get("text", "") if p.get("type") == "text" else "<image>" for p in c
            )
        e = {**m, "content": c or ""}
        msgs.append(e)
    return msgs, body.get("tools", [])


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class Counter:
    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def count(self, messages: list[dict], tools: list[dict]) -> dict:
        try:
            text = self.tok.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
            )
            n_total = len(self.tok(text, add_special_tokens=False).input_ids)
        except Exception:
            text = json.dumps(
                {"messages": messages, "tools": tools}, ensure_ascii=False
            )
            n_total = len(self.tok(text, add_special_tokens=False).input_ids)
        n_tools = (
            len(self.tok(json.dumps(tools), add_special_tokens=False).input_ids)
            if tools
            else 0
        )
        sys_text = "".join(
            m.get("content", "") for m in messages if m.get("role") == "system"
        )
        n_system = (
            len(self.tok(sys_text, add_special_tokens=False).input_ids)
            if sys_text
            else 0
        )
        return {
            "prompt_tokens": n_total,
            "tool_schema_tokens": n_tools,
            "system_tokens": n_system,
            "n_messages": len(messages),
            "n_tools": len(tools),
        }


# ---------------------------------------------------------------------------
# Mock model server
# ---------------------------------------------------------------------------


class MockModel:
    """
    Scripted policy: first call → one shell tool call (`echo probe`),
    subsequent calls → final text. Records every request.
    """

    def __init__(self, counter: Counter, max_tool_turns: int):
        self.counter = counter
        self.max_tool_turns = max_tool_turns
        self.records: list[dict] = []
        self.app = FastAPI()
        self.app.post("/chat/completions")(self.chat)
        self.app.post("/v1/chat/completions")(self.chat)
        self.app.post("/v1/messages")(self.messages)
        self.app.post("/anthropic/v1/messages")(self.messages)

    def _record(self, protocol: str, messages: list[dict], tools: list[dict]) -> dict:
        stats = self.counter.count(messages, tools)
        rec = {
            "call": len(self.records),
            "protocol": protocol,
            **stats,
            "tool_names": [t.get("function", t).get("name") for t in tools],
        }
        self.records.append(rec)
        print(
            f"  call {rec['call']:2d} [{protocol}] prompt={stats['prompt_tokens']:6d} "
            f"tools={stats['tool_schema_tokens']:5d} "
            f"system={stats['system_tokens']:5d} "
            f"msgs={stats['n_messages']}"
        )
        return rec

    def _pick_shell_tool(self, tools: list[dict]) -> str | None:
        names = [t.get("function", t).get("name", "") for t in tools]
        for n in names:
            if any(k in n.lower() for k in ("shell", "bash", "exec", "run", "command")):
                return n
        return names[0] if names else None

    def _decide(self, tools: list[dict]) -> tuple[str | None, dict]:
        n_tool_turns = sum(1 for r in self.records if r.get("emitted_tool_call"))
        tool = self._pick_shell_tool(tools)
        if tool and n_tool_turns < self.max_tool_turns:
            return tool, {"command": "echo nano-areal-probe"}
        return None, {}

    # --- Chat Completions ---------------------------------------------------
    async def chat(self, request: Request):
        body = await request.json()
        messages, tools = _chat_to_openai(body)
        rec = self._record("chat-completions", messages, tools)
        tool, args = self._decide(tools)
        rec["emitted_tool_call"] = tool is not None
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if tool:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args)},
                    }
                ]
            }
            finish = "tool_calls"
        else:
            delta = {"content": "Probe complete."}
            finish = "stop"
        usage = {
            "prompt_tokens": rec["prompt_tokens"],
            "completion_tokens": 8,
            "total_tokens": rec["prompt_tokens"] + 8,
        }
        if body.get("stream"):

            def gen():
                base = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "probe"),
                }
                yield (
                    "data: "
                    + json.dumps(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", **delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            **base,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": finish}
                            ],
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "probe"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", **delta},
                        "finish_reason": finish,
                    }
                ],
                "usage": usage,
            }
        )

    # --- Anthropic Messages -------------------------------------------------
    async def messages(self, request: Request):
        body = await request.json()
        messages, tools = _messages_to_openai(body)
        rec = self._record("messages", messages, tools)
        tool, args = self._decide(tools)
        rec["emitted_tool_call"] = tool is not None
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        if tool:
            block = {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:8]}",
                "name": tool,
                "input": args,
            }
            stop = "tool_use"
        else:
            block = {"type": "text", "text": "Probe complete."}
            stop = "end_turn"
        usage = {"input_tokens": rec["prompt_tokens"], "output_tokens": 8}

        def ev(name: str, data: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        def gen():
            yield ev(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": mid,
                        "type": "message",
                        "role": "assistant",
                        "model": body.get("model", "probe"),
                        "content": [],
                        "stop_reason": None,
                        "usage": usage,
                    },
                },
            )
            if block["type"] == "text":
                yield ev(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield ev(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
            else:
                yield ev(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": {},
                        },
                    },
                )
                yield ev(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block["input"]),
                        },
                    },
                )
            yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield ev(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": 8},
                },
            )
            yield ev("message_stop", {"type": "message_stop"})

        if body.get("stream", True):
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "probe"),
                "content": [block],
                "stop_reason": stop,
                "usage": usage,
            }
        )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_in_thread(app: FastAPI, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return server


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--profile", default="sdk-minimal")
    ap.add_argument(
        "--patch",
        action="append",
        default=[],
        help="Cordis patch yml (e.g. scripts/dsh_probe_patch.yml for chat-completions)",
    )
    ap.add_argument(
        "--dsh-home", default=None, help="isolated DSH_HOME (default: temp dir)"
    )
    ap.add_argument("--workspace", default=None, help="agent cwd (default: temp dir)")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument(
        "--task",
        default="List the files in the current directory and summarize what you see.",
    )
    ap.add_argument("--max-tool-turns", type=int, default=2)
    ap.add_argument("--out", default=None, help="write JSON records here")
    args = ap.parse_args()

    try:
        from deepseek_harness import DeepSeekHarness
    except ImportError:
        raise SystemExit(
            "deepseek-harness-sdk not installed: pip install deepseek-harness-sdk"
        )

    counter = Counter(args.tokenizer)
    mock = MockModel(counter, max_tool_turns=args.max_tool_turns)
    port = _free_port()
    server = serve_in_thread(mock.app, port)
    base_url = f"http://127.0.0.1:{port}"
    print(
        f"mock model server at {base_url}; "
        f"tokenizer={args.tokenizer}; profile={args.profile}"
    )

    dsh_home = args.dsh_home or tempfile.mkdtemp(prefix="dsh-probe-home-")
    workspace = args.workspace or tempfile.mkdtemp(prefix="dsh-probe-ws-")
    Path(workspace, "README.md").write_text("# probe workspace\n")

    kwargs = dict(
        dsh_home=str(Path(dsh_home).resolve()),
        cwd=str(Path(workspace).resolve()),
        profile=args.profile,
        provider="deepseek-official",
        model=args.model,
        base_url=base_url,
        api_key="probe",
    )
    if args.patch:
        kwargs["patches"] = tuple(str(Path(p).resolve()) for p in args.patch)

    error = None
    try:
        with DeepSeekHarness(**kwargs) as harness:
            result = harness.run(args.task, session_id=f"probe-{uuid.uuid4().hex[:8]}")
            print(
                f"dsh finished: "
                f"finish_reason={getattr(result, 'finish_reason', None)!r}"
            )
    except Exception as e:  # keep the captured requests even if dsh bails out
        error = f"{type(e).__name__}: {e}"
        print(f"dsh run failed: {error}")
    finally:
        server.should_exit = True

    if not mock.records:
        raise SystemExit("no model requests captured — check profile/protocol/base_url")

    first = mock.records[0]["prompt_tokens"]
    last = mock.records[-1]["prompt_tokens"]
    per_turn = (last - first) / max(len(mock.records) - 1, 1)
    print("\nsummary")
    print(f"  calls                 : {len(mock.records)}")
    print(f"  first prompt tokens   : {first}")
    print(f"  last prompt tokens    : {last}")
    print(f"  growth per tool turn  : {per_turn:.0f}")
    print(f"  tool schema tokens    : {mock.records[0]['tool_schema_tokens']}")
    print(f"  system prompt tokens  : {mock.records[0]['system_tokens']}")
    for budget in (4096, 8192, 16384, 32768):
        turns = (
            int((budget - first - 1024) / per_turn) if per_turn > 0 else float("inf")
        )
        print(
            f"  max_model_len={budget:6d} → ~{turns} tool turns "
            f"(with 1k response headroom)"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "tokenizer": args.tokenizer,
                    "profile": args.profile,
                    "patches": args.patch,
                    "error": error,
                    "records": mock.records,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
