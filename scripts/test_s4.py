"""Offline regressions for dsh capture, accounting, and trusted verification."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.responses import JSONResponse

from nano_areal.agent.dsh import DshAgent, run_with_deadline
from nano_areal.agent.dsh_audit import reconcile_session
from nano_areal.engine.messages import anthropic_request_to_openai
from nano_areal.engine.proxy import (
    ADMIN_KEY,
    OpenAIProxyClient,
    OpenAIProxyServer,
    ProxySession,
)
from nano_areal.reward.dsh_task import external_test_report
from scripts.build_nano_swe import build


def audit_fixture():
    call = {
        "id": "call-1",
        "type": "tool-call",
        "name": "bash",
        "arguments": '{"command":"pwd"}',
    }
    events = [
        {
            "type": "assistant/message",
            "data": {
                "usage": {"inputTokens": 10, "outputTokens": 3, "totalTokens": 13},
                "message": {"content": [call]},
            },
        },
        {
            "type": "tool/call",
            "data": {
                "callId": "call-1",
                "name": "bash",
                "arguments": call["arguments"],
            },
        },
        {
            "type": "tool/result",
            "data": {
                "message": {
                    "toolCallId": "call-1",
                    "content": [{"type": "text", "text": "/task"}],
                    "isError": False,
                }
            },
        },
        {
            "type": "assistant/message",
            "data": {
                "usage": {"inputTokens": 20, "outputTokens": 2, "totalTokens": 22},
                "message": {"content": [{"type": "text", "text": "Done"}]},
            },
        },
    ]
    interactions = [
        {
            "prompt_tokens": 10,
            "output_tokens": 3,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "bash", "arguments": call["arguments"]},
                }
            ],
            "messages": [],
        },
        {
            "prompt_tokens": 20,
            "output_tokens": 2,
            "tool_calls": [],
            "messages": [
                {"role": "tool", "tool_call_id": "call-1", "content": "/task"}
            ],
        },
    ]
    return events, interactions


def test_reconciliation_covers_counts_calls_execution_and_replay():
    events, interactions = audit_fixture()
    assert reconcile_session(events, interactions)["matched"]
    for field in ("inputTokens", "outputTokens", "totalTokens"):
        broken = copy.deepcopy(events)
        broken[0]["data"]["usage"][field] += 1
        assert not reconcile_session(broken, interactions)["matched"]
    broken = copy.deepcopy(interactions)
    broken[0]["tool_calls"][0]["function"]["arguments"] = '{"command":"ls"}'
    assert not reconcile_session(events, broken)["matched"]
    broken = copy.deepcopy(interactions)
    broken[1]["messages"][0]["content"] = "changed"
    assert not reconcile_session(events, broken)["matched"]
    assert not reconcile_session(events[:-1], interactions)["matched"]
    assert not reconcile_session(events[:2] + events[3:], interactions)["matched"]
    assert not reconcile_session([], [])["matched"]


def test_deadline_closes_runtime_even_when_sdk_waits_without_rpc_timeout():
    closed = threading.Event()

    class WaitingHarness:
        def close(self):
            closed.set()

        def run(self, prompt):
            assert closed.wait(1)
            raise RuntimeError("Runtime closed")

    with pytest.raises(TimeoutError, match="activity deadline"):
        run_with_deadline(WaitingHarness(), "Test", 0.02)
    assert closed.is_set()


def test_session_end_joins_only_owned_requests_and_blocks_early_export():
    async def scenario():
        server = OpenAIProxyServer("http://localhost/v1", None)
        first = ProxySession("first", "first-key")
        second = ProxySession("second", "second-key")
        for session in (first, second):
            server._sessions[session.session_id] = session
            server._key_to_session[session.api_key] = session.session_id
        started = {name: asyncio.Event() for name in ("first", "second")}
        release = asyncio.Event()

        async def wait_for_backend(body, session, return_payload=False):
            started[session.session_id].set()
            await release.wait()
            return JSONResponse({"ok": True})

        server._handle_chat_completion = wait_for_backend
        async with httpx.AsyncClient(
            base_url="http://localhost", transport=httpx.ASGITransport(app=server.app)
        ) as client:
            first_request = asyncio.create_task(client.post(
                "/v1/chat/completions", json={"messages": []},
                headers={"Authorization": "Bearer first-key"},
            ))
            second_request = asyncio.create_task(client.post(
                "/v1/chat/completions", json={"messages": []},
                headers={"Authorization": "Bearer second-key"},
            ))
            await asyncio.gather(*(event.wait() for event in started.values()))
            early = await client.post(
                "/export_trajectories", json={"session_id": "first"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert early.status_code == 409
            ended = await client.post(
                "/rl/end_session", headers={"Authorization": "Bearer first-key"}
            )
            assert ended.json()["requests"] == [
                {"session_id": "first", "status": "cancelled"}
            ]
            with pytest.raises(asyncio.CancelledError):
                await first_request
            assert not first.pending and second.pending
            assert not second_request.done()
            rejected = await client.post(
                "/v1/chat/completions", json={"messages": []},
                headers={"Authorization": "Bearer first-key"},
            )
            assert rejected.status_code == 410
            release.set()
            assert (await second_request).status_code == 200
            assert second.requests[0]["status"] == "response-ready"
            assert not second.pending
            await client.post(
                "/rl/end_session", headers={"Authorization": "Bearer second-key"}
            )
            assert second.requests[0]["status"] == "response-ready"

    asyncio.run(scenario())


def test_async_dsh_cancellation_joins_runtime_and_persists_result(tmp_path):
    started, closed = threading.Event(), threading.Event()

    class WaitingHarness:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            closed.set()

        def run(self, prompt):
            started.set()
            assert closed.wait(2)
            raise RuntimeError("Runtime closed")

    agent = DshAgent(session_log_dir=str(tmp_path / "logs"))

    async def scenario():
        worker = asyncio.create_task(agent.run(
            {"prompt": "Test", "workspace": str(tmp_path)},
            base_url="http://localhost/v1", api_key="local-test-key",
        ))
        assert await asyncio.to_thread(started.wait, 2)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert closed.is_set()
        capture = Path(agent.last_result["session_log_dir"])
        saved = json.loads((capture / "result.json").read_text())
        assert saved["status"] == "cancelled"
        assert saved["error_type"] == "CancelledError"
        assert saved["verified_reward"] == 0.0

    with patch.dict(sys.modules, {
        "deepseek_harness": SimpleNamespace(DeepSeekHarness=WaitingHarness)
    }):
        asyncio.run(scenario())


def test_failed_episode_discards_training_export_but_retains_request_ledger():
    async def scenario():
        server = OpenAIProxyServer("http://localhost/v1", None)
        session = ProxySession("failed", "failed-key")
        session.interactions["backend-only"] = object()
        session.requests.append({
            "status": "response-ready", "response_id": "backend-only"
        })
        server._sessions[session.session_id] = session
        server._key_to_session[session.api_key] = session.session_id
        client = OpenAIProxyClient("http://localhost")
        client.session_id, client.api_key = session.session_id, session.api_key
        client._http = httpx.AsyncClient(
            base_url="http://localhost", transport=httpx.ASGITransport(app=server.app)
        )
        await client.__aexit__(TimeoutError, TimeoutError("deadline"), None)
        assert client.trajectories == []
        assert client.request_audit == session.requests
        assert "backend-only" in session.interactions
        assert session.session_id not in server._sessions
        assert client._http.is_closed

    asyncio.run(scenario())


def test_embedded_proxy_leaves_signal_ownership_with_caller():
    class EmbeddedServer:
        def __init__(self, config):
            self.should_exit = False
            self.started = False

        def capture_signals(self):
            raise AssertionError("Embedded server must not capture process signals")

        async def serve(self):
            with self.capture_signals():
                self.started = True
                while not self.should_exit:
                    await asyncio.sleep(0.01)

    async def scenario():
        server = OpenAIProxyServer("http://localhost/v1", None)
        await server.start()
        assert server._server.started
        await server.stop()
        assert server._server_task.done()

    with patch.dict(sys.modules, {
        "uvicorn": SimpleNamespace(Config=lambda *args, **kwargs: None,
                                   Server=EmbeddedServer)
    }):
        asyncio.run(scenario())


def test_tool_only_history_uses_empty_blocks_not_empty_text():
    result = anthropic_request_to_openai(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "bash",
                            "input": {"command": "pwd"},
                        }
                    ],
                },
            ]
        }
    )
    assert result["messages"][0]["content"] == []
    assert result["messages"][0]["tool_calls"][0]["function"]["name"] == "bash"


def test_nano_swe_builder_keeps_official_check_outside_workspace(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({
        "task_id": "HumanEval/0", "entry_point": "add",
        "prompt": "def add(a, b):\n    pass\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }) + "\n")
    manifest_path = build(
        source, "HumanEval/0", tmp_path / "tasks", tmp_path / "hidden",
    )
    manifest = json.loads(manifest_path.read_text())
    task = Path(manifest["workspace_root"])
    hidden = Path(manifest["hidden_tests"])
    assert not hidden.is_relative_to(task)
    assert "candidate(2, 3)" not in (task / "README.md").read_text()
    (task / "solution.py").write_text("def add(a, b):\n    return a - b\n")
    assert external_test_report(task, hidden, "", runner="pytest")["passed"] == 0
    with pytest.raises(ValueError):
        build(source, "HumanEval/0", task, task / "hidden")


def test_external_pytest_executes_functions_and_keeps_file_denominator(tmp_path):
    workspace, hidden = tmp_path / "task", tmp_path / "hidden"
    workspace.mkdir()
    hidden.mkdir()
    (hidden / "test_checks.py").write_text("def test_bad():\n    assert False\n")
    report = external_test_report(workspace, hidden, "", runner="pytest")
    assert report["total"] == 1 and report["passed"] == 0
    assert report["unit"] == "test_file"
    (hidden / "test_checks.py").write_text("def test_good():\n    assert True\n")
    assert external_test_report(workspace, hidden, "", runner="pytest")["passed"] == 1
    (hidden / "test_timeout.py").write_text("import time\ntime.sleep(60)\n")
    report = external_test_report(workspace, hidden, "", timeout_s=0.1)
    assert report["total"] == 2 and report["passed"] == 1
    assert report["outcomes"][1]["status"] == "timeout"
    with pytest.raises(ValueError):
        external_test_report(workspace, workspace, "")


@pytest.mark.parametrize("failure", [False, True])
def test_dsh_retains_raw_home_and_result_on_success_or_timeout(tmp_path, failure):
    workspace = tmp_path / "task"
    workspace.mkdir()

    class FakeHarness:
        def __init__(self, **kwargs):
            self.home = Path(kwargs["dsh_home"])

        def __enter__(self):
            self.home.mkdir(parents=True)
            (self.home / "session.v4.jsonl").write_text('{"type":"session"}\n')
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            pass

        def run(self, prompt):
            if failure:
                raise TimeoutError("secret-token")
            return SimpleNamespace(
                session_id="session-1",
                events=[],
                finish_reason="completed",
                final_response="Done",
            )

    agent = DshAgent(session_log_dir=str(tmp_path / "logs"))
    with patch.dict(
        sys.modules,
        {
            "deepseek_harness": SimpleNamespace(
                DeepSeekHarness=FakeHarness,
            )
        },
    ):
        data = {"workspace": str(workspace), "prompt": "Test", "reward": 1}
        if failure:
            with pytest.raises(TimeoutError):
                agent._run_sync(data, "http://localhost/v1", "secret-token")
        else:
            assert agent._run_sync(data, "http://localhost/v1", "secret-token") == 1
    capture = Path(agent.last_result["session_log_dir"])
    assert (capture / "dsh-home/session.v4.jsonl").is_file()
    saved = json.loads((capture / "result.json").read_text())
    assert saved == agent.last_result
    assert "secret-token" not in json.dumps(saved)
    assert saved["status"] == ("timeout" if failure else "completed")
