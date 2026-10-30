"""Offline regressions for dsh capture, accounting, and trusted verification."""

from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nano_areal.agent.dsh import DshAgent, run_with_deadline
from nano_areal.agent.dsh_audit import reconcile_session
from nano_areal.engine.messages import anthropic_request_to_openai
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
