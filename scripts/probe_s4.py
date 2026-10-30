"""Run a bounded local nano-swe foundation group, not a benchmark or RL run."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from nano_areal.agent.dsh import DshAgent
from nano_areal.agent.dsh_audit import reconcile_session
from nano_areal.engine.proxy import OpenAIProxyServer, OpenAIProxyWorkflow


class AuditProxy(OpenAIProxyServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend_audit = []
        self.failures = []
        self.pending = set()

    async def start(self):
        await super().start()
        create = self._vllm.chat.completions.create

        async def capture(**request):
            entry = {"response": None}
            self.backend_audit.append(entry)
            task = asyncio.current_task()
            self.pending.add(task)
            try:
                response = await create(**request)
            except Exception as error:
                entry["error_type"] = type(error).__name__
                raise
            finally:
                self.pending.discard(task)
            entry["response"] = response.model_dump()
            return response

        self._vllm.chat.completions.create = capture

    async def _handle_chat_completion(self, body, session, return_payload=False):
        body = {**body, "temperature": self.temperature}
        try:
            result = await super()._handle_chat_completion(
                body, session, return_payload
            )
        except Exception as error:
            self.failures.append(
                {
                    "error": str(error),
                    "request": body,
                    "proxy_prompt_ids": self._tokenize_messages(
                        body["messages"], body.get("tools")
                    ),
                }
            )
            raise
        record = session.last_interaction()
        output = sum(record.response_mask)
        self.audit.append(
            {
                "response_id": record.response_id,
                "prompt_tokens": len(record.input_ids) - output,
                "output_tokens": output,
                "tool_calls": record.tool_calls,
                "messages": record.messages,
                "finish_reason": record.finish_reason,
            }
        )
        return result


async def probe(args):
    root = args.task_root.resolve()
    hidden = args.hidden_tests.resolve()
    manifest = None
    if args.task_manifest:
        manifest = json.loads(args.task_manifest.read_text())
        root = Path(manifest["workspace_root"]).resolve()
        hidden = Path(manifest["hidden_tests"]).resolve()
    if hidden.is_relative_to(root):
        raise ValueError("Hidden tests must be outside all task workspaces")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    proxy = AuditProxy(args.base_url, tokenizer, model_name=args.model)
    proxy.temperature = args.temperature
    report = {
        "scope": "local synthetic nano-swe foundation; not benchmark accuracy",
        "task_groups": 1,
        "attempted": args.episodes,
        "episodes": [],
        "temperature": args.temperature,
    }
    if manifest:
        report["scope"] = "HumanEval-derived nano-swe foundation, one task group"
        report["task"] = manifest
    await proxy.start()
    try:
        for index in range(args.episodes):
            root.mkdir(parents=True, exist_ok=True)
            workspace = Path(tempfile.mkdtemp(prefix=f"episode-{index}-", dir=root))
            (workspace / "README.md").write_text(
                (root / "README.md").read_text()
                if manifest
                else "Implement solution.py with add(a, b), "
                "returning the arithmetic sum.\n"
            )
            agent = DshAgent(
                model_name=args.model,
                max_output_tokens=512,
                timeout_seconds=args.timeout,
                patches=(str(Path("scripts/dsh_patch.yml").resolve()),),
                session_log_dir=str(args.output / "sessions"),
            )
            proxy.audit = []
            proxy.failures = []
            proxy.backend_audit = []
            row = {"index": index, "workspace": str(workspace)}
            try:
                trajectories = await OpenAIProxyWorkflow(agent, proxy).arun_episode(
                    {
                        "workspace": str(workspace),
                        "hidden_test_dir": str(hidden),
                        "hidden_test_runner": "pytest",
                        "prompt": (
                            "Work in the current directory using actual bash "
                            "tool calls. First run pwd and cat README.md. "
                            "After reading the result, "
                            "use a second bash call to implement the README interface "
                            "in solution.py. "
                            "Use a heredoc or printf to preserve Python newlines, "
                            "not echo. "
                            "After that result, use a third bash call to run "
                            "a quick local Python check of the implemented function. "
                            "Do not merely suggest commands or ask for a path."
                        ),
                    }
                )
                row["exported_interactions"] = len(trajectories)
                row["exported_tokens"] = [
                    [len(item.input_ids) - int(item.response_mask.sum()),
                     int(item.response_mask.sum())] for item in trajectories
                ]
            except Exception as error:
                row["error_type"] = type(error).__name__
            if proxy.pending:
                _, pending = await asyncio.wait(tuple(proxy.pending), timeout=30)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                    row["cancelled_backend_requests"] = len(pending)
            row["result"] = agent.last_result
            capture = Path(agent.last_result.get("session_log_dir") or
                           args.output / f"setup-failure-{index}")
            capture.mkdir(parents=True, exist_ok=True)
            logs = list(capture.glob("dsh-home/sessions/*/*/session.v4.jsonl"))
            events = [
                json.loads(line)
                for log in logs
                for line in log.read_text().splitlines()
            ]
            row["reconciliation"] = reconcile_session(events, proxy.audit)
            expected_tokens = [[item["prompt_tokens"], item["output_tokens"]]
                               for item in proxy.audit]
            if row.get("exported_tokens") != expected_tokens:
                row["reconciliation"]["matched"] = False
                row["reconciliation"]["errors"].append("Export token count mismatch")
            row["failed_proxy_requests"] = len(proxy.failures)
            attempts = sum(event.get("type") == "assistant/attempt" for event in events)
            row["failed_dsh_attempts"] = attempts
            row["backend_requests"] = len(proxy.backend_audit)
            row["backend_tokens"] = [
                entry["response"].get("usage") if entry["response"] else None
                for entry in proxy.backend_audit
            ]
            request_count = len(proxy.audit) + len(proxy.failures)
            if (attempts != len(proxy.failures)
                    or len(proxy.backend_audit) != request_count):
                row["reconciliation"]["matched"] = False
                row["reconciliation"]["errors"].append("Request attempt count mismatch")
            (capture / "backend.json").write_text(
                json.dumps(proxy.backend_audit) + "\n"
            )
            (capture / "failures.json").write_text(json.dumps(proxy.failures) + "\n")
            (capture / "proxy.json").write_text(
                json.dumps(proxy.audit, indent=2) + "\n"
            )
            tool_turns = sum(bool(item["tool_calls"]) for item in proxy.audit)
            row["tool_turns"] = tool_turns
            row["passed"] = (
                row["reconciliation"]["matched"]
                and len(logs) == 1
                and not proxy.failures
                and tool_turns >= 2
                and row.get("exported_interactions") == len(proxy.audit)
                and agent.last_result.get("finish_reason") == "completed"
                and agent.last_result.get("verified_reward") == 1.0
            )
            row["classification"] = (
                "passed"
                if row["passed"]
                else agent.last_result.get("status", "error")
                if ("error_type" in row or agent.last_result.get("status")
                    in {"error", "timeout", "max-tokens"})
                else "accounting-mismatch"
                if not row["reconciliation"]["matched"]
                else "insufficient-tool-turns"
                if tool_turns < 2
                else "verification-failed"
            )
            report["episodes"].append(row)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    finally:
        await proxy.stop()
    report["passed"] = sum(row["passed"] for row in report["episodes"])
    report["pass_rate"] = report["passed"] / args.episodes
    report["task_groups_passed"] = int(report["passed"] == args.episodes)
    report["test_files_planned"] = len(list(hidden.glob("test_*.py"))) * args.episodes
    report["test_files_passed"] = sum(
        row["result"].get("verification", {}).get("passed", 0)
        for row in report["episodes"]
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == args.episodes else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    parser.add_argument(
        "--model-path", default=".s1-cache/models/Qwen3-VL-4B-Instruct-4bit"
    )
    parser.add_argument("--task-root", type=Path, default=Path(".s1-cache/nano-swe"))
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument(
        "--hidden-tests", type=Path, default=Path(".s1-cache/nano-swe-hidden-tests")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/s4/sync-smoke"))
    parser.add_argument("--episodes", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--temperature", type=float, default=0)
    raise SystemExit(asyncio.run(probe(parser.parse_args())))
