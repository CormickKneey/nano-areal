"""Hidden-test reward for a dsh task directory.

The score is the fraction of ``test_*.py`` files that pass. No tests means
unverified (``None``), which the agent turns into reward 0. The agent process
does not grade itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def external_test_pass_rate(
    workspace: Path,
    test_dir: Path,
    response: str,
    timeout_s: float = 30.0,
) -> float:
    return external_test_report(workspace, test_dir, response, timeout_s)["pass_rate"]


def external_test_report(
    workspace: Path,
    test_dir: Path,
    response: str,
    timeout_s: float = 30.0,
    runner: str = "script",
) -> dict:
    """Run trusted tests outside the agent workspace, only after dsh exits."""
    workspace = workspace.resolve()
    test_dir = test_dir.resolve()
    if test_dir == workspace or workspace in test_dir.parents:
        raise ValueError("Hidden tests must be outside the task workspace")
    tests = sorted(test_dir.glob("test_*.py"))
    if not tests:
        raise ValueError("No external hidden tests found")
    if runner not in {"script", "pytest"}:
        raise ValueError("Hidden test runner must be script or pytest")
    with tempfile.TemporaryDirectory(prefix="nano-areal-verifier-") as directory:
        response_path = Path(directory) / "response.txt"
        response_path.write_text(response)
        environment = {
            **os.environ,
            "TASK_WORKSPACE": str(workspace),
            "TASK_RESPONSE_PATH": str(response_path),
        }
        outcomes = []
        for test in tests:
            try:
                result = subprocess.run(
                    (
                        [sys.executable, "-m", "pytest", "-q", str(test)]
                        if runner == "pytest"
                        else [sys.executable, str(test)]
                    ),
                    cwd=workspace,
                    env=environment,
                    capture_output=True,
                    timeout=timeout_s,
                    check=False,
                )
                outcomes.append(
                    {
                        "test": test.name,
                        "status": (
                            "passed"
                            if result.returncode == 0
                            else "collection-error"
                            if runner == "pytest" and result.returncode == 2
                            else "failed"
                        ),
                        "returncode": result.returncode,
                        "stdout": result.stdout.decode(errors="replace"),
                        "stderr": result.stderr.decode(errors="replace"),
                    }
                )
            except subprocess.TimeoutExpired:
                outcomes.append({"test": test.name, "status": "timeout"})
        passed = sum(item["status"] == "passed" for item in outcomes)
        return {
            "passed": passed,
            "total": len(tests),
            "unit": "test_file",
            "runner": runner,
            "pass_rate": passed / len(tests),
            "outcomes": outcomes,
        }


def hidden_test_pass_rate(workspace: Path, timeout_s: float = 120.0) -> float | None:
    if not workspace.is_dir():
        return None
    tests = sorted(p for p in workspace.glob("test_*.py") if p.is_file())
    if not tests:
        return None
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", *[str(p) for p in tests]],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    passed, failed = _parse_pytest_summary(proc.stdout + "\n" + proc.stderr)
    total = passed + failed
    if total == 0:
        return 1.0 if proc.returncode == 0 else 0.0
    return passed / total


def _parse_pytest_summary(text: str) -> tuple[int, int]:
    passed = failed = 0
    for line in reversed(text.splitlines()):
        parts = line.replace(",", " ").split()
        counts = {"passed": 0, "failed": 0}
        seen = False
        for index, token in enumerate(parts):
            if token not in {"passed", "failed", "error", "errors"} or index == 0:
                continue
            try:
                count = int(parts[index - 1])
            except ValueError:
                continue
            seen = True
            if token == "passed":
                counts["passed"] = count
            else:
                counts["failed"] += count
        if seen:
            return counts["passed"], counts["failed"]
    return passed, failed
