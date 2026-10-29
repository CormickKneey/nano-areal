"""Hidden-test reward for a dsh task directory.

The score is the fraction of ``test_*.py`` files that pass. No tests means
unverified (``None``), which the agent turns into reward 0. The agent process
does not grade itself.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
