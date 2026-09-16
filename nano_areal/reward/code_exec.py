"""
Sandboxed Python execution for code rewards (nano-SWE, chart, dsh tasks).

Each verification runs in a fresh temporary workspace as a separate
`python -I` subprocess with a wall-clock timeout and no inherited environment
besides PATH. This is process isolation, not a security sandbox — run untrusted
policies inside a container on shared machines.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


_BASE_ENV_KEYS = ("PATH", "SYSTEMROOT", "HOME", "TMPDIR", "MPLBACKEND")


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _BASE_ENV_KEYS if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    if extra:
        env.update(extra)
    return env


async def run_python(
    files: dict[str, str],
    argv: list[str],
    timeout: float = 30.0,
    workspace: str | Path | None = None,
    keep: bool = False,
) -> tuple[ExecResult, Path]:
    """Write `files` into a workspace and run `python -I argv...` there."""
    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="nano-areal-exec-"))
    ws.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        *argv,
        cwd=str(ws),
        env=_clean_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        out, err = await proc.communicate()
    res = ExecResult(proc.returncode if not timed_out else -9,
                     out.decode(errors="replace"), err.decode(errors="replace"), timed_out)
    if not keep and workspace is None:
        shutil.rmtree(ws, ignore_errors=True)
    return res, ws


def extract_code_block(text: str, lang: str = "python") -> str:
    """Last fenced code block (``` or ```python); whole text when none."""
    import re

    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.S)
    if not blocks:
        return text.strip()
    for b in reversed(blocks):
        if lang in b or lang == "python":
            return b.strip()
    return blocks[-1].strip()


_TEST_RUNNER = """\
import importlib, sys, traceback
sys.path.insert(0, ".")
try:
    import solution  # noqa
except Exception:
    traceback.print_exc()
    print("__RESULT__ 0 0")
    sys.exit(1)
ns = dict(vars(solution))
tests = open("tests.txt").read().split("\\n__SEP__\\n")
passed = 0
for t in tests:
    if not t.strip():
        continue
    try:
        exec(t, dict(ns))
        passed += 1
    except Exception:
        pass
print(f"__RESULT__ {passed} {len([t for t in tests if t.strip()])}")
sys.exit(0 if passed == len([t for t in tests if t.strip()]) else 2)
"""


async def run_assert_tests(code: str, tests: list[str], timeout: float = 20.0) -> tuple[float, ExecResult]:
    """Fraction of `assert` statements that pass against `code` (MBPP/HumanEval style)."""
    res, _ = await run_python(
        {"solution.py": code, "tests.txt": "\n__SEP__\n".join(tests), "_run.py": _TEST_RUNNER},
        ["_run.py"],
        timeout=timeout,
    )
    frac = 0.0
    for line in res.stdout.splitlines():
        if line.startswith("__RESULT__"):
            _, p, n = line.split()
            frac = int(p) / max(1, int(n))
    return frac, res
