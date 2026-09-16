"""
Outcome reward for coding tasks solved in a workspace (dsh / tool agents).

The agent edits files in `workspace`; verification copies `solution.py` (or
the file named by `entry_file`) out of the workspace and runs the sample's
assert tests in a fresh subprocess. Reward is the pass fraction, optionally
mixed with a syntax-validity bonus so partially working edits are not all
scored 0.
"""
from __future__ import annotations

import ast
from pathlib import Path

from nano_areal.reward.code_exec import run_assert_tests
from nano_areal.tasks.code import CodeSample


def workspace_files(sample: CodeSample) -> dict[str, str]:
    """Initial repo contents for a nano-SWE sample."""
    return {
        "README.md": sample.readme(),
        "solution.py": sample.starter if sample.starter and "def " in sample.starter else "# implement here\n",
        "tests/test_solution.py": "from solution import *  # noqa: F403\n\n"
        + "\n\n".join(f"def test_{i}():\n    {t}" for i, t in enumerate(sample.tests)),
    }


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


async def dsh_task_reward(
    sample: CodeSample,
    workspace: str | Path,
    entry_file: str = "solution.py",
    syntax_bonus: float = 0.1,
    timeout: float = 20.0,
) -> tuple[float, dict]:
    path = Path(workspace) / entry_file
    if not path.exists():
        return 0.0, {"reason": "missing_solution"}
    code = path.read_text()
    if not syntax_ok(code):
        return 0.0, {"reason": "syntax_error"}
    frac, res = await run_assert_tests(code, sample.tests, timeout=timeout)
    reward = frac if frac >= 1.0 else min(0.99, frac * (1 - syntax_bonus) + syntax_bonus)
    return reward, {"pass_frac": frac, "timed_out": res.timed_out, "rc": res.returncode}
