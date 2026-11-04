"""Materialize one trusted local HumanEval row as a nano-swe task group."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def build(data_path: Path, task_id: str, root: Path, hidden_root: Path) -> Path:
    root, hidden_root = root.resolve(), hidden_root.resolve()
    if hidden_root.is_relative_to(root) or root.is_relative_to(hidden_root):
        raise ValueError("Task and trusted-test roots must be disjoint")
    opener = gzip.open if data_path.suffix == ".gz" else open
    with opener(data_path, "rt") as stream:
        matches = [
            json.loads(line)
            for line in stream
            if line.strip() and json.loads(line).get("task_id") == task_id
        ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching HumanEval row")
    row = matches[0]
    entry = row["entry_point"]
    if not entry.isidentifier():
        raise ValueError("Invalid HumanEval entry point")
    slug = "human-eval-" + task_id.split("/")[-1]
    if not slug.removeprefix("human-eval-").isdigit():
        raise ValueError("Expected a numeric HumanEval task ID")
    group, hidden = root / slug, hidden_root / slug
    group.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(parents=True, exist_ok=True)
    (group / "README.md").write_text(
        f"# {task_id}\n\nImplement the following interface in solution.py.\n\n"
        f"```python\n{row['prompt']}\n```\n"
    )
    (hidden / "test_solution.py").write_text(
        "import os\nimport sys\n"
        'sys.path.insert(0, os.environ["TASK_WORKSPACE"])\n'
        f"from solution import {entry}\n\n{row['test']}\n\n"
        f"def test_official_check():\n    check({entry})\n"
    )
    manifest = {
        "task_id": task_id,
        "workspace_root": str(group),
        "hidden_tests": str(hidden),
        "dataset": "openai/human-eval",
        "entry_point": entry,
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "hidden_assertion_sites": row["test"].count("assert "),
    }
    path = group / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--task-id", default="HumanEval/0")
    parser.add_argument("--root", type=Path, default=Path(".s1-cache/nano-swe"))
    parser.add_argument(
        "--hidden-root", type=Path, default=Path(".s1-cache/nano-swe-hidden-tests")
    )
    args = parser.parse_args()
    print(build(args.data_path, args.task_id, args.root, args.hidden_root))
