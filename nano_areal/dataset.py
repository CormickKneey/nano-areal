"""
BFCL-V4 dataset loader.

Data files are downloaded from the gorilla GitHub repo via:
    uv run python scripts/download_bfcl.py

Default data dir: ~/.cache/nano-areal/bfcl/
Override via:     BFCL_DATA_DIR env var  or  DatasetConfig(data_dir=...)

Three files per category are merged by index:
  BFCL_v4_<category>.json                     ← question turns + initial_config
  multi_turn_func_doc/BFCL_v4_<cat>.json      ← function / tool definitions
  possible_answer/BFCL_v4_<cat>.json          ← ground truth per turn
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------

def default_data_dir() -> Path:
    return Path.home() / ".cache" / "nano-areal" / "bfcl"


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir:
        return Path(data_dir)
    env = os.environ.get("BFCL_DATA_DIR")
    if env:
        return Path(env)
    return default_data_dir()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _category_filename(category: str) -> str:
    return f"BFCL_v4_{category}.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BFCLSample:
    id: str
    # question[turn_idx] = list of {"role": ..., "content": ...}
    question: list[list[dict]]
    # tools in OpenAI tool-call format
    tools: list[dict]
    # ground_truth[turn_idx] = {fn_name: {arg: [accepted_values]}}
    ground_truth: list[dict]
    # initial environment state (file system, APIs, ...)
    initial_config: dict
    # API classes required for this test
    involved_classes: list[str]


def _to_openai_tool(fn: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _parse_question(raw) -> list[list[dict]]:
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [raw]       # single-turn: wrap in outer list
    return raw             # multi-turn: already list[list[dict]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    categories: list[str] = field(default_factory=lambda: [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ])
    max_samples: int = -1        # -1 = all
    data_dir: str = ""           # "" → use BFCL_DATA_DIR env or ~/.cache/nano-areal/bfcl


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BFCLDataset:
    """
    Loads BFCL-V4 multi-turn function-calling data from local JSONL files.

    Run `uv run python scripts/download_bfcl.py` first to fetch the data.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = resolve_data_dir(config.data_dir or None)
        self.samples: list[BFCLSample] = []
        self._load()

    def _load(self):
        missing: list[str] = []

        for category in self.config.categories:
            fname = _category_filename(category)

            # 1. Test entries (required)
            entries = _load_jsonl(self.data_dir / fname)
            if not entries:
                missing.append(str(self.data_dir / fname))
                continue

            # 2. Function/tool definitions (aligned by index)
            func_docs_raw = _load_jsonl(
                self.data_dir / "multi_turn_func_doc" / fname
            )
            # Each line may be a list of fn dicts, or {"function": [...]}
            func_docs: list[list[dict]] = []
            for fd in func_docs_raw:
                if isinstance(fd, list):
                    func_docs.append(fd)
                elif isinstance(fd, dict) and "function" in fd:
                    func_docs.append(fd["function"])
                else:
                    func_docs.append([])

            # 3. Ground truth (aligned by index)
            gt_raw = _load_jsonl(self.data_dir / "possible_answer" / fname)

            for i, entry in enumerate(entries):
                # Tools for this sample
                if func_docs and i < len(func_docs):
                    tools = [_to_openai_tool(f) for f in func_docs[i]]
                elif "function" in entry:
                    tools = [_to_openai_tool(f) for f in entry["function"]]
                else:
                    tools = []

                # Ground truth per turn
                if gt_raw and i < len(gt_raw):
                    gt_entry = gt_raw[i]
                    gt_per_turn = gt_entry if isinstance(gt_entry, list) else [gt_entry]
                else:
                    gt_per_turn = []

                question = _parse_question(entry.get("question", []))

                # Ensure gt covers all turns
                while len(gt_per_turn) < len(question):
                    gt_per_turn.append({})

                self.samples.append(BFCLSample(
                    id=entry.get("id", f"{category}_{i}"),
                    question=question,
                    tools=tools,
                    ground_truth=gt_per_turn,
                    initial_config=entry.get("initial_config", {}),
                    involved_classes=entry.get("involved_classes", []),
                ))

        if missing:
            raise FileNotFoundError(
                f"BFCL data files not found:\n"
                + "\n".join(f"  {p}" for p in missing)
                + f"\n\nRun:  uv run python scripts/download_bfcl.py"
                + (f"\n      or set BFCL_DATA_DIR to an existing data directory" if not missing else "")
            )

        if self.config.max_samples > 0:
            self.samples = self.samples[: self.config.max_samples]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BFCLSample:
        return self.samples[idx]

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = True,
    ) -> Iterator[list[BFCLSample]]:
        indices = list(range(len(self.samples)))
        while True:
            if shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = [self.samples[j] for j in indices[i : i + batch_size]]
                if batch:
                    yield batch
            if not repeat:
                break


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def build_system_prompt(involved_classes: list[str]) -> str:
    classes_str = ", ".join(involved_classes) if involved_classes else "the available tools"
    return (
        f"You are a helpful assistant with access to {classes_str}. "
        "When you need to call a function, respond with a JSON tool call. "
        "Think step-by-step before calling any function. "
        "Only call functions that are necessary to answer the user's request."
    )
