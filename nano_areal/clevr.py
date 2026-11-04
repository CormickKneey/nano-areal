"""Read local official CLEVR question/image pairs without downloading data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CLEVRExample:
    question: str
    answer: str
    image_path: Path | None
    question_index: int
    image_bytes: bytes | None = None


def load_clevr(root: Path, split="train", limit=1):
    if split not in {"train", "val"} or limit <= 0:
        raise ValueError("Expected train/val split and a positive sample limit")
    question_file = root / "questions" / f"CLEVR_{split}_questions.json"
    if not question_file.is_file():
        raise FileNotFoundError(f"Local CLEVR questions not found: {question_file}")
    questions = json.loads(question_file.read_text())["questions"]
    images = (root / "images" / split).resolve()
    examples = []
    for question in questions[:limit]:
        image_path = (images / question["image_filename"]).resolve()
        if not image_path.is_relative_to(images) or not image_path.is_file():
            raise ValueError("CLEVR image is missing or escapes its split directory")
        if not question.get("question") or "answer" not in question:
            raise ValueError("CLEVR example requires a question and answer")
        examples.append(
            CLEVRExample(
                question["question"],
                str(question["answer"]),
                image_path,
                int(question["question_index"]),
            )
        )
    if not examples:
        raise ValueError("CLEVR split contains no examples")
    return examples


def load_clevr_count_parquet(path: Path, limit=1):
    """Load the plan-approved BUAADreamer CLEVR count parquet subset."""
    if limit <= 0:
        raise ValueError("Expected a positive sample limit")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("The parquet subset requires pyarrow") from error

    table = parquet.read_table(path).slice(0, limit)
    examples = []
    for index, row in enumerate(table.to_pylist()):
        images = row.get("images") or []
        if not row.get("problem") or "answer" not in row or not images:
            raise ValueError("CLEVR count row requires question, answer, and image")
        image_bytes = images[0].get("bytes")
        if not image_bytes:
            raise ValueError("CLEVR count row has no embedded image bytes")
        question = row["problem"].replace("<image>", "", 1).strip()
        examples.append(
            CLEVRExample(
                question=question,
                answer=str(row["answer"]),
                image_path=None,
                question_index=index,
                image_bytes=image_bytes,
            )
        )
    if not examples:
        raise ValueError("CLEVR count parquet contains no examples")
    return examples
