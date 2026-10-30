"""Read local official CLEVR question/image pairs without downloading data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CLEVRExample:
    question: str
    answer: str
    image_path: Path
    question_index: int


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
