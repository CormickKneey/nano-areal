"""
Single-turn visual QA tasks (mirrors AReaL's clevr_count_70k / geometry3k):

  clevr        BUAADreamer/clevr_count_70k  - "How many objects?" exact count
  geometry3k   hiyouga/geometry3k           - numeric answer, \\boxed{} extraction
  synthetic    PIL-drawn shapes, offline    - CPU smoke test without downloads

All three share VisionQASample / VisionQAAgent; only the loader and the
answer matcher differ.
"""
from __future__ import annotations

import base64
import io
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from nano_areal.agent.base import AgentBase
from nano_areal.tasks.base import ListDataset

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


@dataclass
class VisionQASample:
    image: object                # PIL.Image.Image | str (path / URL / data URI)
    question: str
    answer: str
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer extraction / rewards
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Prefer \\boxed{...}, then '<answer>...</answer>', then the last number."""
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    m2 = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    if m2:
        return m2[-1].strip()
    nums = _NUM.findall(text)
    return nums[-1] if nums else text.strip()


def _to_float(s: str) -> float | None:
    s = s.strip().rstrip(".")
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def clevr_count_reward(response: str, answer: str) -> float:
    """Exact integer match on the extracted answer."""
    pred, gt = _to_float(extract_answer(response)), _to_float(answer)
    return float(pred is not None and gt is not None and abs(pred - gt) < 1e-6)


def geometry3k_reward(response: str, answer: str, rel_tol: float = 1e-2) -> float:
    """Numeric match with relative tolerance (Geometry3K answers are often decimals)."""
    pred, gt = _to_float(extract_answer(response)), _to_float(answer)
    if pred is None or gt is None:
        return float(extract_answer(response).strip().lower() == answer.strip().lower())
    return float(abs(pred - gt) <= rel_tol * max(1.0, abs(gt)))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def image_to_data_uri(image) -> str:
    if isinstance(image, str):
        return image
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class VisionQAAgent(AgentBase):
    """One image + one question → one answer; reward from `reward_fn`."""

    def __init__(
        self,
        reward_fn: Callable[[str, str], float],
        system_prompt: str = (
            "Answer the question about the image. Think briefly, then put the final "
            "answer in \\boxed{}."
        ),
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ):
        self.reward_fn = reward_fn
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def run(self, data: VisionQASample, **extra_kwargs) -> dict[str, float]:
        client_kwargs: dict = dict(
            base_url=extra_kwargs["base_url"], api_key=extra_kwargs["api_key"], max_retries=0
        )
        if extra_kwargs.get("http_client") is not None:
            client_kwargs["http_client"] = extra_kwargs["http_client"]
        client = AsyncOpenAI(**client_kwargs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(data.image)}},
                    {"type": "text", "text": data.question},
                ],
            },
        ]
        resp = await client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        return {resp.id: self.reward_fn(text, data.answer)}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _hf_rows(name: str, split: str, limit: int):
    from datasets import load_dataset  # heavy optional dep

    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_clevr_count(path: str = "BUAADreamer/clevr_count_70k", split: str = "train", limit: int = 0) -> ListDataset:
    """Rows: {images: [PIL], problem: str, answer: str}."""
    out = []
    for row in _hf_rows(path, split, limit):
        imgs = row.get("images") or [row.get("image")]
        out.append(
            VisionQASample(
                image=imgs[0],
                question=str(row.get("problem") or row.get("question")).replace("<image>", "").strip(),
                answer=str(row["answer"]),
                meta={"source": "clevr_count"},
            )
        )
    return ListDataset(out)


def load_geometry3k(path: str = "hiyouga/geometry3k", split: str = "train", limit: int = 0) -> ListDataset:
    """Rows: {images: [PIL], problem: str, answer: str}."""
    out = []
    for row in _hf_rows(path, split, limit):
        imgs = row.get("images") or [row.get("image")]
        out.append(
            VisionQASample(
                image=imgs[0],
                question=str(row["problem"]).replace("<image>", "").strip(),
                answer=str(row["answer"]),
                meta={"source": "geometry3k"},
            )
        )
    return ListDataset(out)


def synthetic_clevr(n: int = 32, size: int = 64, max_objects: int = 5, seed: int = 0) -> ListDataset:
    """Offline stand-in for CLEVR-count: coloured shapes on a grey canvas."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = ["red", "blue", "green", "yellow", "purple", "cyan"]
    out = []
    for i in range(n):
        k = rng.randint(1, max_objects)
        img = Image.new("RGB", (size, size), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        for _ in range(k):
            x, y = rng.randint(4, size - 16), rng.randint(4, size - 16)
            r = rng.randint(4, 8)
            c = rng.choice(colors)
            if rng.random() < 0.5:
                draw.ellipse([x, y, x + 2 * r, y + 2 * r], fill=c)
            else:
                draw.rectangle([x, y, x + 2 * r, y + 2 * r], fill=c)
        out.append(
            VisionQASample(
                image=img,
                question="How many objects are there in the image?",
                answer=str(k),
                meta={"source": "synthetic_clevr", "idx": i},
            )
        )
    return ListDataset(out)
