"""
Chart-to-code (ChartMimic-style): the model sees a rendered chart and writes
matplotlib code that reproduces it. Reward = SSIM-lite between the rendered
candidate and the reference image (0 when the code fails to run).

`synthetic_charts()` renders simple bar/line charts locally, so the task works
offline; `load_chartmimic()` reads the HF release when available.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from nano_areal.agent.base import AgentBase
from nano_areal.reward.code_exec import extract_code_block, run_python
from nano_areal.tasks.base import ListDataset
from nano_areal.tasks.vision import image_to_data_uri


@dataclass
class ChartSample:
    image: object                 # PIL.Image | str
    instruction: str = "Write matplotlib code that reproduces this chart as closely as possible."
    reference_code: str = ""
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Visual similarity
# ---------------------------------------------------------------------------

def ssim_lite(a, b, size: int = 64) -> float:
    """Global SSIM on grayscale thumbnails (no windowing; enough as a dense reward)."""
    import numpy as np

    def prep(img):
        g = img.convert("L").resize((size, size))
        return np.asarray(g, dtype=np.float64) / 255.0

    x, y = prep(a), prep(b)
    c1, c2 = 0.01**2, 0.03**2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(max(0.0, min(1.0, s)))


_RENDER_WRAPPER = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("candidate.py").read())
plt.savefig("out.png", dpi=72)
"""


async def render_code(code: str, timeout: float = 30.0):
    """Run matplotlib code in a subprocess, return PIL image or None."""
    from PIL import Image

    res, ws = await run_python(
        {"candidate.py": code.replace("plt.show()", ""), "_render.py": _RENDER_WRAPPER},
        ["_render.py"],
        timeout=timeout,
        keep=True,
    )
    img = None
    out = ws / "out.png"
    if res.ok and out.exists():
        img = Image.open(io.BytesIO(out.read_bytes())).convert("RGB")
    import shutil

    shutil.rmtree(ws, ignore_errors=True)
    return img


async def chart_reward(response: str, reference) -> float:
    img = await render_code(extract_code_block(response))
    return 0.0 if img is None else ssim_lite(img, reference)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ChartAgent(AgentBase):
    def __init__(self, temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 1024):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def run(self, data: ChartSample, **extra_kwargs) -> dict[str, float]:
        from PIL import Image

        client_kwargs: dict = dict(
            base_url=extra_kwargs["base_url"], api_key=extra_kwargs["api_key"], max_retries=0
        )
        if extra_kwargs.get("http_client") is not None:
            client_kwargs["http_client"] = extra_kwargs["http_client"]
        client = AsyncOpenAI(**client_kwargs)
        resp = await client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "Reply with one ```python block using matplotlib.pyplot as plt. "
                                             "Do not call plt.show()."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_uri(data.image)}},
                        {"type": "text", "text": data.instruction},
                    ],
                },
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        ref = data.image if not isinstance(data.image, str) else Image.open(data.image)
        return {resp.id: await chart_reward(resp.choices[0].message.content or "", ref)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def synthetic_charts(n: int = 16, seed: int = 0) -> ListDataset:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.randint(3, 6)
        ys = [rng.randint(1, 10) for _ in range(k)]
        kind = rng.choice(["bar", "plot"])
        code = f"plt.figure(figsize=(3, 2))\nplt.{kind}(range({k}), {ys})\nplt.title('chart {i}')\n"
        fig = plt.figure(figsize=(3, 2))
        if kind == "bar":
            plt.bar(range(k), ys)
        else:
            plt.plot(range(k), ys)
        plt.title(f"chart {i}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72)
        plt.close(fig)
        buf.seek(0)
        out.append(ChartSample(image=Image.open(buf).convert("RGB"), reference_code=code, meta={"source": "synthetic"}))
    return ListDataset(out)


def load_chartmimic(path: str = "ChartMimic/ChartMimic", split: str = "test", limit: int = 0) -> ListDataset:
    from datasets import load_dataset

    ds = load_dataset(path, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ListDataset(
        [ChartSample(image=row["image"], reference_code=row.get("code", ""), meta={"source": "chartmimic"})
         for row in ds]
    )
