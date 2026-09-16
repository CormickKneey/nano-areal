"""
nano-SWE: repository-shaped coding tasks built from MBPP / HumanEval.

Each sample is a small "repo" (README with the spec, an empty `solution.py`
and hidden assert tests). The direct `CodeAgent` asks the model for the file
in one turn; `DSHAgent` (agent/dsh.py) lets DeepSeek Harness edit the same
workspace with tools. Both are scored with `run_assert_tests`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI

from nano_areal.agent.base import AgentBase
from nano_areal.reward.code_exec import extract_code_block, run_assert_tests
from nano_areal.tasks.base import ListDataset


@dataclass
class CodeSample:
    task_id: str
    prompt: str                      # natural-language spec (+ signature)
    tests: list[str]                 # python assert statements
    entry_point: str = ""
    starter: str = ""                # code shown to the model (HumanEval prompt)
    meta: dict = field(default_factory=dict)

    def readme(self) -> str:
        sig = f"\n\nSignature / starter:\n```python\n{self.starter}\n```" if self.starter else ""
        return f"# Task {self.task_id}\n\n{self.prompt}{sig}\n\nImplement the solution in `solution.py`."


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def load_mbpp(split: str = "train", limit: int = 0, path: str = "google-research-datasets/mbpp") -> ListDataset:
    from datasets import load_dataset

    ds = load_dataset(path, "sanitized", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = [
        CodeSample(
            task_id=f"mbpp-{row['task_id']}",
            prompt=row["prompt"],
            tests=list(row["test_list"]),
            starter=row.get("test_setup_code", "") or "",
            meta={"source": "mbpp"},
        )
        for row in ds
    ]
    return ListDataset(out)


def load_humaneval(limit: int = 0, path: str = "openai/openai_humaneval") -> ListDataset:
    from datasets import load_dataset

    ds = load_dataset(path, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for row in ds:
        # HumanEval ships a `check(candidate)` function; adapt to assert list.
        test = row["test"] + f"\ncheck({row['entry_point']})"
        out.append(
            CodeSample(
                task_id=row["task_id"],
                prompt=row["prompt"],
                tests=[test],
                entry_point=row["entry_point"],
                starter=row["prompt"],
                meta={"source": "humaneval"},
            )
        )
    return ListDataset(out)


def builtin_nano_swe() -> ListDataset:
    """Tiny offline set for smoke tests."""
    return ListDataset(
        [
            CodeSample(
                "builtin-add",
                "Write a function add(a, b) that returns the sum of two numbers.",
                ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
                entry_point="add",
            ),
            CodeSample(
                "builtin-rev",
                "Write a function reverse_words(s) that reverses the order of words in s.",
                ["assert reverse_words('a b c') == 'c b a'", "assert reverse_words('x') == 'x'"],
                entry_point="reverse_words",
            ),
            CodeSample(
                "builtin-fib",
                "Write a function fib(n) returning the n-th Fibonacci number with fib(0)=0, fib(1)=1.",
                ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55"],
                entry_point="fib",
            ),
        ]
    )


def build_nano_swe(source: str = "builtin", limit: int = 0) -> ListDataset:
    if source in ("", "builtin"):
        return builtin_nano_swe()
    if source == "mbpp":
        return load_mbpp(limit=limit)
    if source == "humaneval":
        return load_humaneval(limit=limit)
    raise ValueError(f"unknown nano-swe source {source!r} (builtin | mbpp | humaneval)")


# ---------------------------------------------------------------------------
# Direct (single-turn) agent
# ---------------------------------------------------------------------------

class CodeAgent(AgentBase):
    def __init__(self, temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 1024, timeout: float = 20.0):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def run(self, data: CodeSample, **extra_kwargs) -> dict[str, float]:
        client_kwargs: dict = dict(
            base_url=extra_kwargs["base_url"], api_key=extra_kwargs["api_key"], max_retries=0
        )
        if extra_kwargs.get("http_client") is not None:
            client_kwargs["http_client"] = extra_kwargs["http_client"]
        client = AsyncOpenAI(**client_kwargs)
        resp = await client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "You are a Python programmer. Reply with a single ```python "
                                             "code block containing solution.py."},
                {"role": "user", "content": data.readme()},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        code = extract_code_block(resp.choices[0].message.content or "")
        frac, _ = await run_assert_tests(code, data.tests, timeout=self.timeout)
        return {resp.id: frac}
