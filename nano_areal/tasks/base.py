"""Task = dataset + agent + reward, the unit `train.py --task` selects."""
from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from nano_areal.agent.base import AgentBase

S = TypeVar("S")


class ListDataset(Generic[S]):
    """In-memory dataset with the `iter_batches` protocol used by train.py."""

    def __init__(self, samples: Sequence[S]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> S:
        return self.samples[idx]

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, repeat: bool = True, seed: int = 0
    ) -> Iterator[list[S]]:
        rng = random.Random(seed)
        while True:
            order = list(range(len(self.samples)))
            if shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), batch_size):
                yield [self.samples[j] for j in order[i : i + batch_size]]
            if not repeat:
                return


@dataclass
class Task:
    name: str
    dataset: ListDataset
    agent: AgentBase
    # True when samples carry images (proxy needs a PromptProcessor with an
    # image processor; the model must keep its vision tower)
    multimodal: bool = False
