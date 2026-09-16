"""Task registry: `build_task(cfg)` → Task(dataset, agent)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from nano_areal.tasks.base import ListDataset, Task

if TYPE_CHECKING:
    from nano_areal.config import NanoArealConfig

TASKS = ("bfcl", "clevr", "geometry3k", "nano_swe", "chart", "dsh")


def build_task(cfg: "NanoArealConfig") -> Task:
    r = cfg.rollout
    name, data, limit = cfg.task, cfg.task_data, cfg.task_limit
    sampling = dict(temperature=r.temperature, top_p=r.top_p, max_tokens=r.max_new_tokens)

    if name == "bfcl":
        from nano_areal.agent.bfcl import BFCLAgent
        from nano_areal.dataset import BFCLDataset

        return Task("bfcl", BFCLDataset(cfg.dataset), BFCLAgent(max_turns=r.max_turns, **sampling))

    if name == "clevr":
        from nano_areal.tasks.vision import (
            VisionQAAgent,
            clevr_count_reward,
            load_clevr_count,
            synthetic_clevr,
        )

        ds = synthetic_clevr(limit or 32) if data in ("", "synthetic") else load_clevr_count(data, limit=limit)
        return Task("clevr", ds, VisionQAAgent(clevr_count_reward, **sampling), multimodal=True)

    if name == "geometry3k":
        from nano_areal.tasks.vision import (
            VisionQAAgent,
            geometry3k_reward,
            load_geometry3k,
        )

        ds = load_geometry3k(data or "hiyouga/geometry3k", limit=limit)
        return Task("geometry3k", ds, VisionQAAgent(geometry3k_reward, **sampling), multimodal=True)

    if name == "nano_swe":
        from nano_areal.tasks.code import CodeAgent, build_nano_swe

        return Task("nano_swe", build_nano_swe(data, limit=limit), CodeAgent(**sampling))

    if name == "chart":
        from nano_areal.tasks.chart import ChartAgent, load_chartmimic, synthetic_charts

        ds = synthetic_charts(limit or 16) if data in ("", "synthetic") else load_chartmimic(data, limit=limit)
        return Task("chart", ds, ChartAgent(**sampling), multimodal=True)

    if name == "dsh":
        from nano_areal.agent.dsh import DSHAgent
        from nano_areal.tasks.code import build_nano_swe

        return Task("dsh", build_nano_swe(data, limit=limit), DSHAgent(max_turns=r.max_turns, **sampling))

    raise ValueError(f"unknown task {name!r}; choose from {TASKS}")


__all__ = ["ListDataset", "Task", "TASKS", "build_task"]
