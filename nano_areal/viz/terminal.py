"""
Rich-based real-time terminal visualization.

Displays:
  - Current trajectory (multi-turn conversation with tool calls + rewards)
  - Live training stats (loss, KL, reward, off-policyness, buffer fullness)
"""
from __future__ import annotations

import json
import time
from collections import deque
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from nano_areal.agent.base import CompletionWithReward
    from nano_areal.engine.buffer import AsyncTrajectoryBuffer


console = Console()


def _reward_badge(reward: float) -> Text:
    if reward >= 0.9:
        return Text(f"reward={reward:.2f} ✓", style="bold green")
    if reward >= 0.5:
        return Text(f"reward={reward:.2f} ~", style="yellow")
    return Text(f"reward={reward:.2f} ✗", style="red")


def render_trajectory(
    sample_id: str,
    completions: list["CompletionWithReward"],
) -> Panel:
    """Render one multi-turn trajectory as a Rich panel."""
    content = Text()

    for comp in completions:
        content.append(f"\n  ── Turn {comp.turn_index + 1} ", style="bold cyan")
        content.append(_reward_badge(comp.reward))
        content.append("\n")

        # Find the last assistant message in this completion's messages
        messages = comp.messages
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                text = msg.get("content") or ""
                tool_calls = msg.get("tool_calls", [])

                if text:
                    # Truncate long thoughts
                    display = text[:300] + ("…" if len(text) > 300 else "")
                    content.append(f"  [model] {display}\n", style="white")

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        args_str = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args_str = fn.get("arguments", "")
                    args_display = args_str[:120] + ("…" if len(args_str) > 120 else "")
                    content.append(
                        f"  [tool]  {fn.get('name', '?')}({args_display})\n",
                        style="cyan",
                    )
                break

        # Tool response (next tool message)
        for msg in messages:
            if msg.get("role") == "tool":
                result = msg.get("content", "")[:100]
                content.append(f"  [resp]  {result}\n", style="dim")
                break

    total_reward = sum(c.reward for c in completions)
    title = f"[bold]{sample_id}[/bold]  total_reward={total_reward:.3f}"
    return Panel(content, title=title, border_style="blue")


class TrainingDashboard:
    """
    Live terminal dashboard updated each training step.

    Usage:
        dash = TrainingDashboard()
        with dash:
            # in training loop:
            dash.update_stats(stats)
            dash.update_trajectory(sample_id, completions)
    """

    def __init__(self, history_len: int = 200):
        self._stats_history: deque[dict] = deque(maxlen=history_len)
        self._latest_trajectory: Panel | None = None
        self._live: Live | None = None
        self._start_time = time.time()

    def __enter__(self):
        layout = Layout()
        layout.split_column(
            Layout(name="stats", size=12),
            Layout(name="traj"),
        )
        self._layout = layout
        self._live = Live(layout, console=console, refresh_per_second=2)
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)

    def update_stats(self, stats: dict, buffer: "AsyncTrajectoryBuffer | None" = None):
        self._stats_history.append(stats)
        if self._layout and self._live:
            self._layout["stats"].update(self._render_stats(stats, buffer))
            self._live.refresh()

    def update_trajectory(
        self,
        sample_id: str,
        completions: list["CompletionWithReward"],
    ):
        self._latest_trajectory = render_trajectory(sample_id, completions)
        if self._layout and self._live:
            self._layout["traj"].update(self._latest_trajectory)
            self._live.refresh()

    def _render_stats(
        self,
        stats: dict,
        buffer: "AsyncTrajectoryBuffer | None",
    ) -> Panel:
        elapsed = time.time() - self._start_time

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", justify="right")
        table.add_column()
        table.add_column(style="bold cyan", justify="right")
        table.add_column()
        table.add_column(style="bold cyan", justify="right")
        table.add_column()

        step = stats.get("step", 0)
        pv = stats.get("policy_version", 0)
        loss = stats.get("loss", 0.0)
        kl = stats.get("kl", 0.0)
        reward_mean = stats.get("reward_mean", 0.0)
        reward_std = stats.get("reward_std", 0.0)
        ratio = stats.get("ratio_mean", 1.0)

        buf_info = "—"
        if buffer is not None:
            buf_info = f"{buffer.qsize}/{buffer.maxsize}"

        table.add_row(
            "step", str(step),
            "policy_v", str(pv),
            "elapsed", f"{elapsed:.0f}s",
        )
        table.add_row(
            "loss", f"{loss:.4f}",
            "KL", f"{kl:.4f}",
            "ratio", f"{ratio:.3f}",
        )
        table.add_row(
            "reward", f"{reward_mean:.3f} ± {reward_std:.3f}",
            "buffer", buf_info,
            "discard", str(buffer.stats.total_discarded) if buffer else "—",
        )

        return Panel(table, title="[bold]nano-areal training[/bold]", border_style="green")

    def log(self, msg: str, style: str = ""):
        console.print(msg, style=style)
