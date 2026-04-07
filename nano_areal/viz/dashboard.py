"""
Plotly HTML training dashboard.

Saves an interactive HTML file after each eval or on demand.
Charts:
  - Reward mean ± std over training steps
  - KL divergence over steps
  - Off-policyness distribution (version gap histogram)
  - Per-turn success rate (turn 1, 2, 3 …)
  - Trajectory replay: select a step to view full conversation
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nano_areal.agent.base import CompletionWithReward


@dataclass
class StepRecord:
    step: int
    policy_version: int
    loss: float
    kl: float
    reward_mean: float
    reward_std: float
    ratio_mean: float
    # Per-turn success rates
    turn_success: list[float] = field(default_factory=list)
    # Version gaps of trajectories in this batch
    version_gaps: list[int] = field(default_factory=list)
    # Sampled trajectory for replay
    sample_id: str = ""
    conversation: list[dict] = field(default_factory=list)


class HTMLDashboard:
    """
    Accumulates per-step records and writes an interactive Plotly HTML dashboard.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records: list[StepRecord] = []
        self._jsonl_path = os.path.join(log_dir, "training_log.jsonl")

    def record(
        self,
        stats: dict,
        completions: list["CompletionWithReward"] | None = None,
        trajectories=None,
        sample_id: str = "",
    ):
        """Add one step's data. Call this every train step."""
        version_gaps = []
        turn_success: dict[int, list[float]] = defaultdict(list)

        if completions:
            for comp in completions:
                turn_success[comp.turn_index].append(float(comp.reward >= 0.9))
            if trajectories:
                policy_v = stats.get("policy_version", 0)
                for t in trajectories:
                    version_gaps.append(policy_v - t.policy_version)

        per_turn = [
            sum(v) / len(v) if v else 0.0
            for v in [turn_success.get(i, []) for i in range(5)]
        ]

        # Build conversation for replay
        conversation = []
        if completions:
            last = completions[-1]
            conversation = last.messages

        rec = StepRecord(
            step=stats.get("step", len(self.records)),
            policy_version=stats.get("policy_version", 0),
            loss=stats.get("loss", 0.0),
            kl=stats.get("kl", 0.0),
            reward_mean=stats.get("reward_mean", 0.0),
            reward_std=stats.get("reward_std", 0.0),
            ratio_mean=stats.get("ratio_mean", 1.0),
            turn_success=per_turn,
            version_gaps=version_gaps,
            sample_id=sample_id,
            conversation=conversation,
        )
        self.records.append(rec)

        # Append to JSONL for persistence
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def save(self, path: str | None = None):
        """Render and save the HTML dashboard."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("pip install plotly")

        if not self.records:
            return

        steps = [r.step for r in self.records]
        reward_mean = [r.reward_mean for r in self.records]
        reward_std = [r.reward_std for r in self.records]
        kl = [r.kl for r in self.records]
        loss = [r.loss for r in self.records]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Reward", "KL Divergence", "Loss", "Per-Turn Success Rate"),
            vertical_spacing=0.15,
        )

        # Reward with confidence band
        fig.add_trace(go.Scatter(
            x=steps + steps[::-1],
            y=[m + s for m, s in zip(reward_mean, reward_std)]
              + [m - s for m, s in zip(reward_mean[::-1], reward_std[::-1])],
            fill="toself", fillcolor="rgba(0,100,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name="reward_band",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=steps, y=reward_mean,
            mode="lines", name="reward",
            line=dict(color="royalblue", width=2),
        ), row=1, col=1)

        # KL
        fig.add_trace(go.Scatter(
            x=steps, y=kl, mode="lines", name="KL",
            line=dict(color="orange", width=2),
        ), row=1, col=2)

        # Loss
        fig.add_trace(go.Scatter(
            x=steps, y=loss, mode="lines", name="loss",
            line=dict(color="red", width=2),
        ), row=2, col=1)

        # Per-turn success rates
        colors = ["green", "blue", "purple", "brown", "gray"]
        for turn_idx in range(min(5, len(self.records[0].turn_success))):
            rates = [r.turn_success[turn_idx] for r in self.records]
            fig.add_trace(go.Scatter(
                x=steps, y=rates,
                mode="lines", name=f"turn_{turn_idx + 1}",
                line=dict(color=colors[turn_idx], width=1.5),
            ), row=2, col=2)

        fig.update_layout(
            title="nano-areal Training Dashboard",
            height=700,
            template="plotly_dark",
        )

        out_path = path or os.path.join(self.log_dir, f"dashboard_step{steps[-1]}.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        return out_path

    def save_trajectory_replay(self, path: str | None = None):
        """
        Save a separate HTML file with interactive trajectory replay.
        Each step's conversation can be selected via a dropdown.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        if not self.records:
            return

        # Build dropdown steps with conversation HTML
        buttons = []
        all_annotations = []

        for i, rec in enumerate(self.records):
            conv_html = _render_conversation_html(rec.conversation)
            buttons.append(dict(
                label=f"Step {rec.step} | r={rec.reward_mean:.2f}",
                method="update",
                args=[{}, {"title": f"Step {rec.step} — {rec.sample_id}<br>{conv_html}"}],
            ))

        fig = go.Figure()
        fig.add_annotation(
            text=_render_conversation_html(self.records[-1].conversation),
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, align="left",
            font=dict(size=13, family="monospace"),
        )
        fig.update_layout(
            updatemenus=[dict(buttons=buttons, direction="down", x=0.1, y=1.1)],
            title=f"Trajectory Replay — Step {self.records[-1].step}",
            height=800,
            template="plotly_dark",
        )

        out_path = path or os.path.join(self.log_dir, "trajectory_replay.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        return out_path


def _render_conversation_html(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = str(msg.get("content") or "")[:200]
        tool_calls = msg.get("tool_calls", [])
        color = {"user": "#88c0d0", "assistant": "#a3be8c", "tool": "#d08770"}.get(role, "#eceff4")
        lines.append(f'<span style="color:{color}">[{role}]</span> {content}')
        for tc in tool_calls:
            fn = tc.get("function", {})
            lines.append(f'<span style="color:#b48ead">  → {fn.get("name","?")}({fn.get("arguments","")[:80]})</span>')
    return "<br>".join(lines)
