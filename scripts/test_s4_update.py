"""Small CPU tests for episode-balanced bounded updates, without model loading."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from nano_areal.archon.s4_update import bounded_step, trajectory_from_turn


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.zeros(2))
        self.enabled = True

    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def tiny_trainer():
    model = TinyPolicy()

    def logprobs(trajectory):
        logits = model.adapter if model.enabled else torch.zeros(2)
        return logits.log_softmax(-1)[trajectory.input_ids[-1:]].unsqueeze(0)

    return SimpleNamespace(
        model=model, parameters={"adapter": model.adapter},
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        response_logprobs=logprobs,
    )


def episodes(rewards):
    return [{"reward": reward, "turns": [{
        "input_ids": [0, index], "response_mask": [0, 1],
        "old_log_probs": [0, -0.6931471805599453],
        "discounted_reward": 0.123, "truncated": False,
    }]} for index, reward in enumerate(rewards)]


def test_single_step_uses_external_episode_advantages():
    trainer = tiny_trainer()
    stages = []
    report = bounded_step(trainer, episodes([0.0, 1.0]), stages.append)
    assert report["episode_advantages"] == [-1.0, 1.0]
    assert report["optimizer_steps"] == 1
    assert report["adapter_delta_l1"] > 0
    assert trainer.model.adapter[1] > trainer.model.adapter[0]
    assert stages == ["forward", "backward", "forward", "backward", "optimizer"]


def test_homogeneous_group_never_steps():
    trainer = tiny_trainer()
    with pytest.raises(ValueError, match="Homogeneous"):
        bounded_step(trainer, episodes([1.0, 1.0]), lambda stage: None)
    assert trainer.optimizer.state == {}
    assert trainer.model.adapter.tolist() == [0.0, 0.0]


def test_episode_weighting_does_not_depend_on_turn_count():
    trainer = tiny_trainer()
    baseline = bounded_step(trainer, episodes([0.0, 1.0]), lambda stage: None)
    repeated = episodes([0.0, 1.0])
    repeated[0]["turns"] *= 4
    trainer = tiny_trainer()
    report = bounded_step(trainer, repeated, lambda stage: None)
    assert report["grad_norm"] == pytest.approx(baseline["grad_norm"])
    assert report["episode_advantages"] == [-1.0, 1.0]


def test_wall_clock_limit_never_steps():
    trainer = tiny_trainer()
    with pytest.raises(TimeoutError, match="wall-clock"):
        bounded_step(trainer, episodes([0.0, 1.0]), lambda stage: None, max_seconds=-1)
    assert trainer.optimizer.state == {}


@pytest.mark.parametrize("field,value", [
    ("truncated", True), ("old_log_probs", [0]),
    ("response_mask", [1, 1]), ("response_mask", [0, 0]),
    ("response_mask", [0, 0.5]),
    ("old_log_probs", [0, float("nan")]),
])
def test_rejects_invalid_training_turn(field, value):
    turn = episodes([0.0, 1.0])[0]["turns"][0]
    with pytest.raises(ValueError, match="Malformed"):
        trajectory_from_turn({**turn, field: value})
