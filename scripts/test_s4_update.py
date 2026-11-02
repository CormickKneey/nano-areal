"""Small CPU tests for episode-balanced bounded updates, without model loading."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import probe_s4_training
import pytest
import torch
from probe_s4_training import phase_paths

from nano_areal.archon.s4_update import bounded_step, trajectory_from_turn


@pytest.mark.parametrize("version", [0, 1, 2, 14])
def test_phase_paths_support_sequential_versions(tmp_path, version):
    suffix = f"-v{version}" if version else ""
    assert phase_paths(tmp_path, "update", version) == (
        tmp_path / f"update{suffix}.json")
    assert phase_paths(tmp_path, "preflight", version) == (
        tmp_path / f"preflight{suffix}.json")


@pytest.mark.parametrize("destination", ["update-v1.json", "adapter-v2",
                                          "policy-v2.json"])
def test_update_rejects_existing_child_evidence(tmp_path, destination):
    (tmp_path / destination).touch()
    with pytest.raises(FileExistsError, match="Immutable"):
        phase_paths(tmp_path, "update", 1)


def test_update_rejects_dangling_child_symlink(tmp_path):
    (tmp_path / "adapter-v2").symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="Immutable"):
        phase_paths(tmp_path, "update", 1)


@pytest.mark.parametrize("phase,version", [("prepare", 1), ("update", -1),
                                           ("preflight", -1)])
def test_phase_paths_reject_invalid_versions(phase, version):
    with pytest.raises(ValueError, match="version"):
        phase_paths(Path("unused"), phase, version)


@pytest.mark.parametrize("phase", ["prepare", "preflight"])
def test_phase_paths_never_overwrite_reports(tmp_path, phase):
    (tmp_path / f"{phase}.json").touch()
    with pytest.raises(FileExistsError, match="Immutable"):
        phase_paths(tmp_path, phase, 0)


@pytest.mark.parametrize("bound_version,message", [
    (0, "version differs"), (1, "adapter differs"),
])
def test_update_rejects_wrong_parent_before_model_load(
        tmp_path, monkeypatch, bound_version, message):
    monkeypatch.setattr("sys.argv", [
        "probe_s4_training.py", "update", "--policy-version", "1",
        "--infer-model", "unused", "--train-model", "unused",
        "--group", "unused", "--output", str(tmp_path),
    ])
    monkeypatch.setattr(probe_s4_training, "require_stopped_server", lambda: None)
    monkeypatch.setattr(probe_s4_training, "memory_snapshot",
                        lambda process: {"swap": "used = 0.00M"})
    monkeypatch.setattr(probe_s4_training, "command_output", lambda command: "")
    monkeypatch.setattr(probe_s4_training.PolicyBinding, "read", lambda path:
                        SimpleNamespace(version=bound_version,
                                        verify_files=lambda: None,
                                        manifest={"adapter": str(tmp_path / "other")}))

    def reject_model_load(*args):
        pytest.fail("Invalid parent must be rejected before loading a model")

    monkeypatch.setattr(probe_s4_training, "NativeQwen3VLLoRATrainer",
                        reject_model_load)
    with pytest.raises(ValueError, match=message):
        probe_s4_training.main()
    assert not (tmp_path / "adapter-v2").exists()
    assert not (tmp_path / "policy-v2.json").exists()


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


def test_whole_group_backward_matches_independent_turn_accumulation():
    baseline = tiny_trainer()
    candidate = tiny_trainer()
    calls = []

    def group_logprobs(trajectories):
        calls.append((candidate.model.enabled, torch.is_grad_enabled()))
        return [candidate.response_logprobs(trajectory) for trajectory in trajectories]

    candidate.response_group_logprobs = group_logprobs
    group = episodes([0.0, 1.0])
    group[0]["turns"] *= 3
    expected = bounded_step(baseline, group, lambda stage: None)
    stages = []
    actual = bounded_step(candidate, group, stages.append)
    torch.testing.assert_close(candidate.model.adapter, baseline.model.adapter)
    assert actual["loss"] == pytest.approx(expected["loss"])
    assert actual["grad_norm"] == pytest.approx(expected["grad_norm"])
    assert calls == [(False, False), (True, True)]
    assert stages == ["forward"] * 4 + ["backward", "optimizer"]


def test_wall_clock_limit_never_steps():
    trainer = tiny_trainer()
    with pytest.raises(TimeoutError, match="wall-clock"):
        bounded_step(trainer, episodes([0.0, 1.0]), lambda stage: None, max_seconds=-1)
    assert trainer.optimizer.state == {}


def test_group_backward_deadline_never_steps(monkeypatch):
    trainer = tiny_trainer()
    elapsed = 0
    monkeypatch.setattr("nano_areal.archon.s4_update.time.monotonic", lambda: elapsed)
    trainer.response_group_logprobs = lambda trajectories: [
        trainer.response_logprobs(trajectory) for trajectory in trajectories]

    def snapshot(stage):
        nonlocal elapsed
        if stage == "backward":
            elapsed = 601

    with pytest.raises(TimeoutError, match="wall-clock"):
        bounded_step(trainer, episodes([0.0, 1.0]), snapshot)
    assert trainer.optimizer.state == {}
    assert trainer.model.adapter.tolist() == [0.0, 0.0]


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
