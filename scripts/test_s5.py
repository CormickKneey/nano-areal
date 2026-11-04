"""Offline S5 algorithm-switch contracts; no checkpoints, serving or MPS loads."""

from __future__ import annotations

import ast
import asyncio
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch

import train
from nano_areal.archon.losses import (
    LOSS_NAMES,
    compute_policy_loss,
    group_advantages,
    is_offpolicy,
)
from nano_areal.archon.trainer import ArchonTrainer, Trajectory, grpo_loss
from nano_areal.config import NanoArealConfig, TrainConfig
from nano_areal.engine.buffer import AsyncTrajectoryBuffer
from nano_areal.engine.types import InteractionRecord


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_loss_backward_masks_and_frozen_inputs(loss_name):
    policy = torch.tensor([[-1.0, -1.2, -0.8], [-1.1, -0.9, -1.3]],
                          requires_grad=True)
    behavior = policy.detach().clone().requires_grad_()
    reference = (policy.detach() - 0.1).requires_grad_()
    rewards = torch.tensor([2.0, 0.0], requires_grad=True)
    masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    loss, stats = compute_policy_loss(
        loss_name, policy, reference, rewards, masks,
        old_log_probs=behavior, group_size=2,
    )
    assert torch.isfinite(loss)
    assert stats["loss_name"] == loss_name
    assert stats["advantage_scope"] == "group"
    assert all(math.isfinite(value) for value in stats.values()
               if isinstance(value, (int, float)))
    loss.backward()
    assert torch.isfinite(policy.grad).all()
    assert torch.count_nonzero(policy.grad[masks.bool()]) == 3
    assert torch.count_nonzero(policy.grad[~masks.bool()]) == 0
    assert behavior.grad is reference.grad is rewards.grad is None


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_exact_surrogate_and_gradient(loss_name):
    policy = torch.tensor([[math.log(1.5), math.log(0.9)],
                           [math.log(0.5), math.log(1.1)]],
                          dtype=torch.float64, requires_grad=True)
    masks = torch.ones_like(policy)
    advantages = torch.tensor([2.0, -1.0], dtype=torch.float64,
                              requires_grad=True)
    loss, _ = compute_policy_loss(
        loss_name, policy, torch.zeros_like(policy), torch.tensor([2.0, 0.0]),
        masks, kl_coef=0.0, old_log_probs=torch.zeros_like(policy),
        advantages=advantages,
    )
    if loss_name == "cispo":
        expected = (-2 * 1.2 * math.log(1.5) - 2 * 0.9 * math.log(0.9)
                    + 0.8 * math.log(0.5) + 1.1 * math.log(1.1)) / 4
        expected_grad = [[-0.6, -0.45], [0.2, 0.275]]
    elif loss_name == "gspo":
        positive_ratio = math.sqrt(1.5 * 0.9)
        expected = (-2 * positive_ratio + 0.8) / 2
        expected_grad = [[-positive_ratio / 2, -positive_ratio / 2], [0, 0]]
    else:
        expected = (-2 * 1.2 - 2 * 0.9 + 0.8 + 1.1) / 4
        expected_grad = [[0, -0.45], [0, 0.275]]
    torch.testing.assert_close(loss, policy.new_tensor(expected))
    loss.backward()
    torch.testing.assert_close(policy.grad, policy.new_tensor(expected_grad))
    assert advantages.grad is None


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_zero_advantage_retains_kl_and_all_masked_is_zero(loss_name):
    policy = torch.full((2, 3), -1.0, requires_grad=True)
    loss, stats = compute_policy_loss(
        loss_name, policy, torch.zeros_like(policy), torch.ones(2),
        torch.ones_like(policy), kl_coef=0.25, group_size=2,
    )
    torch.testing.assert_close(loss, policy.new_tensor(0.25 * (math.e - 2)))
    assert stats["reward_std"] == 0
    loss.backward()
    assert torch.count_nonzero(policy.grad) == 6
    policy.grad = None
    masked_loss, _ = compute_policy_loss(
        loss_name, policy, torch.zeros_like(policy), torch.tensor([1.0, 0.0]),
        torch.zeros_like(policy), group_size=2,
    )
    assert masked_loss.item() == 0
    masked_loss.backward()
    assert torch.count_nonzero(policy.grad) == 0


def test_default_grpo_wrapper_and_implicit_behavior():
    policy = torch.tensor([[-1.0, -2.0], [-2.0, -1.0]], requires_grad=True)
    args = (policy, torch.zeros_like(policy), torch.tensor([1.0, 0.0]),
            torch.ones_like(policy))
    default, default_stats = grpo_loss(*args)
    explicit, explicit_stats = compute_policy_loss(
        "grpo", *args, old_log_probs=policy.detach(),
    )
    torch.testing.assert_close(default, explicit)
    assert default_stats == explicit_stats
    default_grad = torch.autograd.grad(default, policy)[0]
    explicit_grad = torch.autograd.grad(explicit, policy)[0]
    torch.testing.assert_close(default_grad, explicit_grad)


def test_drgrpo_unscaled_advantages_and_sequence_reduction():
    rewards = torch.tensor([4.0, 0.0])
    masks = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    policy = torch.zeros_like(masks, requires_grad=True)
    scaled, _ = group_advantages(rewards, 2)
    unscaled, _ = group_advantages(rewards, 2, scale=False)
    torch.testing.assert_close(scaled, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(unscaled, torch.tensor([2.0, -2.0]))
    loss, _ = compute_policy_loss(
        "drgrpo", policy, torch.zeros_like(policy), rewards, masks,
        group_size=2, kl_coef=0.0,
    )
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(
        policy.grad, torch.tensor([[-1 / 3, -1 / 3, -1 / 3], [1.0, 0, 0]]),
    )


def test_gspo_geometric_mean_uses_only_response_lengths():
    policy = torch.tensor([[math.log(1.1), math.log(1.1), 5.0],
                           [math.log(0.9), 5.0, 5.0]], requires_grad=True)
    masks = torch.tensor([[1.0, 1.0, 0], [1.0, 0, 0]])
    loss, stats = compute_policy_loss(
        "gspo", policy, torch.zeros_like(policy), torch.tensor([1.0, 0.0]),
        masks, old_log_probs=torch.zeros_like(policy), group_size=2, kl_coef=0,
    )
    torch.testing.assert_close(loss, policy.new_tensor((-2 * 1.1 + 0.9) / 3))
    assert stats["ratio_mean"] == pytest.approx((2 * 1.1 + 0.9) / 3)
    loss.backward()
    torch.testing.assert_close(
        policy.grad, policy.new_tensor([[-1.1 / 3, -1.1 / 3, 0], [0.9 / 3, 0, 0]]),
    )


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("on_incomplete", ["batch", "zeros"])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5])
def test_incomplete_group_contract(count, on_incomplete, scale):
    rewards = torch.arange(count, dtype=torch.float32, requires_grad=True)
    actual, stats = group_advantages(
        rewards, 3, scale=scale, on_incomplete=on_incomplete,
    )
    expected = torch.zeros_like(rewards)

    def normalize(values):
        centered = values - values.mean()
        if not scale:
            return centered
        spread = values.std(unbiased=False)
        return centered / (spread + 1e-8) if spread > 1e-6 else centered * 0

    if count:
        if count % 3 and on_incomplete == "batch":
            expected = normalize(rewards)
        else:
            for start in range(0, count - 2, 3):
                expected[start:start + 3] = normalize(rewards[start:start + 3])
    torch.testing.assert_close(actual, expected)
    assert stats["incomplete_groups"] == int(count > 0 and count % 3 != 0)
    if count:
        assert not actual.requires_grad


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
@pytest.mark.parametrize("on_incomplete", ["batch", "zeros"])
def test_incomplete_loss_dispatch(loss_name, on_incomplete):
    policy = torch.zeros(2, 3, requires_grad=True)
    loss, stats = compute_policy_loss(
        loss_name, policy, torch.zeros_like(policy), torch.tensor([1.0, 0.0]),
        torch.ones_like(policy), group_size=3, on_incomplete=on_incomplete,
        kl_coef=0.0,
    )
    assert torch.isfinite(loss)
    assert stats["incomplete_groups"] == 1
    loss.backward()
    assert torch.isfinite(policy.grad).all()
    assert bool(torch.count_nonzero(policy.grad)) == (on_incomplete == "batch")


def test_explicit_groups_do_not_mix_tasks_and_detach_rewards():
    rewards = torch.tensor([1.0, 100.0, 3.0, 104.0, 999.0], requires_grad=True)
    ids = ["a", "b", "a", "b", "singleton"]
    advantages, stats = group_advantages(rewards, None, group_ids=ids)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, -1.0, 1.0, 1.0, 0]))
    assert stats["advantage_scope"] == "group_ids"
    assert stats["incomplete_groups"] == 1
    assert not advantages.requires_grad
    with pytest.raises(ValueError, match="group_ids"):
        group_advantages(rewards, None, group_ids=ids[:-1])
    flat, stats = group_advantages(torch.ones(4), 2)
    assert torch.count_nonzero(flat) == 0
    assert stats["skipped_groups"] == 2


def test_loss_rejects_invalid_inputs():
    policy = torch.zeros(2, 3, requires_grad=True)
    args = (policy, torch.zeros_like(policy), torch.ones(2), torch.ones_like(policy))
    with pytest.raises(ValueError, match="Unknown loss"):
        compute_policy_loss("ppo", *args)
    with pytest.raises(ValueError, match="at least one trajectory"):
        compute_policy_loss("grpo", policy[:0], policy[:0], torch.empty(0), policy[:0])
    with pytest.raises(ValueError, match="identical shapes"):
        compute_policy_loss("grpo", *args, old_log_probs=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="identical shapes"):
        compute_policy_loss("grpo", policy, torch.zeros(2, 2), args[2], args[3])
    with pytest.raises(ValueError, match="Precomputed advantages"):
        compute_policy_loss("grpo", *args, advantages=torch.zeros(3))
    with pytest.raises(ValueError, match="on_incomplete"):
        group_advantages(torch.arange(4.0), 3, on_incomplete="invalid")


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.3, 0.3, 8))

    def compute_log_probs(self, input_ids, response_mask):
        return self.logits.log_softmax(0)[input_ids[:, 1:]] * response_mask[:, 1:]


@pytest.mark.parametrize("loss_name", LOSS_NAMES)
def test_same_trajectory_trainer_dispatch_truncation_and_update(loss_name):
    trainer = ArchonTrainer.__new__(ArchonTrainer)
    trainer.config = TrainConfig(loss=loss_name, compile=False, gpu_ids=[],
                                 lr=0.01, weight_decay=0.0)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.device = torch.device("cpu")
    trainer.model = TinyPolicy()
    trainer.ref_model = copy.deepcopy(trainer.model).requires_grad_(False)
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=trainer.config.lr, weight_decay=0.0,
    )
    trainer.policy_version = 0
    trainer._step = 0
    trainer._accum_stats = []
    trajectories = [
        Trajectory(
            input_ids=torch.arange(length),
            response_mask=torch.tensor([0] + [1] * (length - 1)),
            reward=reward, truncated=truncated, group_id=group_id,
            old_log_probs=torch.full((length,), -2.0),
        )
        for length, reward, truncated, group_id in (
            (5, 1.0, False, "a"), (4, 1.0, False, "b"),
            (3, 99.0, True, "a"), (4, 0.0, False, "b"),
        )
    ]
    before = trainer.model.logits.detach().clone()
    gradients = []
    trainer.model.logits.register_hook(lambda grad: gradients.append(grad.clone()))
    with patch("nano_areal.archon.losses.compute_policy_loss",
               wraps=compute_policy_loss) as dispatch:
        stats = trainer.train_step(trajectories)
    assert dispatch.call_args.args == (loss_name,)
    forwarded = dispatch.call_args.kwargs
    torch.testing.assert_close(forwarded["rewards"], torch.tensor([1.0, 1.0, 0, 0]))
    torch.testing.assert_close(
        forwarded["masks"], torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0],
                                           [1, 1, 0, 0], [1, 1, 1, 0]]),
    )
    torch.testing.assert_close(
        forwarded["old_log_probs"],
        torch.tensor([[-2, -2, -2, -2], [-2, -2, -2, 0],
                      [-2, -2, 0, 0], [-2, -2, -2, 0]], dtype=torch.float32),
    )
    assert forwarded["group_ids"] == ["a", "b", "a", "b"]
    assert stats["truncated"] == 1
    assert stats["loss_name"] == loss_name
    assert stats["policy_version"] == 1
    assert math.isfinite(stats["loss"])
    assert len(gradients) == 1 and torch.isfinite(gradients[0]).all()
    assert torch.count_nonzero(gradients[0]) > 0
    assert not torch.equal(before, trainer.model.logits)
    assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())


@pytest.mark.parametrize("finish_reason,truncated", [("length", True), ("stop", False)])
def test_captured_truncation_contract(finish_reason, truncated):
    trajectory = InteractionRecord(
        response_id="s5", input_ids=[1, 2, 3], response_mask=[0, 1, 1],
        token_logprobs=[-0.1, -0.2], reward=1.0, finish_reason=finish_reason,
    ).to_trajectory()
    assert trajectory.truncated is truncated
    assert trajectory.reward == (0.0 if truncated else 1.0)


@pytest.mark.parametrize("maximum", [0, 1, 4])
@pytest.mark.parametrize("gap", [-1, 0, 1, 4, 5])
def test_offpolicy_boundaries(maximum, gap):
    assert is_offpolicy(gap, maximum) is (gap < 0 or gap > maximum)


@pytest.mark.parametrize("maximum", [0, 2])
@pytest.mark.parametrize("sync", [False, True])
def test_sync_and_async_share_filter_at_consumption(maximum, sync):
    async def run():
        buffer = AsyncTrajectoryBuffer(max_head_offpolicyness=maximum, maxsize=8)
        trajectories = [
            Trajectory(torch.tensor([1, 2]), torch.tensor([0, 1]), 1.0,
                       policy_version=version)
            for version in [0, 1, 2, 3, 4]
        ]
        for trajectory in trajectories:
            await buffer.put(trajectory)
        buffer.update_policy_version(3)
        expected = [item for item in trajectories
                    if not is_offpolicy(3 - item.policy_version, maximum)]
        with patch("nano_areal.engine.buffer.is_offpolicy",
                   wraps=is_offpolicy) as predicate:
            if sync:
                batch = await asyncio.wait_for(buffer.get_batch_sync(len(expected)), 1)
            else:
                batch = await buffer.get_batch(len(expected), timeout=1)
            assert [id(item) for item in batch] == [id(item) for item in expected]
            assert predicate.call_count == 4
            assert await buffer.get_batch(1, timeout=0.01) == []
            assert predicate.call_args.args == (-1, maximum)
        assert buffer.qsize == 0
        assert buffer.stats.total_produced == 5
        assert buffer.stats.total_consumed == len(expected)
        assert buffer.stats.total_discarded == 5 - len(expected)

    asyncio.run(run())


@pytest.mark.parametrize("arguments", [["--sync"], ["--backend", "metal"]])
@pytest.mark.parametrize("loss_name", [None, *LOSS_NAMES])
def test_metal_cli_and_config_default_grpo(arguments, loss_name):
    assert TrainConfig().loss == NanoArealConfig.for_macos().train.loss == "grpo"
    options = arguments + ([] if loss_name is None else ["--loss", loss_name])
    with patch("sys.argv", ["train.py", *options]), \
         patch("train.main_async", new_callable=AsyncMock) as run:
        train.main()
    config = run.call_args.args[0]
    assert config.backend == config.rollout.inference_backend == "metal"
    assert config.train.loss == (loss_name or "grpo")
    assert config.async_rl.max_head_offpolicyness == 0
    assert not config.train.compile


def test_cli_rejects_unknown_loss():
    with patch("sys.argv", ["train.py", "--backend", "metal", "--loss", "ppo"]), \
         pytest.raises(SystemExit) as failure:
        train.parse_args()
    assert failure.value.code == 2


def test_native_metal_runner_keeps_grpo_only_without_importing_mlx():
    source = Path(__file__).resolve().with_name("run_agentic_rl.py")
    syntax = ast.parse(source.read_text())
    declarations = [
        node for node in ast.walk(syntax)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--loss"
    ]
    assert len(declarations) == 1
    options = {item.arg: ast.literal_eval(item.value)
               for item in declarations[0].keywords}
    assert options["default"] == "grpo"
    assert options["choices"] == ["grpo"]
