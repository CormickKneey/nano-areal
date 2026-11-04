"""Deterministic GRPO contract checks; no downloaded model weights required."""

from __future__ import annotations

import math
import unittest

import torch

from nano_areal.archon.trainer import grpo_loss


class GRPOParityTests(unittest.TestCase):
    def test_singleton_and_constant_groups(self):
        for rewards in ([1.0], [2.0, 2.0]):
            policy = torch.full((len(rewards), 2), -1.0, requires_grad=True)
            loss, stats = grpo_loss(
                policy, policy.detach(), torch.tensor(rewards), torch.ones_like(policy)
            )
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats["reward_std"], 0)
            self.assertTrue(torch.isfinite(policy.grad).all())

    def test_behavior_ratio_and_clip(self):
        policy = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
        behavior = policy.detach() - torch.tensor([[math.log(2)], [math.log(0.5)]])
        loss, _ = grpo_loss(
            policy,
            policy.detach(),
            torch.tensor([0.0, 1.0]),
            torch.ones_like(policy),
            old_log_probs=behavior,
            kl_coef=0,
        )
        self.assertAlmostEqual(loss.item(), 0.75)
        loss.backward()
        torch.testing.assert_close(policy.grad, torch.tensor([[1.0], [-0.25]]))

    def test_reference_is_not_behavior_policy(self):
        policy = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
        _, stats = grpo_loss(
            policy,
            policy.detach() - 0.5,
            torch.tensor([0.0, 1.0]),
            torch.ones_like(policy),
            kl_coef=0,
        )
        self.assertAlmostEqual(stats["ratio_mean"], 1)
        self.assertGreaterEqual(stats["kl"], 0)

    def test_group_normalization_is_independent(self):
        policy = torch.full((4, 1), -1.0, requires_grad=True)
        behavior = policy.detach() - torch.tensor([[1.1], [0.9], [1.1], [0.9]]).log()
        loss, _ = grpo_loss(
            policy,
            policy.detach(),
            torch.tensor([0.0, 2.0, 100.0, 102.0]),
            torch.ones_like(policy),
            kl_coef=0,
            old_log_probs=behavior,
            group_ids=["first", "first", "second", "second"],
        )
        self.assertAlmostEqual(loss.item(), 0.1, places=6)

    def test_incomplete_group_and_padding(self):
        policy = torch.full((3, 2), -1.0, requires_grad=True)
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        loss, _ = grpo_loss(
            policy,
            policy.detach(),
            torch.tensor([0.0, 1.0, 99.0]),
            mask,
            group_ids=["pair", "pair", "singleton"],
        )
        loss.backward()
        self.assertTrue(torch.isfinite(policy.grad).all())
        torch.testing.assert_close(policy.grad[:, 1], torch.zeros(3))
        torch.testing.assert_close(policy.grad[2], torch.zeros(2))

    def test_k3_kl_value(self):
        policy = torch.full((2, 1), -1.0, requires_grad=True)
        loss, stats = grpo_loss(
            policy,
            policy.detach() - 0.5,
            torch.ones(2),
            torch.ones_like(policy),
            kl_coef=0.2,
        )
        expected = math.exp(-0.5) + 0.5 - 1
        self.assertAlmostEqual(stats["kl"], expected, places=6)
        self.assertAlmostEqual(loss.item(), 0.2 * expected, places=6)

    def test_group_length_must_match(self):
        policy = torch.zeros(2, 1)
        with self.assertRaises(ValueError):
            grpo_loss(
                policy,
                policy,
                torch.ones(2),
                torch.ones_like(policy),
                group_ids=["one"],
            )


if __name__ == "__main__":
    unittest.main()
