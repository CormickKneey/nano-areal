"""Check S1 wiring without launching a model server or downloading weights."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import torch

import train
from nano_areal.agent.dsh import build_dsh_kwargs
from nano_areal.archon.losses import compute_policy_loss
from nano_areal.archon.model import Qwen35Args, build_model
from nano_areal.archon.trainer import ArchonTrainer, Trajectory
from nano_areal.config import TrainConfig
from nano_areal.engine.types import InteractionRecord


class TrainingContractTests(unittest.TestCase):
    def test_dsh_uses_isolated_child_home(self):
        kwargs = build_dsh_kwargs(
            base_url="http://127.0.0.1:8001/v1",
            api_key="local-test-only",
            cwd=".",
            dsh_home="artifacts/s1/example-home",
        )
        self.assertEqual(kwargs["env"]["HOME"], kwargs["dsh_home"])
        self.assertEqual(kwargs["env"]["XDG_CACHE_HOME"], kwargs["dsh_home"] + "/cache")

    def test_sync_cli_preserves_zero_gap(self):
        with patch("sys.argv", ["train.py", "--sync"]):
            with patch("train.main_async", new_callable=AsyncMock) as run:
                train.main()
        config = run.call_args.args[0]
        self.assertEqual(config.async_rl.max_head_offpolicyness, 0)
        self.assertFalse(config.train.compile)

    def test_async_cli_preserves_default_gap(self):
        with patch("sys.argv", ["train.py"]):
            with patch("train.main_async", new_callable=AsyncMock) as run:
                train.main()
        self.assertEqual(run.call_args.args[0].async_rl.max_head_offpolicyness, 4)

    def test_sync_rejects_nonzero_gap(self):
        with patch("sys.argv", ["train.py", "--sync", "--max-offpolicy", "1"]):
            with self.assertRaises(SystemExit):
                train.main()

    def test_incomplete_logprob_capture_rejected(self):
        record = InteractionRecord(
            response_id="incomplete", input_ids=[1, 2], response_mask=[0, 1]
        )
        with self.assertRaises(ValueError):
            record.to_trajectory()

    def test_trainer_pads_and_shifts_behavior_logprobs(self):
        args = Qwen35Args(
            dim=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            vocab_size=32,
            intermediate_size=64,
            max_position_embeddings=32,
            head_dim=16,
        )
        model = build_model(args, torch.device("cpu"), torch.float32)
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        trainer = ArchonTrainer(
            model, TrainConfig(compile=False, gpu_ids=[]), tokenizer
        )
        trajectories = []
        for length, reward in ((6, 0.0), (4, 1.0)):
            trajectories.append(
                Trajectory(
                    input_ids=torch.arange(length),
                    response_mask=torch.ones(length),
                    reward=reward,
                    group_id="problem",
                    old_log_probs=torch.full((length,), -2.0),
                )
            )
        with patch(
            "nano_areal.archon.losses.compute_policy_loss", wraps=compute_policy_loss
        ) as loss:
            stats = trainer.train_step(trajectories)
        behavior = loss.call_args.kwargs["old_log_probs"].cpu()
        torch.testing.assert_close(
            behavior,
            torch.tensor(
                [[-2.0, -2.0, -2.0, -2.0, -2.0], [-2.0, -2.0, -2.0, 0.0, 0.0]]
            ),
        )
        self.assertEqual(loss.call_args.kwargs["group_ids"], ["problem", "problem"])
        self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])))


if __name__ == "__main__":
    unittest.main()
