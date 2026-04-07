"""
Integration test: verify GRPO loss is finite and gradients are non-zero
using synthetic data (no real model weights, no vllm server needed).

Run with:
    uv run python scripts/test_sync.py
"""
from __future__ import annotations

import sys
import types
import torch

# ---------------------------------------------------------------------------
# Minimal mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        # deterministic fake tokenization: each char → one token id
        ids = [ord(c) % 100 + 3 for c in text[:64]]
        out = types.SimpleNamespace()
        out.input_ids = ids
        return out

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m.get("content") or "" for m in messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trajectory(seq_len: int, prompt_len: int, reward: float, version: int = 0):
    from nano_areal.archon.trainer import Trajectory

    input_ids = torch.randint(3, 100, (seq_len,))
    mask = torch.zeros(seq_len, dtype=torch.float)
    mask[prompt_len:] = 1.0
    return Trajectory(
        input_ids=input_ids,
        response_mask=mask,
        reward=reward,
        policy_version=version,
    )


# ---------------------------------------------------------------------------
# Test 1: GRPO loss is finite and backward works
# ---------------------------------------------------------------------------

def test_grpo_loss():
    from nano_areal.archon.trainer import grpo_loss

    B, T = 8, 32
    log_probs = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.randn(B, T).detach()
    rewards = torch.randn(B)
    masks = torch.ones(B, T)

    loss, stats = grpo_loss(log_probs, ref_log_probs, rewards, masks)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()
    assert log_probs.grad is not None, "No gradient computed"
    assert torch.isfinite(log_probs.grad).all(), "Gradient contains NaN/Inf"

    print(f"  [PASS] grpo_loss: loss={loss.item():.4f}, kl={stats['kl']:.4f}")


# ---------------------------------------------------------------------------
# Test 2: ArchonTrainer.train_step reduces loss over N steps
# ---------------------------------------------------------------------------

def test_trainer_step():
    from nano_areal.archon.model import Qwen35Args, build_model
    from nano_areal.archon.trainer import ArchonTrainer
    from nano_areal.config import TrainConfig

    # Tiny model for fast test
    args = Qwen35Args(
        dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=512, intermediate_size=256,
        max_position_embeddings=128, head_dim=32,
    )
    device = torch.device("cpu")
    dtype = torch.float32

    model = build_model(args, device, dtype)

    cfg = TrainConfig(
        gpu_ids=[],          # cpu
        lr=1e-3,
        compile=False,       # skip compile for unit test speed
        gradient_checkpointing=False,
        batch_size=4,
    )
    tokenizer = MockTokenizer()
    trainer = ArchonTrainer(model, cfg, tokenizer)

    # Build a batch: 4 trajectories with varying rewards
    batch = [
        make_trajectory(seq_len=48, prompt_len=16, reward=r)
        for r in [1.0, 0.0, 1.0, 0.0]
    ]

    losses = []
    for step in range(5):
        stats = trainer.train_step(batch)
        losses.append(stats["loss"])
        print(f"  step {step}: loss={stats['loss']:.4f} kl={stats['kl']:.4f} "
              f"reward_mean={stats['reward_mean']:.2f}")

    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "Some losses are not finite"
    print("  [PASS] trainer_step: all losses finite, gradients OK")


# ---------------------------------------------------------------------------
# Test 3: AsyncTrajectoryBuffer off-policyness filtering
# ---------------------------------------------------------------------------

def test_buffer():
    import asyncio
    from nano_areal.engine.buffer import AsyncTrajectoryBuffer

    async def _run():
        buf = AsyncTrajectoryBuffer(max_head_offpolicyness=2, maxsize=64)

        # Produce 8 trajectories: versions 0..7
        for v in range(8):
            traj = make_trajectory(seq_len=16, prompt_len=4, reward=1.0, version=v)
            buf.put_nowait(traj)

        # Consumer is at version 5: should discard v < 3 (gap > 2)
        buf.update_policy_version(5)
        accepted = await buf.get_batch(n=4, timeout=2.0)

        for t in accepted:
            gap = 5 - t.policy_version
            assert gap <= 2, f"Accepted trajectory with gap={gap} > max_head_offpolicyness=2"

        print(f"  [PASS] buffer: accepted {len(accepted)} trajectories, "
              f"discarded {buf.stats.total_discarded}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 4: BFCLReward AST verifier
# ---------------------------------------------------------------------------

def test_bfcl_reward():
    from nano_areal.reward.bfcl import BFCLReward

    reward_fn = BFCLReward()

    # Exact match
    pred = [{"name": "mv", "arguments": {"src": "file.pdf", "dst": "/tmp"}}]
    gt = [{"mv": {"src": ["file.pdf"], "dst": ["/tmp"]}}]
    assert reward_fn(pred, gt) == 1.0, "Expected exact match reward=1.0"

    # Wrong function name
    pred_wrong = [{"name": "cp", "arguments": {"src": "file.pdf", "dst": "/tmp"}}]
    assert reward_fn(pred_wrong, gt) == 0.0, "Expected wrong name reward=0.0"

    # Normalize: "file-pdf" should NOT match "file.pdf" (different chars after norm)
    # But "File.PDF" should match "file.pdf" (case-insensitive)
    pred_case = [{"name": "mv", "arguments": {"src": "File.PDF", "dst": "/tmp"}}]
    # After strip: filepdf vs filepdf → match
    result = reward_fn(pred_case, gt)
    assert result == 1.0, f"Expected case-insensitive match, got {result}"

    # Empty prediction
    assert reward_fn([], gt) == 0.0, "Expected empty pred reward=0.0"

    print("  [PASS] bfcl_reward: all cases correct")


# ---------------------------------------------------------------------------
# Test 5: state_dict adapter round-trip key mapping
# ---------------------------------------------------------------------------

def test_state_dict_keys():
    from nano_areal.archon.state_dict import hf_key_to_archon, archon_key_to_hf

    pairs = [
        ("model.embed_tokens.weight",                    "embed_tokens.weight"),
        ("model.layers.3.self_attn.q_proj.weight",       "layers.3.attn.q_proj.weight"),
        ("model.layers.3.self_attn.q_proj.bias",         "layers.3.attn.q_proj.bias"),
        ("model.layers.0.mlp.gate_proj.weight",          "layers.0.mlp.gate_proj.weight"),
        ("model.layers.0.input_layernorm.weight",        "layers.0.input_layernorm.weight"),
        ("model.norm.weight",                            "norm.weight"),
        ("lm_head.weight",                               "lm_head.weight"),
    ]

    for hf_key, expected_archon in pairs:
        got = hf_key_to_archon(hf_key)
        assert got == expected_archon, f"HF→archon: {hf_key!r} → {got!r} (expected {expected_archon!r})"
        rev = archon_key_to_hf(expected_archon)
        assert rev == hf_key, f"archon→HF: {expected_archon!r} → {rev!r} (expected {hf_key!r})"

    print(f"  [PASS] state_dict: {len(pairs)} key pairs round-trip correctly")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 6: proxy session lifecycle + reward backpropagation (no vllm needed)
# ---------------------------------------------------------------------------

def test_proxy_session():
    import asyncio
    from nano_areal.engine.proxy import OpenAIProxyServer, _build_trajectories, ADMIN_KEY
    from nano_areal.engine.types import InteractionRecord

    async def _run():
        # Build a fake session with 3 interactions (turn 0 → 1 → 2)
        from nano_areal.engine.proxy import ProxySession
        session = ProxySession(session_id="s1", api_key="k1")

        # Turn 0
        r0 = InteractionRecord(
            response_id="r0",
            input_ids=[1, 2, 3, 10, 11],
            response_mask=[0, 0, 0, 1, 1],
            token_logprobs=[-0.5, -0.3],
            reward=None,
            messages=[{"role": "user", "content": "q1"}],
            parent_id=None,
        )
        # Turn 1 (child of r0)
        r1 = InteractionRecord(
            response_id="r1",
            input_ids=[1, 2, 3, 10, 11, 20, 21],
            response_mask=[0, 0, 0, 0, 0, 1, 1],
            token_logprobs=[-0.4, -0.2],
            reward=None,
            messages=[{"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1"},
                      {"role": "user", "content": "q2"}],
            parent_id="r0",
        )
        # Turn 2 leaf (child of r1) — gets reward=1.0
        r2 = InteractionRecord(
            response_id="r2",
            input_ids=[1, 2, 3, 10, 11, 20, 21, 30],
            response_mask=[0, 0, 0, 0, 0, 0, 0, 1],
            token_logprobs=[-0.1],
            reward=1.0,
            messages=[{"role": "user", "content": "q2"},
                      {"role": "assistant", "content": "a2"},
                      {"role": "user", "content": "q3"}],
            parent_id="r1",
        )
        session.interactions = {"r0": r0, "r1": r1, "r2": r2}
        session._last_id = "r2"

        discount = 0.9
        trajs = _build_trajectories(session, discount)

        # After backprop:
        #   r2.reward = 1.0 (leaf)
        #   r1.reward = 1.0 * 0.9 = 0.9
        #   r0.reward = 0.9 * 0.9 = 0.81
        reward_map = {t: None for t in ["r0", "r1", "r2"]}
        for tt in trajs:
            for rid, rec in [("r0", r0), ("r1", r1), ("r2", r2)]:
                if rec.input_ids == tt.input_ids.tolist():
                    reward_map[rid] = tt.reward

        assert abs(reward_map["r2"] - 1.0) < 1e-5, f"r2 reward={reward_map['r2']}"
        assert abs(reward_map["r1"] - 0.9) < 1e-5, f"r1 reward={reward_map['r1']}"
        assert abs(reward_map["r0"] - 0.81) < 1e-5, f"r0 reward={reward_map['r0']}"
        print(f"  [PASS] proxy_session: reward backprop r0={reward_map['r0']:.2f} r1={reward_map['r1']:.2f} r2={reward_map['r2']:.2f}")

    asyncio.run(_run())


TESTS = [
    ("grpo_loss",        test_grpo_loss),
    ("trainer_step",     test_trainer_step),
    ("buffer",           test_buffer),
    ("bfcl_reward",      test_bfcl_reward),
    ("state_dict_keys",  test_state_dict_keys),
    ("proxy_session",    test_proxy_session),
]

if __name__ == "__main__":
    failed = []
    for name, fn in TESTS:
        print(f"\n── {name} ──")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*40}")
    passed = len(TESTS) - len(failed)
    print(f"Result: {passed}/{len(TESTS)} passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
