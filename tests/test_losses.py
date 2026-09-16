import torch

from nano_areal.archon.losses import (
    RejectionSamplingConfig,
    group_normalize_advantages,
    kl_penalty,
    policy_loss,
)
from nano_areal.archon.trainer import grpo_loss


def test_group_normalize_grouped_zero_mean_per_group():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    groups = ["a", "a", "b", "b", "c", "c"]
    adv, stats = group_normalize_advantages(rewards, groups)
    torch.testing.assert_close(adv[:2].sum(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(adv[2:4].sum(), torch.tensor(0.0), atol=1e-6, rtol=0)
    # group c has zero variance → zero advantage, not NaN
    assert torch.all(adv[4:] == 0)
    assert stats["n_groups"] == 3
    assert stats["n_zero_var_groups"] == 1
    assert torch.isfinite(adv).all()


def test_group_normalize_incomplete_group_gets_zero_advantage():
    rewards = torch.tensor([1.0, 0.0, 5.0])
    adv, stats = group_normalize_advantages(rewards, ["a", "a", "lonely"])
    assert adv[2] == 0.0
    assert adv[0] > 0 > adv[1]
    assert stats["n_incomplete_groups"] == 1


def test_group_normalize_batch_mode_matches_legacy():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv, _ = group_normalize_advantages(rewards, ["x"] * 4, norm="batch")
    expected = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    torch.testing.assert_close(adv, expected, atol=1e-5, rtol=0)


def _inputs(B=4, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    logp = torch.randn(B, T, generator=g).mul(0.1).sub(1.0).requires_grad_(True)
    prox = logp.detach() + torch.randn(B, T, generator=g) * 0.05
    behave = prox + torch.randn(B, T, generator=g) * 0.05
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(B, T)
    mask[:, :2] = 0
    return logp, prox, behave, adv, mask


def test_policy_loss_on_policy_grpo_equals_reinforce():
    logp, _, _, adv, mask = _inputs()
    # π_prox == π_θ → ratio == 1, PPO-clip inactive: loss = -mean(adv) over tokens
    loss, stats = policy_loss(logp, logp.detach(), adv, mask, loss_type="grpo")
    expected = -(adv.unsqueeze(1) * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=0)
    assert stats["clip_frac"] == 0
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()
    # prompt tokens receive no gradient
    assert torch.all(logp.grad[:, :2] == 0)


def test_policy_loss_all_variants_finite_with_grad():
    for kind in ("grpo", "dr_grpo", "cispo", "gspo"):
        logp, prox, _, adv, mask = _inputs()
        loss, stats = policy_loss(
            logp, prox, adv, mask, loss_type=kind, clip_eps_high=0.28
        )
        assert torch.isfinite(loss), kind
        loss.backward()
        assert torch.isfinite(logp.grad).all(), kind
        assert 0 <= float(stats["clip_frac"]) <= 1


def test_rejection_mask_drops_stale_tokens():
    logp, prox, behave, adv, mask = _inputs()
    behave = behave.clone()
    behave[0, 3] -= 5.0  # π_prox/π_behave ≈ e^5 ≫ cap → rejected
    cfg = RejectionSamplingConfig(action="mask", upper=5.0)
    loss, stats = policy_loss(
        logp, prox, adv, mask, behavior_logprobs=behave, rejection=cfg
    )
    loss.backward()
    assert stats["rejected_frac"] > 0
    assert logp.grad[0, 3] == 0
    assert logp.grad[0, 4] != 0


def test_rejection_clamp_bounds_weight():
    logp, prox, behave, adv, mask = _inputs()
    behave = behave.clone()
    behave[1, 4] -= 5.0
    cfg = RejectionSamplingConfig(action="clamp", upper=2.0)
    _, stats = policy_loss(
        logp, prox, adv, mask, behavior_logprobs=behave, rejection=cfg
    )
    assert float(stats["behave_imp_weight_mean"]) <= 2.0


def test_kl_k3_nonnegative_and_zero_at_equality():
    logp = torch.randn(2, 5)
    mask = torch.ones(2, 5)
    kl, mean = kl_penalty(logp, logp.clone(), mask, kl_type="k3")
    torch.testing.assert_close(mean, torch.tensor(0.0), atol=1e-7, rtol=0)
    kl, mean = kl_penalty(logp, logp + torch.randn(2, 5), mask, kl_type="k3")
    assert (kl >= 0).all()


def test_grpo_loss_masks_truncated_and_reports_stats():
    B, T = 4, 8
    logp = torch.randn(B, T, requires_grad=True)
    ref = torch.randn(B, T)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(B, T)
    truncated = torch.tensor([True, False, False, False])
    loss, stats = grpo_loss(
        logp,
        ref,
        rewards,
        mask,
        group_ids=["g", "g", "g", "g"],
        truncated=truncated,
        mask_truncated=True,
    )
    assert torch.isfinite(loss)
    assert stats["truncated_frac"] == 0.25
    assert stats["n_groups"] == 1
    assert stats["rejected_frac"] == 0.0
    loss.backward()
    # truncated sample: advantage zeroed → only KL gradient remains (kl_coef=0.01)
    assert logp.grad[0].abs().max() < logp.grad[2].abs().max()


def test_grpo_loss_legacy_signature_still_works():
    B, T = 8, 32
    logp = torch.randn(B, T, requires_grad=True)
    ref = torch.randn(B, T)
    loss, stats = grpo_loss(logp, ref, torch.randn(B), torch.ones(B, T))
    assert torch.isfinite(loss)
    assert "kl" in stats and "reward_mean" in stats
