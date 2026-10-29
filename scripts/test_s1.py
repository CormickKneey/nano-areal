"""S1 probes that do not need Apple Silicon.

Algorithm checks run here. The Mac VLM smoke is reported by
``scripts/probe_mac_vlm.py`` and stays blocked until that machine is available.

    uv run python scripts/test_s1.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch


def test_messages_base_url():
    from nano_areal.engine.messages import messages_base_url

    assert messages_base_url("http://127.0.0.1:8001/v1") == "http://127.0.0.1:8001"
    assert messages_base_url("http://127.0.0.1:8001/v1/") == "http://127.0.0.1:8001"
    assert messages_base_url("http://127.0.0.1:8001") == "http://127.0.0.1:8001"
    print("  [PASS] messages_base_url")


def test_dsh_kwargs_do_not_put_the_key_on_argv():
    from nano_areal.agent.dsh import build_dsh_kwargs, episode_reward

    kwargs = build_dsh_kwargs(
        base_url="http://127.0.0.1:8001/v1",
        api_key="session-secret",
        cwd="/tmp",
        dsh_home="/tmp/dsh-home",
        model="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        context_window_tokens=8192,
        max_tokens=1024,
    )
    assert kwargs["profile"] == "sdk-minimal"
    assert kwargs["provider"] == "deepseek-official"
    assert kwargs["base_url"] == "http://127.0.0.1:8001"
    assert kwargs["api_key"] == "session-secret"
    assert kwargs["env"]["DSH_CONTEXT_WINDOW"] == "8192"
    assert "session-secret" not in json.dumps({k: v for k, v in kwargs.items() if k != "api_key"})
    assert episode_reward("max-tokens", 1.0) == 0.0
    assert episode_reward("error", 1.0) == 0.0
    assert episode_reward("completed", None) == 0.0
    assert episode_reward("completed", 0.5) == 0.5
    print("  [PASS] dsh kwargs")


def test_messages_sse_shape():
    from nano_areal.engine.messages import anthropic_request_to_openai, openai_completion_to_messages_sse

    openai = anthropic_request_to_openai({
        "model": "qwen",
        "system": "be brief",
        "max_tokens": 64,
        "dsh_session_log": {"ignore": True},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "count"},
                {"type": "image", "source": {"type": "base64", "data": "xx"}},
            ],
        }],
        "tools": [{
            "name": "bash",
            "description": "shell",
            "input_schema": {"type": "object", "properties": {}},
        }],
    })
    assert openai["messages"][0]["role"] == "system"
    assert "<image>" in openai["messages"][1]["content"]
    assert "dsh_session_log" not in openai
    assert openai["tools"][0]["function"]["name"] == "bash"

    sse = openai_completion_to_messages_sse({
        "id": "chatcmpl-1",
        "model": "qwen",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": "ok",
                "reasoning_content": "think",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"},
                }],
            },
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    })
    assert "event: message_start" in sse
    assert "event: message_stop" in sse
    assert '"stop_reason": "tool_use"' in sse
    assert "text/event-stream" not in sse  # media type is the HTTP header, not the body
    events = []
    for block in sse.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        payload = json.loads(lines["data"])
        assert lines["event"] == payload["type"]
        events.append(payload)
    assert events[0]["type"] == "message_start"
    assert events[-1]["type"] == "message_stop"
    assert any(ev.get("delta", {}).get("stop_reason") == "tool_use" for ev in events)
    print("  [PASS] messages sse")


def test_loss_switch_and_groups():
    from nano_areal.archon.losses import compute_policy_loss, group_advantages, is_offpolicy

    assert is_offpolicy(1, 0)
    assert not is_offpolicy(0, 0)
    assert not is_offpolicy(2, 4)
    assert is_offpolicy(5, 4)
    assert not is_offpolicy(-1, 0)

    rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])
    adv, stats = group_advantages(rewards, group_size=2, scale=True, on_incomplete="zeros")
    assert stats["skipped_groups"] == 1
    assert torch.allclose(adv[:2], torch.zeros(2))
    assert stats["advantage_scope"] == "group"

    short, short_stats = group_advantages(
        torch.tensor([1.0, 0.0]), group_size=4, scale=True, on_incomplete="batch",
    )
    assert short_stats["incomplete_groups"] == 1
    assert short_stats["advantage_scope"] == "batch"
    assert torch.isfinite(short).all()

    B, T = 4, 6
    ref = torch.zeros(B, T)
    masks = torch.ones(B, T)
    rewards_t = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # Row 0 is far outside the PPO clip and has a positive advantage.
    ppo_lp = torch.zeros(B, T, requires_grad=True)
    ppo_lp.data[0] = 5.0
    ppo, _ = compute_policy_loss("grpo", ppo_lp, ref, rewards_t, masks, kl_coef=0.0)
    ppo.backward()
    assert ppo_lp.grad is not None
    assert float(ppo_lp.grad[0].abs().sum()) == 0.0

    fresh = torch.zeros(B, T, requires_grad=True)
    fresh.data[0] = 5.0
    cispo, cispo_stats = compute_policy_loss("cispo", fresh, ref, rewards_t, masks, kl_coef=0.0)
    cispo.backward()
    assert fresh.grad is not None and float(fresh.grad[0].abs().sum()) > 0
    assert cispo_stats["loss_name"] == "cispo"

    gspo_lp = torch.randn(2, 4, requires_grad=True)
    gspo, gspo_stats = compute_policy_loss(
        "gspo", gspo_lp, torch.zeros(2, 4), torch.tensor([1.0, 0.0]), torch.ones(2, 4), group_size=2,
    )
    assert torch.isfinite(gspo)
    assert gspo_stats["advantage_scope"] == "group"
    gspo.backward()

    dr = torch.tensor([0.0, 1.0, 3.0])
    dr_adv, _ = group_advantages(dr, group_size=3, scale=False)
    centered = dr - dr.mean()
    assert torch.allclose(dr_adv, centered)
    long_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    lp = torch.ones(2, 3, requires_grad=True)
    loss, stats = compute_policy_loss(
        "drgrpo", lp, torch.zeros(2, 3), torch.tensor([1.0, 0.0]), long_mask, group_size=2, kl_coef=0.0,
    )
    assert stats["loss_name"] == "drgrpo"
    assert torch.isfinite(loss)
    try:
        compute_policy_loss("ppo", lp, torch.zeros_like(lp), torch.zeros(2), long_mask)
        raise AssertionError("unknown loss should fail")
    except ValueError:
        pass
    print("  [PASS] loss switch")


def test_backend_command_and_truncation():
    from nano_areal.engine.rollout import build_inference_command
    from nano_areal.engine.types import InteractionRecord

    metal, metal_env = build_inference_command(
        backend="metal",
        model="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        tokenizer="Qwen/Qwen3-VL-4B-Instruct",
        host="127.0.0.1",
        port=8000,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_ids=[0],
    )
    assert metal[0] == "vllm-metal"
    assert "CUDA_VISIBLE_DEVICES" not in metal_env
    cuda, cuda_env = build_inference_command(
        backend="cuda",
        model="Qwen/Qwen3-0.6B",
        tokenizer="Qwen/Qwen3-0.6B",
        host="127.0.0.1",
        port=8000,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_ids=[0],
    )
    assert cuda[0:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert cuda_env["CUDA_VISIBLE_DEVICES"] == "0"

    rec = InteractionRecord(
        response_id="r",
        input_ids=[1, 2, 3],
        response_mask=[0, 1, 1],
        token_logprobs=[-0.1, -0.2],
        reward=1.0,
        finish_reason="length",
    )
    traj = rec.to_trajectory()
    assert traj.truncated
    assert traj.reward == 0.0
    print("  [PASS] backend command + truncation")


def test_hidden_tests():
    from nano_areal.reward.dsh_task import hidden_test_pass_rate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert hidden_test_pass_rate(root) is None
        (root / "test_ok.py").write_text("def test_ok():\n    assert True\n")
        assert hidden_test_pass_rate(root) == 1.0
        (root / "test_bad.py").write_text("def test_bad():\n    assert False\n")
        score = hidden_test_pass_rate(root)
        assert score == 0.5, score
    print("  [PASS] hidden tests")


def test_offpolicy_buffer_sync_gap_zero():
    import asyncio
    from nano_areal.archon.trainer import Trajectory
    from nano_areal.engine.buffer import AsyncTrajectoryBuffer

    async def _run():
        buf = AsyncTrajectoryBuffer(max_head_offpolicyness=0, maxsize=8)
        buf.put_nowait(Trajectory(
            input_ids=torch.zeros(2, dtype=torch.long),
            response_mask=torch.ones(2),
            reward=1.0,
            policy_version=0,
        ))
        buf.update_policy_version(1)
        buf.put_nowait(Trajectory(
            input_ids=torch.zeros(2, dtype=torch.long),
            response_mask=torch.ones(2),
            reward=1.0,
            policy_version=1,
        ))
        batch = await buf.get_batch_sync(1)
        assert len(batch) == 1
        assert batch[0].policy_version == 1
        assert buf.stats.total_discarded == 1

    asyncio.run(_run())
    print("  [PASS] sync offpolicy gap 0")


def main():
    tests = [
        test_messages_base_url,
        test_dsh_kwargs_do_not_put_the_key_on_argv,
        test_messages_sse_shape,
        test_loss_switch_and_groups,
        test_backend_command_and_truncation,
        test_hidden_tests,
        test_offpolicy_buffer_sync_gap_zero,
    ]
    for test in tests:
        test()
    print(f"S1 probes: {len(tests)} passed")


if __name__ == "__main__":
    main()
