"""Unit tests for the v0.2 building blocks: distillation, PRM, turn-level GAE,
tensor transport, image extraction, code execution, dsh reward, tasks."""
import asyncio
import base64
import io

import pytest
import torch

from nano_areal.archon.distill import (
    ConstantTeacher,
    DistillConfig,
    distill_loss,
    groups_to_distill,
    teacher_logprob_tensor,
)
from nano_areal.archon.losses import turn_level_gae
from nano_areal.engine.local_server import parse_hermes_tool_calls, split_reasoning
from nano_areal.engine.multimodal import (
    extract_images,
    normalize_messages_for_template,
    text_of,
)
from nano_areal.engine.types import b64_to_tensor, tensor_to_b64
from nano_areal.reward.code_exec import extract_code_block, run_assert_tests
from nano_areal.reward.dsh_task import dsh_task_reward, workspace_files
from nano_areal.reward.prm import ProcessRewardConfig, process_reward
from nano_areal.tasks.chart import synthetic_charts
from nano_areal.tasks.code import CodeSample, builtin_nano_swe
from nano_areal.tasks.vision import (
    clevr_count_reward,
    geometry3k_reward,
    synthetic_clevr,
)

# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------

def test_groups_to_distill_respects_pass_rate_trigger():
    cfg = DistillConfig(distill_coef=1.0, trigger_max_pass_rate=0.5)
    rewards = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    gids = ["a", "a", "b", "b", "c", "c"]
    assert groups_to_distill(rewards, gids, cfg) == {"b", "c"}


def test_teacher_logprob_tensor_pads_and_clamps():
    t = teacher_logprob_tensor(ConstantTeacher(-50.0), torch.arange(5), None, None, T_pad=8, clamp_min=-20.0)
    assert t.shape == (7,)
    torch.testing.assert_close(t[:4], torch.full((4,), -20.0), atol=0, rtol=0)
    torch.testing.assert_close(t[4:], torch.zeros(3), atol=0, rtol=0)


def test_distill_loss_gradient_pulls_student_toward_teacher():
    student = torch.tensor([[-2.0, -2.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, stats = distill_loss(student, teacher, mask)
    loss.backward()
    assert stats["distill_loss"] == pytest.approx(-1.0)
    assert stats["distill_frac"] == 1.0
    # k1 estimator: d/d logp_s = mask / n; masked position gets no gradient
    torch.testing.assert_close(student.grad, torch.tensor([[0.5, 0.5, 0.0]]), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# PRM + GAE
# ---------------------------------------------------------------------------

def test_process_reward_truncation_and_tool_validity():
    cfg = ProcessRewardConfig(weight=1.0, known_tools=("run",))
    r, parts = process_reward(
        response_content="x", tool_calls=[], n_response_tokens=10, stop_reason="length", cfg=cfg
    )
    assert parts == {"truncated": -1.0} and r == -1.0
    bad = [{"function": {"name": "nope", "arguments": "{}"}}]
    _, parts = process_reward(response_content="", tool_calls=bad, n_response_tokens=1, stop_reason="stop", cfg=cfg)
    assert parts["tool_calls"] == cfg.penalty_bad_tool_call
    good = [{"function": {"name": "run", "arguments": '{"cmd": "ls"}'}}]
    _, parts = process_reward(response_content="", tool_calls=good, n_response_tokens=1, stop_reason="stop", cfg=cfg)
    assert parts["tool_calls"] == cfg.bonus_valid_tool_call
    _, parts = process_reward(response_content="  ", tool_calls=[], n_response_tokens=1, stop_reason="stop", cfg=cfg)
    assert parts == {"empty": cfg.penalty_empty}


def test_turn_level_gae_reduces_to_returns_without_critic():
    assert turn_level_gae([0.0, 0.0, 1.0]) == [1.0, 1.0, 1.0]
    adv = turn_level_gae([0.0, 0.0, 1.0], gamma=0.5)
    assert adv == pytest.approx([0.25, 0.5, 1.0])
    adv = turn_level_gae([1.0, 1.0], turn_values=[0.5, 0.5], gamma=1.0, lam=0.0)
    assert adv == pytest.approx([1.0, 0.5])


# ---------------------------------------------------------------------------
# transport / multimodal helpers
# ---------------------------------------------------------------------------

def test_tensor_b64_roundtrip():
    f = torch.randn(3, 4)
    i = torch.tensor([[1, 2, 3]])
    fr, ir = b64_to_tensor(tensor_to_b64(f)), b64_to_tensor(tensor_to_b64(i))
    assert fr.dtype == torch.float32 and ir.dtype == torch.int64
    torch.testing.assert_close(fr, f, atol=1e-2, rtol=1e-2)  # float16 transport
    assert torch.equal(ir, i)


def _png_data_url() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (255, 0, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_extract_images_and_normalize_messages():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "How many?"},
            {"type": "image_url", "image_url": {"url": _png_data_url()}},
        ]},
        {"role": "assistant", "content": "3", "reasoning_content": "count them"},
    ]
    imgs = extract_images(messages)
    assert len(imgs) == 1 and (imgs[0].meta["width"], imgs[0].meta["height"]) == (8, 6)
    assert imgs[0].meta["source"] == "data:image/png"
    assert text_of(messages[0]["content"]) == "How many?"
    norm = normalize_messages_for_template(messages)
    assert norm[0]["content"][1]["type"] == "image"
    assert "data:" not in str(norm[0]["content"][1])
    assert norm[1]["content"] == "3"


def test_hermes_tool_call_and_reasoning_parsing():
    text = '<think>plan</think>ok <tool_call>{"name": "run", "arguments": {"cmd": "ls"}}</tool_call>'
    reasoning, rest = split_reasoning(text)
    assert reasoning == "plan"
    content, calls = parse_hermes_tool_calls(rest)
    assert content == "ok" and len(calls) == 1
    assert calls[0]["function"]["name"] == "run" and '"cmd"' in calls[0]["function"]["arguments"]
    assert parse_hermes_tool_calls("plain") == ("plain", [])


# ---------------------------------------------------------------------------
# code execution / rewards / tasks
# ---------------------------------------------------------------------------

def test_extract_code_block_prefers_fenced_python():
    txt = "Here:\n```python\ndef f():\n    return 1\n```\nbye"
    assert extract_code_block(txt) == "def f():\n    return 1"
    assert extract_code_block("def g(): pass") == "def g(): pass"


def test_run_assert_tests_partial_pass():
    frac, res = asyncio.run(run_assert_tests(
        "def add(a, b):\n    return a + b\n", ["assert add(1, 2) == 3", "assert add(1, 1) == 3"]))
    assert frac == pytest.approx(0.5) and not res.timed_out


def test_dsh_task_reward_workspace(tmp_path):
    sample = CodeSample(task_id="t", prompt="add", tests=["assert add(1, 2) == 3"],
                        starter="def add(a, b):\n    pass\n")
    files = workspace_files(sample)
    assert set(files) == {"README.md", "solution.py", "tests/test_solution.py"}
    r, info = asyncio.run(dsh_task_reward(sample, tmp_path))
    assert r == 0.0 and info["reason"] == "missing_solution"
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    r, info = asyncio.run(dsh_task_reward(sample, tmp_path))
    assert r == 1.0 and info["pass_frac"] == 1.0
    (tmp_path / "solution.py").write_text("def add(a, b:\n")
    r, info = asyncio.run(dsh_task_reward(sample, tmp_path))
    assert r == 0.0 and info["reason"] == "syntax_error"


def test_vision_rewards():
    assert clevr_count_reward("There are <answer>3</answer> cubes", "3") == 1.0
    assert clevr_count_reward("4", "3") == 0.0
    assert geometry3k_reward("<answer>10.05</answer>", "10") == 1.0
    assert geometry3k_reward("<answer>12</answer>", "10") == 0.0


def test_synthetic_datasets():
    clevr = synthetic_clevr(4, seed=1)
    assert len(clevr) == 4
    s = clevr[0]
    assert s.image.size == (64, 64) and 0 <= int(s.answer) <= 5
    charts = synthetic_charts(3)
    assert len(charts) == 3 and "plt." in charts[0].reference_code
    swe = builtin_nano_swe()
    assert len(swe) >= 3 and all(x.tests for x in swe)
