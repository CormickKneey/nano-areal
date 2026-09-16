import types

import torch

from nano_areal.engine.proxy import _extract_sampled_tokens
from nano_areal.engine.types import InteractionRecord


def _choice(tokens, logprobs, finish="stop"):
    content = [
        types.SimpleNamespace(token=t, logprob=lp) for t, lp in zip(tokens, logprobs)
    ]
    return types.SimpleNamespace(
        logprobs=types.SimpleNamespace(content=content), finish_reason=finish
    )


def test_extract_sampled_tokens_from_token_ids():
    ids, lps = _extract_sampled_tokens(
        _choice(["token_id:12", "token_id:7"], [-0.1, -0.2])
    )
    assert ids == [12, 7]
    assert lps == [-0.1, -0.2]


def test_extract_sampled_tokens_falls_back_on_plain_text():
    ids, lps = _extract_sampled_tokens(_choice(["Hello", " world"], [-0.1, -0.2]))
    assert ids == []
    assert lps == [-0.1, -0.2]


def test_interaction_record_to_trajectory_carries_stop_reason_and_alignment():
    rec = InteractionRecord(
        response_id="r",
        input_ids=[1, 2, 3, 4, 5],
        response_mask=[0, 0, 1, 1, 1],
        token_logprobs=[-0.5, -0.6, -0.7],
        reward=1.0,
        stop_reason="length",
        turn_id=2,
    )
    tt = rec.to_trajectory()
    assert tt.is_truncated and tt.turn_id == 2
    torch.testing.assert_close(tt.logprobs, torch.tensor([0.0, 0.0, -0.5, -0.6, -0.7]))
    # trainer shifts both by [1:] so index j describes target token j+1
    shifted_lp = tt.logprobs[1:]
    shifted_mask = tt.response_mask[1:]
    assert shifted_mask.tolist() == [0, 1, 1, 1]
    torch.testing.assert_close(
        shifted_lp[shifted_mask.bool()],
        torch.tensor([-0.5, -0.6, -0.7]),
        atol=1e-6,
        rtol=0,
    )
