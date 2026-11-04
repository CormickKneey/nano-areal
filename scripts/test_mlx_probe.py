"""Check MLX terminal-token accounting without loading a model or Metal."""

from types import SimpleNamespace

import numpy as np
import pytest
from probe_mlx_vlm import collect_completion


def chunk(token, *, final=None, token_ids=None, count=1, logprob=-0.25):
    return SimpleNamespace(
        text="Red" if final is None else "",
        token=token,
        logprobs=np.array([[logprob, logprob, logprob]]),
        finish_reason=final,
        token_ids=token_ids,
        generation_tokens=count,
        prompt_tokens=84,
    )


def test_stop_token_is_counted_once():
    result = collect_completion(
        [
            chunk(1),
            chunk(2, final="stop", token_ids=[1, 2], count=2),
        ]
    )
    assert result["status"] == "passed"
    assert result["token_ids"] == [1, 2]
    assert result["token_logprobs"] == [-0.25, -0.25]
    assert result["response_text"] == "Red"


def test_length_terminal_does_not_duplicate_last_token():
    result = collect_completion(
        [
            chunk(1),
            chunk(1, final="length", token_ids=[1]),
        ]
    )
    assert result["token_ids"] == [1]
    assert result["logprobs_match_usage"]
    assert result["status"] == "blocked"


@pytest.mark.parametrize("logprob", [float("nan"), float("inf"), 0.1])
def test_invalid_logprobs_block(logprob):
    result = collect_completion(
        [
            chunk(2, final="stop", token_ids=[2], logprob=logprob),
        ]
    )
    assert result["status"] == "blocked"
    assert not result["finite_logprobs"]


def test_missing_terminal_blocks():
    with pytest.raises(ValueError, match="terminal"):
        collect_completion([chunk(1)])


def test_usage_mismatch_blocks():
    result = collect_completion(
        [
            chunk(2, final="stop", token_ids=[2], count=2),
        ]
    )
    assert result["status"] == "blocked"
