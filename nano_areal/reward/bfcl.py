from __future__ import annotations

import ast
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Normalization helpers (mirrors BFCL official eval logic)
# ---------------------------------------------------------------------------

_STRIP_CHARS = re.compile(r'[,.\-/_*^() ]')


def normalize(s: str) -> str:
    """Lowercase + strip punctuation for fuzzy matching."""
    return _STRIP_CHARS.sub("", s.lower())


def _coerce(value: Any, expected_type: str) -> Any:
    """Best-effort type coercion for BFCL parameter matching."""
    try:
        if expected_type in ("integer", "int"):
            return int(float(str(value)))
        if expected_type in ("float", "number"):
            return float(str(value))
        if expected_type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes")
        if expected_type == "string":
            return str(value)
    except (ValueError, TypeError):
        pass
    return value


# ---------------------------------------------------------------------------
# AST-based function call verifier
# ---------------------------------------------------------------------------

class BFCLReward:
    """
    Compute turn-level reward for BFCL function call predictions.

    Scoring follows the BFCL official rubric:
    - Exact function name match (required)
    - All required arguments present with correct values
    - Values are matched against a list of accepted answers (for optional params)
    - Parallel calls: order-independent matching
    - Returns 1.0 on full match, 0.0 otherwise
    """

    def __call__(
        self,
        predicted: list[dict] | None,
        ground_truth: dict | list[dict],
    ) -> float:
        if not predicted:
            # No tool call made; only valid if ground truth is also empty
            return 1.0 if not ground_truth else 0.0

        if isinstance(ground_truth, dict):
            ground_truth = [ground_truth]

        # Parallel calls: find a 1-1 matching between predicted and ground truth
        return float(self._match_parallel(predicted, ground_truth))

    # ------------------------------------------------------------------
    # Internal matching
    # ------------------------------------------------------------------

    def _match_parallel(
        self,
        predicted: list[dict],
        ground_truth: list[dict],
    ) -> bool:
        """Order-independent matching for parallel function calls."""
        if len(predicted) != len(ground_truth):
            return False

        used = [False] * len(ground_truth)
        for pred in predicted:
            matched = False
            for i, gt in enumerate(ground_truth):
                if not used[i] and self._match_single(pred, gt):
                    used[i] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _match_single(self, predicted: dict, ground_truth: dict) -> bool:
        """Match one predicted call against one ground-truth entry."""
        if not ground_truth:
            return True  # empty GT = no constraint

        # ground_truth format: {"fn_name": {"arg": [accepted_values], ...}}
        assert len(ground_truth) == 1, "Each GT entry maps exactly one function"
        fn_name, expected_args = next(iter(ground_truth.items()))

        pred_name = predicted.get("name", "")
        if normalize(pred_name) != normalize(fn_name):
            return False

        pred_args = predicted.get("arguments", {})
        if isinstance(pred_args, str):
            try:
                pred_args = json.loads(pred_args)
            except json.JSONDecodeError:
                return False

        return self._match_args(pred_args, expected_args)

    def _match_args(self, predicted: dict, expected: dict) -> bool:
        """
        Check predicted arguments against expected.
        expected[arg] = list of accepted values (empty list = any value OK).
        """
        for arg, accepted_values in expected.items():
            if not accepted_values:
                # Empty accepted list means the arg can be omitted or any value
                continue

            pred_val = predicted.get(arg)

            # Missing required argument
            if pred_val is None:
                # Check if the ground truth accepts an empty/null value
                if any(v in ("", None) for v in accepted_values):
                    continue
                return False

            if not self._value_in_accepted(pred_val, accepted_values):
                return False

        return True

    def _value_in_accepted(self, value: Any, accepted: list) -> bool:
        """Return True if value matches any entry in accepted."""
        norm_value = normalize(str(value))
        for acc in accepted:
            if acc is None or acc == "":
                continue
            if normalize(str(acc)) == norm_value:
                return True
            # Numeric tolerance
            try:
                if abs(float(value) - float(acc)) < 1e-6:
                    return True
            except (ValueError, TypeError):
                pass
        return False


# ---------------------------------------------------------------------------
# Tool call parser: model output → list[dict]
# ---------------------------------------------------------------------------

def parse_tool_calls(raw_output: Any) -> list[dict]:
    """
    Parse model output into a list of {name, arguments} dicts.
    Handles:
      - openai.types.chat.ChatCompletionMessage with .tool_calls
      - Raw string with JSON function call
      - Already-parsed list/dict
    """
    # OpenAI message object
    if hasattr(raw_output, "tool_calls") and raw_output.tool_calls:
        result = []
        for tc in raw_output.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            result.append({"name": tc.function.name, "arguments": args})
        return result

    # Already a list
    if isinstance(raw_output, list):
        return raw_output

    # Already a dict (single call)
    if isinstance(raw_output, dict):
        return [raw_output]

    # Raw string: try JSON parse
    if isinstance(raw_output, str):
        return _parse_string(raw_output)

    return []


def _parse_string(s: str) -> list[dict]:
    """Extract function calls from a raw string response."""
    s = s.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Extract JSON block from markdown code fence
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if code_block:
        return _parse_string(code_block.group(1).strip())

    # Try Python literal eval as fallback
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except (ValueError, SyntaxError):
        pass

    return []


# ---------------------------------------------------------------------------
# Trajectory-level reward with turn discount (mirrors AReaL's apply_reward_discount)
# ---------------------------------------------------------------------------

def compute_trajectory_reward(
    turn_rewards: list[float],
    turn_discount: float = 0.9,
) -> list[float]:
    """
    Apply exponential discount: earlier correct answers get higher reward.

    turn_rewards[i] ∈ {0.0, 1.0} per turn.
    Returns discounted rewards for each turn.
    """
    discounted = []
    for i, r in enumerate(turn_rewards):
        discounted.append(r * (turn_discount ** i))
    return discounted
