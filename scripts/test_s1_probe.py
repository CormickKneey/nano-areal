"""Offline validation of the Mac probe; no inference engine or weights needed."""

from __future__ import annotations

import argparse
import unittest

from probe_mac_vlm import local_url, red_png, summarize_completion


class ProbeTests(unittest.TestCase):
    def completion(self):
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "red"},
                    "logprobs": {"content": [{"token": "red", "logprob": -0.1}]},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        }

    def test_finite_logprobs_and_usage(self):
        result = summarize_completion(self.completion())
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["logprobs_match_usage"])

    def test_missing_logprobs_blocks(self):
        response = self.completion()
        response["choices"][0]["logprobs"] = None
        self.assertEqual(summarize_completion(response)["status"], "blocked")

    def test_nan_and_truncation_block(self):
        response = self.completion()
        response["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
        self.assertFalse(summarize_completion(response)["finite_logprobs"])
        response = self.completion()
        response["choices"][0]["finish_reason"] = "length"
        self.assertEqual(summarize_completion(response)["status"], "blocked")

    def test_token_count_mismatch_blocks(self):
        response = self.completion()
        response["usage"]["completion_tokens"] = 2
        self.assertEqual(summarize_completion(response)["status"], "blocked")

    def test_only_local_endpoints(self):
        for value in (
            "https://example.com",
            "http://key@localhost/v1",
            "http://localhost/v1?api_key=secret",
        ):
            with self.assertRaises(argparse.ArgumentTypeError):
                local_url(value)
        self.assertEqual(
            local_url("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1"
        )

    def test_image_is_png(self):
        self.assertTrue(red_png().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
