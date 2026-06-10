"""Tests for provider-agnostic canonical trace comparison."""
from __future__ import annotations

import unittest

from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling
from core.canonical_trace_diff import (
    CanonicalTrace,
    find_first_divergence,
    traces_equal,
)


def _trace(
    *,
    model: str = "demo",
    messages=None,
    temperature: float = 0.7,
    prompt: str = "rendered prompt",
    output: str = "answer",
) -> CanonicalTrace:
    return CanonicalTrace(
        request=CanonicalRequest(
            model=model,
            messages=messages
            or [CanonicalMessage(role="user", content="hello")],
            sampling=CanonicalSampling(temperature=temperature, top_p=0.9),
            stop=["END"],
        ),
        prompt=prompt,
        output=output,
    )


class CanonicalTraceDiffTests(unittest.TestCase):
    def test_matching_traces(self) -> None:
        a = _trace()
        b = _trace()
        report = find_first_divergence(a, b)
        self.assertTrue(report["request_match"])
        self.assertTrue(report["prompt_match"])
        self.assertTrue(report["output_match"])
        self.assertIsNone(report["first_divergence_level"])
        self.assertEqual(report["diff_summary"], "traces match")
        self.assertEqual(report["differences"], [])
        self.assertTrue(traces_equal(a, b))

    def test_request_divergence_first(self) -> None:
        a = _trace(temperature=0.7)
        b = _trace(temperature=0.2)
        report = find_first_divergence(a, b)
        self.assertFalse(report["request_match"])
        self.assertEqual(report["first_divergence_level"], "REQUEST")
        self.assertTrue(any(d["aspect"] == "sampling" for d in report["differences"]))

    def test_message_list_divergence(self) -> None:
        a = _trace(messages=[CanonicalMessage(role="user", content="one")])
        b = _trace(messages=[CanonicalMessage(role="user", content="two")])
        report = find_first_divergence(a, b)
        self.assertEqual(report["first_divergence_level"], "REQUEST")
        self.assertTrue(any(d["aspect"] == "messages" for d in report["differences"]))

    def test_prompt_divergence_when_request_matches(self) -> None:
        a = _trace(prompt="prompt A")
        b = _trace(prompt="prompt B")
        report = find_first_divergence(a, b)
        self.assertTrue(report["request_match"])
        self.assertFalse(report["prompt_match"])
        self.assertEqual(report["first_divergence_level"], "PROMPT")
        aspects = {d["aspect"] for d in report["differences"]}
        self.assertIn("string", aspects)

    def test_prompt_whitespace_only_string_diff_fingerprint_match(self) -> None:
        a = _trace(prompt="hello\r\nworld  ")
        b = _trace(prompt="hello\nworld")
        report = find_first_divergence(a, b)
        self.assertTrue(report["request_match"])
        self.assertFalse(report["prompt_match"])
        self.assertEqual(report["first_divergence_level"], "PROMPT")
        aspects = {d["aspect"] for d in report["differences"]}
        self.assertIn("string", aspects)
        self.assertNotIn("fingerprint", aspects)

    def test_output_divergence_last(self) -> None:
        a = _trace(output="alpha")
        b = _trace(output="beta")
        report = find_first_divergence(a, b)
        self.assertTrue(report["request_match"])
        self.assertTrue(report["prompt_match"])
        self.assertFalse(report["output_match"])
        self.assertEqual(report["first_divergence_level"], "OUTPUT")

    def test_dict_traces_coerced(self) -> None:
        a = _trace().to_dict()
        b = _trace().to_dict()
        self.assertTrue(traces_equal(a, b))

    def test_canonical_request_fingerprint_mismatch(self) -> None:
        a = _trace(model="model-a")
        b = _trace(model="model-b")
        report = find_first_divergence(a, b)
        self.assertEqual(report["first_divergence_level"], "REQUEST")
        self.assertTrue(
            any(d["aspect"] == "canonical_request" for d in report["differences"])
        )


if __name__ == "__main__":
    unittest.main()
