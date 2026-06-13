"""Tests for Gemma 4 thought-channel and instruction-echo stripping."""
from __future__ import annotations

import unittest

from core.gemma_output_strip import (
    GemmaThoughtStreamFilter,
    is_gemma_model_identity,
    strip_gemma_output_artifacts,
)


class TestGemmaOutputStrip(unittest.TestCase):
    def test_is_gemma_model_identity(self) -> None:
        self.assertTrue(
            is_gemma_model_identity(
                model_name="Gemma-4-12B-It",
                model_path="/models/gemma-4-12b-it-Q5_K_M.gguf",
            )
        )
        self.assertFalse(is_gemma_model_identity(model_name="gpt-oss-20b"))

    def test_strips_log_derived_instruction_echo_and_keeps_thought_body(self) -> None:
        raw = (
            " Do not include preamble, planning, or meta commentary. "
            "Do not restate or analyze the user's request. "
            "Write only what the user should see. "
            "Keep the response natural and focused.\n"
            "<|channel>thought\n"
            "<channel|>The population of Kathmandu varies depending on the area measured."
        )
        out = strip_gemma_output_artifacts(raw)
        self.assertNotIn("Do not include preamble", out)
        self.assertIn("population of Kathmandu", out)

    def test_instruction_echo_only_returns_empty(self) -> None:
        raw = (
            "Do not include preamble, planning, or meta commentary. "
            "Do not restate or analyze the user's request. "
            "Write only what the user should see. "
            "Keep the response natural and focused."
        )
        self.assertEqual(strip_gemma_output_artifacts(raw), "")

    def test_idempotent_on_clean_text(self) -> None:
        t = "The capital of Nepal is Kathmandu."
        self.assertEqual(strip_gemma_output_artifacts(t), t)

    def test_stream_filter_swallows_echo_then_emits_thought_body(self) -> None:
        f = GemmaThoughtStreamFilter()
        parts = [
            "Do not include preamble, planning, or meta commentary. ",
            "Do not restate or analyze the user's request. ",
            "Write only what the user should see. ",
            "Keep the response natural and focused.\n",
            "<|channel>thought\n",
            "<channel|>The population of Kathmandu is about 1.5 million.",
        ]
        emitted = "".join(f.feed(p) for p in parts) + f.flush()
        self.assertNotIn("Do not include preamble", emitted)
        self.assertIn("population of Kathmandu", emitted)

    def test_stream_filter_passthrough_direct_answer(self) -> None:
        f = GemmaThoughtStreamFilter()
        emitted = f.feed("The capital of Nepal is Kathmandu.") + f.flush()
        self.assertEqual(emitted, "The capital of Nepal is Kathmandu.")

    def test_stream_filter_swallows_bare_thought_label_from_anchor(self) -> None:
        f = GemmaThoughtStreamFilter()
        parts = [
            "thought",
            "\n",
            "<channel|>Since you aren't experiencing itching, redness, or pain.",
        ]
        emitted = "".join(f.feed(p) for p in parts) + f.flush()
        self.assertNotIn("thought", emitted.lower())
        self.assertIn("Since you aren't experiencing", emitted)

    def test_strips_bare_thought_label_before_channel_body(self) -> None:
        raw = (
            "thought\n"
            "<channel|>Since you aren't experiencing itching, redness, or pain."
        )
        out = strip_gemma_output_artifacts(raw)
        self.assertNotIn("thought", out.lower())
        self.assertIn("Since you aren't experiencing", out)


if __name__ == "__main__":
    unittest.main()
