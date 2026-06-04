"""Tests for streaming Harmony final-channel parser."""
from __future__ import annotations

import unittest

from core.harmony_stream_parser import HarmonyStreamParser
from core.output_artifact_strip import strip_harmony_oss_artifacts


class TestHarmonyStreamParser(unittest.TestCase):
    def test_emits_final_channel_text(self) -> None:
        p = HarmonyStreamParser()
        out = p.feed("The sky is blue because Rayleigh scattering.")
        self.assertEqual(out, "The sky is blue because Rayleigh scattering.")

    def test_suppresses_analysis_channel(self) -> None:
        p = HarmonyStreamParser()
        out = []
        out.append(p.feed("We need to plan. "))
        out.append(p.feed("<|channel|>analysis"))
        out.append(p.feed(" hidden reasoning "))
        out.append(p.feed("<|channel|>final"))
        out.append(p.feed("Visible answer."))
        self.assertEqual("".join(out), "Visible answer.")

    def test_split_control_token_across_chunks(self) -> None:
        p = HarmonyStreamParser()
        parts = [
            "Hello ",
            "<|chan",
            "nel|>analysis",
            " secret ",
            "<|channel|>fi",
            "nal",
            " world",
        ]
        emitted = "".join(p.feed(x) for x in parts)
        self.assertEqual(emitted, "Hello  world")

    def test_log_derived_bridge_then_final(self) -> None:
        p = HarmonyStreamParser()
        raw = (
            "We need to explain. Provide concise."
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            "The sky appears blue."
        )
        emitted = p.feed(raw)
        self.assertNotIn("We need to", emitted)
        self.assertIn("The sky appears blue.", emitted)

    def test_no_tail_flash_planning_before_channel_switch(self) -> None:
        """Planning in analysis must not reach UI stream emission."""
        p = HarmonyStreamParser()
        chunks = [
            "Let's clarify. The user says MCP is not Microsoft. ",
            "<|channel|>analysis",
            " more planning ",
            "<|channel|>final",
            "MCP here means Model Context Protocol.",
        ]
        streamed = "".join(p.feed(c) for c in chunks)
        self.assertNotIn("Let's clarify", streamed)
        self.assertNotIn("The user says", streamed)
        self.assertIn("Model Context Protocol", streamed)

    def test_malformed_channel_stripped_in_final(self) -> None:
        p = HarmonyStreamParser()
        out = p.feed("Hello\n<|channel>thought tail")
        self.assertEqual(out, "Hello\n")

    def test_mutes_mid_stream_question_says_meta(self) -> None:
        p = HarmonyStreamParser()
        chunks = [
            "Birds bathe to clean feathers, much like us. ",
            "The question says: \"Why do birds bathe?\". ",
            "We have to answer in natural language.",
        ]
        streamed = "".join(p.feed(c) for c in chunks)
        self.assertIn("Birds bathe", streamed)
        self.assertNotIn("The question says", streamed)
        self.assertNotIn("We have to answer", streamed)
        self.assertTrue(p.final_muted)

    def test_mutes_we_punctuation_loop(self) -> None:
        p = HarmonyStreamParser()
        out = p.feed("Good answer. We...........")
        self.assertEqual(out, "Good answer. ")
        self.assertTrue(p.final_muted)

    def test_backstop_strip_after_parser(self) -> None:
        p = HarmonyStreamParser()
        streamed = p.feed("Answer. We need to answer more.")
        cleaned = strip_harmony_oss_artifacts(streamed)
        self.assertIn("Answer", cleaned)
        self.assertNotIn("We need to", cleaned)
