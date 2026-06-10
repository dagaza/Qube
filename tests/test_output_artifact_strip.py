"""Tests for Harmony / OSS completion artifact stripping."""
from __future__ import annotations

import unittest

from core.output_artifact_strip import strip_harmony_oss_artifacts


class TestStripHarmonyOssArtifacts(unittest.TestCase):
    def test_strips_log_derived_bridge_and_preface(self) -> None:
        raw = (
            "The sky is blue because... We need to explain why? The user asks: "
            '"Why is the sky blue?" Provide explanation: Rayleigh scattering, shorter wavelengths. '
            "Provide concise.<|end|><|start|>assistant<|channel|>final<|message|>"
            "The sky appears blue because clean tail."
        )
        out = strip_harmony_oss_artifacts(raw)
        self.assertNotIn("<|channel|>", out)
        self.assertNotIn("<|end|>", out)
        self.assertNotIn("We need to explain", out)
        self.assertIn("The sky is blue because", out)
        self.assertIn("clean tail", out)

    def test_idempotent_on_clean_text(self) -> None:
        t = "Rayleigh scattering explains blue skies."
        self.assertEqual(strip_harmony_oss_artifacts(t), t)

    def test_strips_question_says_meta_tail(self) -> None:
        raw = (
            "Birds bathe to stay clean. "
            'The question says: "Why do birds bathe?". '
            "We have to answer in natural language, no meta commentary."
        )
        out = strip_harmony_oss_artifacts(raw)
        self.assertEqual(out, "Birds bathe to stay clean.")

    def test_strips_untagged_scratchpad_tail(self) -> None:
        raw = (
            "The sky's blue hue comes from Rayleigh scattering?..????...? "
            "We need to answer: why is sky blue? Provide explanation. "
            "We should produce concise answer."
        )
        out = strip_harmony_oss_artifacts(raw)
        self.assertIn("Rayleigh scattering", out)
        self.assertNotIn("We need to", out)
        self.assertNotIn("We should", out)
        self.assertNotRegex(out, r"[?.!…]{3,}\s*$")

    def test_strips_source_planning_tail(self) -> None:
        raw = (
            "Dr. Evelyn is Dr. Evelyn Vance. "
            "We have sources. Source 1 indicates user has a file mentioning Dr. Evelyn Vance. "
            "We must answer citing sources. Let's produce answer."
        )
        out = strip_harmony_oss_artifacts(raw)
        self.assertEqual(out, "Dr. Evelyn is Dr. Evelyn Vance.")

    def test_strips_lets_clarify_planning_preface(self) -> None:
        raw = (
            "Let's clarify that. The user says “MCP” is not Microsoft "
            "………………………………………………….."
        )
        out = strip_harmony_oss_artifacts(raw)
        self.assertNotIn("Let's clarify", out)
        self.assertNotIn("The user says", out)
        self.assertEqual(out.strip(), "")

    def test_strips_spaced_punctuation_degenerate_tail(self) -> None:
        raw = (
            "Got it—he wasn’t talking about “Microsoft Cloud Platform” at all. "
            "In that context “MCP” is just another way of saying “the … … "
            "……‑…‑…‑…‑……… … … ‑ …‑… … ………… … … … We’re‑…"
        )
        out = strip_harmony_oss_artifacts(raw)
        self.assertEqual(
            out,
            "Got it—he wasn’t talking about “Microsoft Cloud Platform” at all.",
        )

    def test_strips_provide_final_answer_prefix(self) -> None:
        raw = "Provide final answer\nThe sky is blue because air scatters blue light."
        out = strip_harmony_oss_artifacts(raw)
        self.assertEqual(out, "The sky is blue because air scatters blue light.")

    def test_strips_malformed_channel_tail_after_short_answer(self) -> None:
        raw = "Hello\n<|channel>thought <channel|>Hello"
        out = strip_harmony_oss_artifacts(raw)
        self.assertEqual(out, "Hello")

    def test_strips_malformed_channel_fragment_inline(self) -> None:
        raw = "Answer <channel|>thought tail"
        out = strip_harmony_oss_artifacts(raw)
        self.assertNotIn("channel", out.lower())
        self.assertIn("Answer", out)

    def test_strips_mistral_inst_markers(self) -> None:
        raw = "[/INST] Soak brown rice for 30 minutes. </s>"
        out = strip_harmony_oss_artifacts(raw)
        self.assertNotIn("[INST]", out)
        self.assertNotIn("[/INST]", out)
        self.assertNotIn("</s>", out)
        self.assertIn("Soak brown rice", out)

    def test_stream_fragments_keep_boundary_spaces(self) -> None:
        """Regression: per-delta strip must not strip() or words glue together in TTS."""
        from core.output_artifact_strip import strip_mistral_instruct_artifacts

        self.assertEqual(strip_mistral_instruct_artifacts(" world"), " world")
        self.assertEqual(strip_mistral_instruct_artifacts("Yes"), "Yes")
        combined = "".join(
            strip_harmony_oss_artifacts(p)
            for p in ("Yes", ",", " soaking", " brown", " rice", " helps.")
        )
        self.assertEqual(combined, "Yes, soaking brown rice helps.")
