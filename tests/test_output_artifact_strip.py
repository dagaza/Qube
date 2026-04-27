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

    def test_strips_provide_final_answer_prefix(self) -> None:
        raw = "Provide final answer\nThe sky is blue because air scatters blue light."
        out = strip_harmony_oss_artifacts(raw)
        self.assertEqual(out, "The sky is blue because air scatters blue light.")
