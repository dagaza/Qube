"""Harmony system guidance merged into rendered prompts."""
from __future__ import annotations

import unittest

from core.harmony_reply_guidance import HARMONY_FINAL_REPLY_GUIDANCE
from core.harmony_renderer import render_harmony_final_prompt


class TestHarmonyReplyGuidance(unittest.TestCase):
    def test_single_turn_includes_guidance_without_user_system(self) -> None:
        prompt = render_harmony_final_prompt(
            [{"role": "user", "content": "Why do birds bathe?"}]
        )
        self.assertIn(HARMONY_FINAL_REPLY_GUIDANCE, prompt)
        self.assertIn("2–4 short sections", prompt)

    def test_merges_with_existing_system(self) -> None:
        prompt = render_harmony_final_prompt(
            [
                {"role": "system", "content": "You are Qube."},
                {"role": "user", "content": "Why do birds bathe?"},
            ]
        )
        self.assertIn("You are Qube.", prompt)
        self.assertIn("Cleaning, Temperature", prompt)
        self.assertEqual(prompt.count(HARMONY_FINAL_REPLY_GUIDANCE), 1)
