"""Harmony-aware prompt integrity validation."""
from __future__ import annotations

import unittest

from core.harmony_renderer import render_harmony_final_prompt
from core.prompt_integrity_validator import validate_chat_inference


class TestHarmonyPromptValidation(unittest.TestCase):
    def test_harmony_rendered_prompt_is_ok_not_broken(self) -> None:
        prompt = render_harmony_final_prompt(
            [
                {"role": "system", "content": "Be direct."},
                {"role": "user", "content": "Why is the sky blue?"},
            ]
        )
        result = validate_chat_inference(
            rendered_prompt=prompt,
            messages=[],
            chat_format="",
            merged_stop_tokens=["<|return|>"],
            eos_token_str="",
            model_metadata={},
            reconstruction_ok=True,
        )
        self.assertEqual(result.verdict, "OK")
        self.assertTrue(result.assistant_anchor_present)
        self.assertTrue(result.user_message_closed_properly)
        self.assertFalse(result.stop_tokens_suspicious)
