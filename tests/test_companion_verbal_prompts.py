"""Companion verbal prompt assembly."""
from __future__ import annotations

import unittest

from core.companion_verbal_prompts import (
    COMPANION_LINE_MAX_CHARS,
    build_companion_line_prompt,
    build_companion_line_system,
    build_companion_line_user_payload,
    truncate_companion_caption,
)
from core.companion_verbal_traits import CompanionVerbalTraitPreset


class TestCompanionVerbalPrompts(unittest.TestCase):
    def test_system_includes_trait_and_user_prompt(self) -> None:
        system = build_companion_line_system(
            trait_preset=CompanionVerbalTraitPreset.WITTY,
            user_system_prompt="Prefer puns about coffee.",
        )
        self.assertIn("clever humor", system.lower())
        self.assertIn("coffee", system)

    def test_user_payload_download_context(self) -> None:
        user = build_companion_line_user_payload(
            trigger="download_complete",
            basename="model-q4.gguf",
        )
        self.assertIn("trigger: download_complete", user)
        self.assertIn("basename: model-q4.gguf", user)
        self.assertNotIn("memory", user.lower())

    def test_user_payload_test_context(self) -> None:
        user = build_companion_line_user_payload(trigger="test")
        self.assertIn("trigger: test", user)
        self.assertIn("settings preview", user.lower())

    def test_build_prompt_uses_chatml(self) -> None:
        prompt = build_companion_line_prompt(
            chat_format="chatml",
            trait_preset="neutral",
            trigger="idle",
        )
        self.assertIn("assistant", prompt.lower())
        self.assertLessEqual(COMPANION_LINE_MAX_CHARS, 72)

    def test_truncate_on_word_boundary(self) -> None:
        text = "Enjoy your Qube journey today and beyond the horizon"
        out = truncate_companion_caption(text, 40)
        self.assertLessEqual(len(out), 40)
        self.assertTrue(out.endswith("…"))
        self.assertFalse(out.rstrip("…").endswith("horiz"))
        self.assertTrue(out.startswith("Enjoy your Qube"))


if __name__ == "__main__":
    unittest.main()
