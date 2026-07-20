"""Tests for composer @ mention user guide content."""

from __future__ import annotations

import unittest

from core.composer_mention_guide import build_composer_mention_guide_text
from core.skills.registry import iter_skills


class ComposerMentionGuideTests(unittest.TestCase):
    def test_guide_includes_core_sections(self) -> None:
        text = build_composer_mention_guide_text()
        self.assertIn("COMPOSER @ GUIDE", text)
        self.assertIn("MIXING CAPABILITIES", text)
        self.assertIn("search everything", text)
        self.assertIn("@[tool:internet]", text)
        self.assertIn("@[skill:decision_analysis]", text)
        self.assertIn("first one in your message", text)

    def test_guide_lists_all_builtin_skills(self) -> None:
        text = build_composer_mention_guide_text()
        for skill in iter_skills():
            self.assertIn(f"@[skill:{skill.id}]", text)

    def test_guide_mentions_settings_help_path(self) -> None:
        text = build_composer_mention_guide_text()
        self.assertIn("Settings → Help", text)

    def test_guide_documents_advanced_and_source_pins(self) -> None:
        text = build_composer_mention_guide_text()
        self.assertIn("Advanced tools", text)
        self.assertIn("@[tool:source:", text)
        self.assertIn("My knowledge presets", text)
        self.assertIn("first among @[file:…]", text)

if __name__ == "__main__":
    unittest.main()
