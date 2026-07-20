"""Tests for @help action block parsing."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_action_blocks import parse_help_action_blocks


class HelpActionBlocksTests(unittest.TestCase):
    def test_parse_open_settings_section(self) -> None:
        raw = (
            "Open Settings → AI & Models.\n\n"
            '[action:open_settings_section settings_section=ai.models label="Open AI & Models settings"]'
        )
        stripped, actions = parse_help_action_blocks(raw)
        self.assertNotIn("[action:", stripped)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].settings_section, "ai.models")
        self.assertEqual(actions[0].label, "Open AI & Models settings")

    def test_unknown_action_kind_left_intact(self) -> None:
        raw = "[action:open_page_tour tour_id=settings.knowledge]"
        stripped, actions = parse_help_action_blocks(raw)
        self.assertIn("open_page_tour", stripped)
        self.assertEqual(actions, [])

    def test_default_label_when_missing(self) -> None:
        raw = "[action:open_settings_section settings_section=knowledge]"
        _stripped, actions = parse_help_action_blocks(raw)
        self.assertEqual(actions[0].label, "Open Knowledge settings")


if __name__ == "__main__":
    unittest.main()
