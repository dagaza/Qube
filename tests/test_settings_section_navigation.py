"""Tests for settings section registry and navigation resolution."""

from __future__ import annotations

import unittest

from ui.views.settings.registry import (
    SETTINGS_SECTIONS,
    get_section,
    resolve_section_id,
)


class SettingsRegistryTests(unittest.TestCase):
    def test_resolve_by_stable_id(self) -> None:
        self.assertEqual(resolve_section_id("ai.models"), "ai.models")

    def test_resolve_by_display_title(self) -> None:
        self.assertEqual(resolve_section_id("Voice & Audio"), "voice.audio")

    def test_resolve_legacy_titles(self) -> None:
        self.assertEqual(resolve_section_id("AI MODELS & ROUTING"), "ai.models")
        self.assertEqual(resolve_section_id("NATIVE ENGINE & LOCAL LIBRARY"), "ai.models")
        self.assertEqual(resolve_section_id("NLP RAG TRIGGERS"), "knowledge")
        self.assertEqual(resolve_section_id("Memory & Knowledge"), "memory")
        self.assertEqual(resolve_section_id("JSON SETTINGS"), "advanced")
        self.assertEqual(resolve_section_id("HELP & GUIDANCE"), "help")
        self.assertEqual(resolve_section_id("CONTACT & FEEDBACK"), "contact.feedback")

    def test_eleven_sections_registered(self) -> None:
        self.assertEqual(len(SETTINGS_SECTIONS), 11)

    def test_themes_section_registered(self) -> None:
        sec = get_section("appearance.themes")
        self.assertIsNotNone(sec)
        assert sec is not None
        self.assertEqual(sec.title, "Themes")
        self.assertEqual(resolve_section_id("Themes"), "appearance.themes")

    def test_support_group_sections(self) -> None:
        support = [s for s in SETTINGS_SECTIONS if s.group == "Support"]
        self.assertEqual([s.id for s in support], ["help", "contact.feedback"])

    def test_get_section_returns_def(self) -> None:
        sec = get_section("notifications")
        self.assertIsNotNone(sec)
        assert sec is not None
        self.assertEqual(sec.title, "Notifications")


if __name__ == "__main__":
    unittest.main()
