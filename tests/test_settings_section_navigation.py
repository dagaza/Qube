"""Tests for settings section registry and navigation resolution."""

from __future__ import annotations

import unittest

from ui.views.settings.registry import (
    SETTINGS_SECTIONS,
    get_section,
    resolve_section_id,
    resolve_settings_navigation,
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
        self.assertEqual(resolve_section_id("DIAGNOSTIC LOGS"), "diagnostics")
        self.assertEqual(resolve_section_id("PRIVACY & DATA"), "privacy.data")
        self.assertEqual(resolve_section_id("LICENSE"), "license")
        self.assertEqual(resolve_section_id("HELP & GUIDANCE"), "help")
        self.assertEqual(resolve_section_id("CONTACT & FEEDBACK"), "contact.feedback")

    def test_fifteen_sections_registered(self) -> None:
        self.assertEqual(len(SETTINGS_SECTIONS), 15)

    def test_system_group_sections(self) -> None:
        system = [s for s in SETTINGS_SECTIONS if s.group == "System"]
        self.assertEqual(
            [s.id for s in system],
            ["privacy.data", "diagnostics", "license", "advanced"],
        )

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

    def test_legacy_advanced_anchor_redirects(self) -> None:
        self.assertEqual(
            resolve_settings_navigation("advanced", anchor="logs"),
            ("diagnostics", "logs"),
        )
        self.assertEqual(
            resolve_settings_navigation("advanced", anchor="license"),
            ("license", "license"),
        )
        self.assertEqual(
            resolve_settings_navigation("advanced", anchor="routing_debug"),
            ("privacy.data", "routing_debug"),
        )
        self.assertEqual(
            resolve_settings_navigation("advanced", anchor="web_search_audit"),
            ("privacy.data", "web_search_audit"),
        )
        self.assertEqual(
            resolve_settings_navigation("advanced", anchor="llm_debug"),
            ("privacy.data", "llm_debug"),
        )
        self.assertEqual(
            resolve_settings_navigation("advanced", anchor="json"),
            ("advanced", "json"),
        )

    def test_non_advanced_navigation_unchanged(self) -> None:
        self.assertEqual(
            resolve_settings_navigation("diagnostics", anchor="app_log"),
            ("diagnostics", "app_log"),
        )


if __name__ == "__main__":
    unittest.main()
