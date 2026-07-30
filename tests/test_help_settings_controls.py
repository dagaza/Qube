"""Tests for generated settings controls extraction."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_settings_controls import (
    extract_settings_controls,
    generate_all_settings_controls,
    generate_settings_controls_markdown,
    settings_section_slug,
)
from ui.views.settings.registry import SETTINGS_SECTIONS


class HelpSettingsControlsTests(unittest.TestCase):
    def test_all_sections_have_controls_fragments(self) -> None:
        fragments = generate_all_settings_controls()
        self.assertEqual(len(fragments), len(SETTINGS_SECTIONS))
        for section in SETTINGS_SECTIONS:
            slug = settings_section_slug(section.id)
            self.assertIn(f"controls/{slug}.md", fragments)

    def test_ai_models_includes_gpu_layers(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("ai.models")]
        self.assertIn("GPU offload layers", labels)

    def test_notifications_includes_dnd(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("notifications")]
        joined = "\n".join(labels)
        self.assertIn("Do Not Disturb", joined)

    def test_memory_includes_enrichment_toggle(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("memory")]
        self.assertIn("Enable Memory Enrichment & Reflection (Requires more RAM)", labels)

    def test_knowledge_includes_library_search_phrases(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("knowledge")]
        self.assertIn("Enable Local Knowledge Base", labels)
        self.assertIn("Enable NLP Auto-Activator", labels)

    def test_knowledge_includes_library_pro_depth(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("knowledge")]
        self.assertIn("Default precision ingest on import", labels)
        self.assertIn("Precision retrieval", labels)

    def test_generated_markdown_banner(self) -> None:
        text = generate_settings_controls_markdown("memory")
        self.assertIn("GENERATED CONTROLS", text)
        self.assertIn("Reset to default configuration", text)

    def test_themes_includes_per_card_wallpaper_actions(self) -> None:
        from core.help_settings_controls import _read_section_sources, extract_settings_controls

        source = _read_section_sources("appearance.themes")
        self.assertIn("themes_library_apply_btn", source)
        self.assertIn("themes_library_revert_btn", source)
        self.assertNotIn("Same as Chat", source)
        labels = [entry.label for entry in extract_settings_controls("appearance.themes")]
        self.assertIn("Apply", labels)
        self.assertIn("Revert", labels)

    def test_themes_includes_theme_pack_actions(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("appearance.themes")]
        self.assertIn("Import theme pack…", labels)
        self.assertIn("Export theme pack…", labels)

    def test_help_uninstall_labels_stable_across_platforms(self) -> None:
        labels = [entry.label for entry in extract_settings_controls("help")]
        self.assertIn("Remove Qube package only… (Linux)", labels)
        self.assertIn("Remove Qube app only… (macOS)", labels)
        self.assertNotIn("Remove Qube package only…", labels)


if __name__ == "__main__":
    unittest.main()
