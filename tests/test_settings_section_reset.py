"""Tests for settings section reset helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import app_settings
from core.settings_section_reset import (
    SECTION_SETTING_KEYS,
    reset_settings_section,
)
from core.settings_store import SettingsStore, reset_settings_store_for_tests
import core.settings_store as settings_store_module


class SettingsSectionResetTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_settings_store_for_tests()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.user_path = Path(self._tmpdir.name) / "settings.json"
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            settings_store_module._store = SettingsStore(user_path=self.user_path)

    def tearDown(self) -> None:
        reset_settings_store_for_tests()
        self._tmpdir.cleanup()

    def test_notifications_section_keys_match_page_controls(self) -> None:
        keys = set(SECTION_SETTING_KEYS["notifications"])
        self.assertIn(app_settings.KEY_NOTIFICATIONS_ENABLED, keys)
        self.assertIn(app_settings.KEY_NOTIFICATIONS_CATEGORY_MEMORY, keys)
        self.assertNotIn(app_settings.KEY_NOTIFICATIONS_KEEP_HISTORY, keys)

    def test_reset_notifications_restores_schema_defaults(self) -> None:
        app_settings.set_notifications_enabled(False)
        app_settings.set_notifications_dnd(True)
        app_settings.set_notifications_category_memory(True)

        changed = reset_settings_section("notifications")

        self.assertIn(app_settings.KEY_NOTIFICATIONS_ENABLED, changed)
        self.assertTrue(app_settings.get_notifications_enabled())
        self.assertFalse(app_settings.get_notifications_dnd())
        self.assertFalse(app_settings.get_notifications_category_memory())

    def test_reset_unknown_section_raises(self) -> None:
        with self.assertRaises(ValueError):
            reset_settings_section("help")

    def test_themes_section_keys_include_appearance_and_scheme(self) -> None:
        keys = set(SECTION_SETTING_KEYS["appearance.themes"])
        self.assertIn(app_settings.KEY_UI_COLOR_SCHEME_ID, keys)
        self.assertIn(app_settings.KEY_UI_THEME_APPEARANCE, keys)
        self.assertIn(app_settings.KEY_LAST_SCHEME_DARK, keys)


if __name__ == "__main__":
    unittest.main()
