"""Tests for Memory settings simple vs advanced disclosure (Phase 0.2)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core import app_settings
from ui.views.settings.handlers.memory import MemoryHandlersMixin


class _MemoryHost(MemoryHandlersMixin):
    def __init__(self) -> None:
        from PyQt6.QtWidgets import QWidget

        from ui.components.toggle import PrestigeToggle

        self.advanced_memory_panel = QWidget()
        self.advanced_memory_toggle = PrestigeToggle()
        self._tour_memory_preview_active = False
        self._tour_memory_row_preview_active = False


class TestMemorySettingsSimpleMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_advanced_memory_unlocked_defaults_false(self) -> None:
        with patch.object(
            app_settings,
            "_store",
            return_value=MagicMock(get=MagicMock(return_value=False)),
        ):
            self.assertFalse(app_settings.get_advanced_memory_unlocked())

    def test_panel_hidden_when_locked_and_not_in_tour(self) -> None:
        host = _MemoryHost()
        with patch(
            "ui.views.settings.handlers.memory.get_advanced_memory_unlocked",
            return_value=False,
        ):
            host._apply_advanced_memory_panel_visibility()
        self.assertTrue(host.advanced_memory_panel.isHidden())
        self.assertFalse(host.advanced_memory_toggle.isChecked())

    def test_panel_visible_when_unlocked(self) -> None:
        host = _MemoryHost()
        with patch(
            "ui.views.settings.handlers.memory.get_advanced_memory_unlocked",
            return_value=True,
        ):
            host._apply_advanced_memory_panel_visibility()
        self.assertFalse(host.advanced_memory_panel.isHidden())
        self.assertTrue(host.advanced_memory_toggle.isChecked())

    def test_tour_preview_reveals_panel_without_persisting_unlock(self) -> None:
        host = _MemoryHost()
        with patch(
            "ui.views.settings.handlers.memory.get_advanced_memory_unlocked",
            return_value=False,
        ):
            host.begin_memory_advanced_tutorial_preview()
            self.assertFalse(host.advanced_memory_panel.isHidden())
            host.end_memory_advanced_tutorial_preview()
            self.assertTrue(host.advanced_memory_panel.isHidden())


if __name__ == "__main__":
    unittest.main()
