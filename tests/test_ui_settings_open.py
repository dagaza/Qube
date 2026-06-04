"""UI smoke tests for settings navigation."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt


@pytest.mark.ui
def test_settings_view_builds_from_main_window(main_window, qtbot):
    settings = main_window.settings_view
    assert settings is not None
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    assert main_window.main_stage.currentWidget() is settings


@pytest.mark.ui
def test_conversations_view_is_default_stack_page(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_chat, Qt.MouseButton.LeftButton)
    assert main_window.main_stage.currentWidget() is main_window.conversations_view
