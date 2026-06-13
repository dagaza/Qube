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
def test_settings_search_filters_to_voice_section(main_window, qtbot):
    settings = main_window.settings_view
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    search = settings.settings_search_input
    qtbot.keyClicks(search, "wakeword")
    assert not settings.settings_section_list.item(
        settings._section_row_by_id["voice.audio"]
    ).isHidden()
    assert settings.settings_section_list.item(
        settings._section_row_by_id["ai.models"]
    ).isHidden()


@pytest.mark.ui
def test_select_settings_section_by_id(main_window, qtbot):
    settings = main_window.settings_view
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings.select_settings_section("memory")
    assert (
        settings.settings_section_list.currentRow()
        == settings._section_row_by_id["memory"]
    )
    settings.select_settings_section("knowledge")
    assert (
        settings.settings_section_list.currentRow()
        == settings._section_row_by_id["knowledge"]
    )
    settings.select_settings_section("AI MODELS & ROUTING")
    assert (
        settings.settings_section_list.currentRow()
        == settings._section_row_by_id["ai.models"]
    )


@pytest.mark.ui
def test_conversations_view_is_default_stack_page(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_chat, Qt.MouseButton.LeftButton)
    assert main_window.main_stage.currentWidget() is main_window.conversations_view
