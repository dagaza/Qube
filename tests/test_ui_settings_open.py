"""UI smoke tests for settings navigation."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt


@pytest.mark.ui
def test_main_stages_are_lazy_until_navigation(main_window, qtbot):
    assert main_window._main_stage_built == {0}
    assert main_window._library_view is None
    assert main_window._settings_view is None
    assert main_window.main_stage.count() == 6

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    assert 5 in main_window._main_stage_built
    assert main_window._settings_view is not None
    assert main_window.main_stage.currentWidget() is main_window._settings_view


@pytest.mark.ui
def test_settings_view_builds_from_main_window(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    assert main_window.main_stage.currentWidget() is settings


@pytest.mark.ui
def test_settings_search_filters_to_voice_section(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
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
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
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
