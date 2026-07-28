"""UI smoke tests for settings navigation."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt


@pytest.mark.ui
def test_main_stages_are_lazy_until_navigation(fresh_main_window, qtbot):
    main_window = fresh_main_window
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
def test_settings_section_header_updates_tour_button(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    btn = settings.settings_section_tour_btn
    title = settings.settings_section_title_lbl

    settings.select_settings_section("voice.audio")
    assert title.text() == "Voice & Audio"
    assert btn.tour_id == "settings.voice_audio"
    assert btn.isEnabled()

    settings.select_settings_section("memory")
    assert title.text() == "Memory"
    assert btn.tour_id == "settings.memory"

    settings.select_settings_section("companion.desktop")
    assert title.text() == "Desktop Companion"
    assert btn.tour_id == "settings.companion_desktop"


@pytest.mark.ui
def test_all_settings_section_tour_targets_on_real_view(main_window, qtbot):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from ui.onboarding.tour_registry import build_tour

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    tour_ids = [
        "settings.voice_audio",
        "settings.ai_models",
        "settings.memory",
        "settings.knowledge",
        "settings.general",
        "settings.companion_desktop",
        "settings.notifications",
        "settings.help",
        "settings.contact_feedback",
        "settings.advanced",
    ]
    for tour_id in tour_ids:
        tour = build_tour(tour_id, main_window)
        assert tour is not None
        missing: list[str] = []
        for step in tour._steps:
            if step.on_enter is not None:
                step.on_enter(main_window)
            QApplication.processEvents()
            if step.target_getter is None:
                continue
            target = step.target_getter(main_window)
            if target is None:
                missing.append(step.step_id)
        assert missing == [], f"{tour_id} missing targets: {missing}"


@pytest.mark.ui
def test_conversations_view_is_default_stack_page(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_chat, Qt.MouseButton.LeftButton)
    assert main_window.main_stage.currentWidget() is main_window.conversations_view
