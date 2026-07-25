"""Tests for lazy Settings section construction."""

from __future__ import annotations

from PyQt6.QtCore import Qt

from ui.views.settings.registry import SETTINGS_SECTIONS


def test_settings_lazy_sections_voice_audio_is_active_on_open(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    assert "voice.audio" in settings._sections_built
    assert settings._settings_active_section_id == "voice.audio"
    assert settings.settings_section_stack.currentIndex() == settings._section_stack_index_by_id[
        "voice.audio"
    ]

    settings.select_settings_section("memory")
    qtbot.wait(10)
    assert "memory" in settings._sections_built


def test_settings_rag_toolbar_controls_exist_without_knowledge(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    assert settings.rag_kb_cb is not None
    assert settings.auto_activator_cb is not None


def test_settings_prefetch_builds_remaining_sections(main_window, qtbot):
    from PyQt6.QtWidgets import QApplication

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    for _ in range(len(SETTINGS_SECTIONS) * 2):
        QApplication.processEvents()
        if len(settings._sections_built) == len(SETTINGS_SECTIONS):
            break

    assert settings._sections_built == {sec.id for sec in SETTINGS_SECTIONS}
