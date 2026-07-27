"""Tests for lazy Settings section construction."""

from __future__ import annotations

from PyQt6.QtCore import Qt

from core.theme.widget_styles import SETTINGS_CHECKBOX
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


def test_lazy_built_settings_checkboxes_get_prestige_style(main_window, qtbot):
    """Sections built after init must receive SETTINGS_CHECKBOX QSS (not only voice.audio)."""
    from core.theme.view_theme import view_resolved_theme

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    settings._ensure_section_built("notifications")
    cb = settings.notifications_enabled_cb
    theme = view_resolved_theme(settings)
    expected = theme.style(SETTINGS_CHECKBOX).strip()
    assert cb.styleSheet().strip() == expected

    settings._ensure_section_built("ai.models")
    llm_cb = settings.llm_output_limit_cb
    assert llm_cb.styleSheet().strip() == expected


def test_companion_commentary_checkboxes_keep_prestige_style_when_disabled(
    main_window, qtbot
):
    """Commentary checkboxes are gated by the master switch; disabled must not revert to native GTK."""
    from core.theme.view_theme import view_resolved_theme
    from core.theme.widget_styles import SETTINGS_CHECKBOX

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    settings._ensure_section_built("companion.desktop")
    theme = view_resolved_theme(settings)
    expected = theme.style(SETTINGS_CHECKBOX).strip()

    settings.companion_enabled_cb.blockSignals(True)
    settings.companion_enabled_cb.setChecked(False)
    settings.companion_enabled_cb.blockSignals(False)
    settings._sync_companion_verbal_controls_enabled()

    for name in (
        "companion_verbal_enabled_cb",
        "companion_verbal_react_ingest_cb",
    ):
        cb = getattr(settings, name)
        assert not cb.isEnabled()
        assert theme.surface_pressed in cb.styleSheet()
        assert "image: none" in cb.styleSheet()

    visibility_cb = settings.companion_tray_hidden_cb
    assert visibility_cb.isEnabled()
    assert visibility_cb.styleSheet().strip() == expected


def test_lazy_built_ai_models_gguf_list_gets_bordered_panel(main_window, qtbot):
    """Downloaded-models list must receive SETTINGS_BORDERED_LIST QSS after lazy build."""
    from core.theme.view_theme import view_resolved_theme

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    settings._ensure_section_built("ai.models")
    theme = view_resolved_theme(settings)
    sheet = settings.local_gguf_list.styleSheet()
    assert theme.background in sheet
    assert theme.border_subtle in sheet or theme.border in sheet
    assert "border-radius: 8px" in sheet
