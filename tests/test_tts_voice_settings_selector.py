"""Settings TTS voice selector stays wired when TTS loads before Settings opens."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt


def _prestige_menu_item_labels(menu) -> list[str]:
    for action in menu.actions():
        widget = action.defaultWidget()
        if widget is None:
            continue
        from PyQt6.QtWidgets import QListWidget

        if isinstance(widget, QListWidget):
            return [widget.item(i).text() for i in range(widget.count())]
    return []


@pytest.mark.ui
def test_settings_voice_selector_syncs_after_lazy_load(main_window, qtbot):
    voices = ["af_heart", "af_bella", "am_adam"]
    main_window._tts_worker = MagicMock(active_voice_name="af_bella")

    main_window.update_tts_voice_dropdowns("kokoro-v1.0.onnx", voices)
    assert main_window._settings_view is None
    toolbar_labels = _prestige_menu_item_labels(main_window.global_voice_selector.menu())
    assert toolbar_labels == voices

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    settings_labels = _prestige_menu_item_labels(settings.voice_selector.menu())
    assert settings_labels == voices
    assert settings.voice_selector.text() == "af_bella"


@pytest.mark.ui
def test_settings_tts_voice_enable_toggle_syncs_with_toolbar(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    toolbar_toggle = main_window.voice_bypass_toggle
    settings_toggle = settings.tts_voice_enabled_toggle
    assert settings_toggle.isChecked() == toolbar_toggle.isChecked()

    with qtbot.waitSignal(settings_toggle.toggled, timeout=1000):
        settings_toggle.setChecked(not toolbar_toggle.isChecked())
    assert toolbar_toggle.isChecked() == settings_toggle.isChecked()

    with qtbot.waitSignal(toolbar_toggle.toggled, timeout=1000):
        toolbar_toggle.setChecked(not settings_toggle.isChecked())
    assert settings_toggle.isChecked() == toolbar_toggle.isChecked()
