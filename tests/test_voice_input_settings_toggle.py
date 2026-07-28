"""Settings Enable Voice Input toggle stays wired to the toolbar control."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt


@pytest.fixture(autouse=True)
def _stub_voice_input_bootstrap_prompt(monkeypatch):
    """Prevent the voice input enable toggle from opening a blocking download modal."""
    monkeypatch.setattr(
        "ui.bootstrap_feature_prompts.ensure_bootstrap_model_downloaded",
        lambda *args, **kwargs: True,
        raising=False,
    )


@pytest.mark.ui
def test_settings_voice_input_enable_toggle_syncs_with_toolbar(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None

    toolbar_toggle = main_window.voice_input_toggle
    settings_toggle = settings.voice_input_enabled_toggle
    assert settings_toggle.isChecked() == toolbar_toggle.isChecked()

    with qtbot.waitSignal(settings_toggle.toggled, timeout=1000):
        settings_toggle.setChecked(not toolbar_toggle.isChecked())
    assert toolbar_toggle.isChecked() == settings_toggle.isChecked()

    with qtbot.waitSignal(toolbar_toggle.toggled, timeout=1000):
        toolbar_toggle.setChecked(not settings_toggle.isChecked())
    assert settings_toggle.isChecked() == toolbar_toggle.isChecked()
