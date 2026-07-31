"""Settings TTS voice selector stays wired when TTS loads before Settings opens."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt


def _stop_settings_section_prefetch(settings) -> None:
    queue = getattr(settings, "_settings_prefetch_queue", None)
    if queue is not None:
        queue.clear()


def _reset_settings_lazy_state(main_window, qtbot) -> None:
    """Drop a built Settings stage so tests can exercise first-open wiring."""
    from PyQt6.QtWidgets import QWidget

    from ui.main_window import MAIN_STAGE_SETTINGS

    main_window._settings_view_wired = False
    if MAIN_STAGE_SETTINGS not in main_window._main_stage_built:
        main_window._settings_view = None
        return

    widget = main_window.main_stage.widget(MAIN_STAGE_SETTINGS)
    if widget is not None:
        release = getattr(widget, "_release_themes_manager_subscription", None)
        if callable(release):
            release()
        stop_watcher = getattr(widget, "_teardown_settings_file_watcher", None)
        if callable(stop_watcher):
            stop_watcher()
        main_window.main_stage.removeWidget(widget)
        widget.deleteLater()
    main_window._settings_view = None
    main_window._main_stage_built.discard(MAIN_STAGE_SETTINGS)
    main_window._main_stage_app_wired.discard(MAIN_STAGE_SETTINGS)

    placeholder = QWidget()
    placeholder.setObjectName(f"MainStagePlaceholder_{MAIN_STAGE_SETTINGS}")
    main_window.main_stage.insertWidget(MAIN_STAGE_SETTINGS, placeholder)
    qtbot.wait(10)


@pytest.fixture(autouse=True)
def _stub_tts_bootstrap_prompt(monkeypatch):
    """Prevent the TTS enable toggle from opening a blocking download modal.

    Toggling voice output on when no TTS model is on disk (the CI state) calls
    ``ensure_bootstrap_model_downloaded``, which runs ``PrestigeDialog.exec()``.
    A modal exec blocks the event loop forever in headless CI, hanging the job
    until the 6-hour cap. Pretend a model is already available so the handler
    never shows the prompt.
    """
    monkeypatch.setattr(
        "core.tts_models.any_supported_tts_model_on_disk",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        "ui.bootstrap_feature_prompts.ensure_bootstrap_model_downloaded",
        lambda *args, **kwargs: True,
        raising=False,
    )


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
    _reset_settings_lazy_state(main_window, qtbot)
    voices = ["af_heart", "af_bella", "am_adam"]
    main_window._tts_worker = MagicMock(active_voice_name="af_bella")

    main_window.update_tts_voice_dropdowns("kokoro-v1.0.onnx", voices)
    assert main_window._settings_view is None
    toolbar_labels = _prestige_menu_item_labels(main_window.global_voice_selector.menu())
    assert toolbar_labels == voices

    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    _stop_settings_section_prefetch(settings)

    settings_labels = _prestige_menu_item_labels(settings.voice_selector.menu())
    assert settings_labels == voices
    assert settings.voice_selector.text() == "af_bella"


@pytest.mark.ui
def test_settings_tts_voice_enable_toggle_syncs_with_toolbar(main_window, qtbot):
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    _stop_settings_section_prefetch(settings)

    toolbar_toggle = main_window.voice_bypass_toggle
    settings_toggle = settings.tts_voice_enabled_toggle
    assert settings_toggle.isChecked() == toolbar_toggle.isChecked()

    with qtbot.waitSignal(settings_toggle.toggled, timeout=1000):
        settings_toggle.setChecked(not toolbar_toggle.isChecked())
    assert toolbar_toggle.isChecked() == settings_toggle.isChecked()

    with qtbot.waitSignal(toolbar_toggle.toggled, timeout=1000):
        toolbar_toggle.setChecked(not settings_toggle.isChecked())
    assert settings_toggle.isChecked() == toolbar_toggle.isChecked()
