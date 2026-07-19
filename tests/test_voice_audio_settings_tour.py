"""Voice & Audio settings tour — advanced panel preview and coverage."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ui.onboarding.tour_registry import build_tour


class TestVoiceAudioSettingsTour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self):
        from PyQt6.QtWidgets import QCheckBox, QFrame, QLabel, QListWidget, QPushButton, QWidget

        from ui.components.toggle import PrestigeToggle

        host = QWidget()
        sv = QWidget(host)
        host.settings_view = sv
        host.nav_settings = QPushButton(host)
        host.ensure_settings_view = MagicMock(return_value=sv)
        host._route_view = MagicMock()
        sv.select_settings_section = MagicMock()

        attrs = {
            "mic_selector": QPushButton(sv),
            "device_selector": QPushButton(sv),
            "tts_voice_enabled_toggle": PrestigeToggle(sv),
            "voice_selector": QPushButton(sv),
            "wakeword_selector": QPushButton(sv),
            "wakeword_download_open_btn": QPushButton(sv),
            "wakeword_download_community_btn": QPushButton(sv),
            "wakeword_test_lab_btn": QPushButton(sv),
            "timeout_spinner": QPushButton(sv),
            "threshold_spinner": QPushButton(sv),
            "pin_audio_cb": QCheckBox(sv),
            "pin_tts_voice_cb": QCheckBox(sv),
            "advanced_stt_toggle": PrestigeToggle(sv),
            "advanced_tts_toggle": PrestigeToggle(sv),
            "stt_model_list": QListWidget(sv),
            "use_stt_model_btn": QPushButton(sv),
            "reset_stt_model_btn": QPushButton(sv),
            "refresh_stt_model_btn": QPushButton(sv),
            "delete_stt_model_btn": QPushButton(sv),
            "active_stt_model_lbl": QLabel(sv),
            "tts_model_list": QListWidget(sv),
            "use_tts_model_btn": QPushButton(sv),
            "reset_tts_model_btn": QPushButton(sv),
            "refresh_tts_model_btn": QPushButton(sv),
            "delete_tts_model_btn": QPushButton(sv),
            "active_tts_model_lbl": QLabel(sv),
        }
        for name, widget in attrs.items():
            widget.show()
            setattr(sv, name, widget)

        stt_panel = QFrame(sv)
        tts_panel = QFrame(sv)
        stt_panel.hide()
        tts_panel.hide()
        sv.advanced_stt_panel = stt_panel
        sv.advanced_tts_panel = tts_panel
        sv._tour_stt_preview_active = False
        sv._tour_tts_preview_active = False

        def begin_stt() -> None:
            sv._tour_stt_preview_active = True
            stt_panel.show()

        def end_stt() -> None:
            sv._tour_stt_preview_active = False
            stt_panel.hide()

        def begin_tts() -> None:
            sv._tour_tts_preview_active = True
            tts_panel.show()

        def end_tts() -> None:
            sv._tour_tts_preview_active = False
            tts_panel.hide()

        sv.begin_voice_audio_stt_tutorial_preview = begin_stt
        sv.end_voice_audio_stt_tutorial_preview = end_stt
        sv.begin_voice_audio_tts_tutorial_preview = begin_tts
        sv.end_voice_audio_tts_tutorial_preview = end_tts
        return host

    def test_stt_and_tts_steps_reveal_hidden_panels(self) -> None:
        host = self._make_host()
        host.show()
        self._app.processEvents()
        tour = build_tour("settings.voice_audio", host)
        assert tour is not None

        stt_steps = {
            "stt_models",
            "stt_use",
            "stt_reset",
            "stt_refresh",
            "stt_delete",
            "active_stt",
        }
        tts_steps = {
            "tts_models",
            "tts_use",
            "tts_reset",
            "tts_refresh",
            "tts_delete",
            "active_tts",
        }

        for step in tour._steps:
            if step.on_enter is not None:
                step.on_enter(host)
            if step.target_getter is None:
                continue
            target = step.target_getter(host)
            self.assertIsNotNone(target, step.step_id)
            if step.step_id in stt_steps:
                self.assertTrue(host.settings_view.advanced_stt_panel.isVisible(), step.step_id)
            if step.step_id in tts_steps:
                self.assertTrue(host.settings_view.advanced_tts_panel.isVisible(), step.step_id)

    def test_preview_end_restores_hidden_panels(self) -> None:
        from PyQt6.QtWidgets import QFrame

        from ui.components.toggle import PrestigeToggle

        class FakeSettings:
            def __init__(self) -> None:
                self.advanced_stt_panel = QFrame()
                self.advanced_tts_panel = QFrame()
                self.advanced_stt_toggle = PrestigeToggle()
                self.advanced_tts_toggle = PrestigeToggle()
                self.advanced_stt_panel.hide()
                self.advanced_tts_panel.hide()
                self._tour_stt_preview_active = False
                self._tour_tts_preview_active = False

            def _apply_advanced_stt_panel_visibility(self) -> None:
                from core.app_settings import get_advanced_stt_unlocked

                unlocked = get_advanced_stt_unlocked()
                visible = unlocked or self._tour_stt_preview_active
                self.advanced_stt_panel.setVisible(visible)
                self.advanced_stt_toggle.blockSignals(True)
                self.advanced_stt_toggle.setChecked(
                    True if self._tour_stt_preview_active else unlocked
                )
                self.advanced_stt_toggle.blockSignals(False)

            def _apply_advanced_tts_panel_visibility(self) -> None:
                from core.app_settings import get_advanced_tts_unlocked

                unlocked = get_advanced_tts_unlocked()
                visible = unlocked or self._tour_tts_preview_active
                self.advanced_tts_panel.setVisible(visible)
                self.advanced_tts_toggle.blockSignals(True)
                self.advanced_tts_toggle.setChecked(
                    True if self._tour_tts_preview_active else unlocked
                )
                self.advanced_tts_toggle.blockSignals(False)

            def begin_voice_audio_stt_tutorial_preview(self) -> None:
                self._tour_stt_preview_active = True
                self._apply_advanced_stt_panel_visibility()

            def end_voice_audio_stt_tutorial_preview(self) -> None:
                if not self._tour_stt_preview_active:
                    return
                self._tour_stt_preview_active = False
                self._apply_advanced_stt_panel_visibility()

            def begin_voice_audio_tts_tutorial_preview(self) -> None:
                self._tour_tts_preview_active = True
                self._apply_advanced_tts_panel_visibility()

            def end_voice_audio_tts_tutorial_preview(self) -> None:
                if not self._tour_tts_preview_active:
                    return
                self._tour_tts_preview_active = False
                self._apply_advanced_tts_panel_visibility()

        with patch("core.app_settings.get_advanced_stt_unlocked", return_value=False), patch(
            "core.app_settings.get_advanced_tts_unlocked", return_value=False
        ):
            sv = FakeSettings()
            sv.begin_voice_audio_stt_tutorial_preview()
            self.assertTrue(sv.advanced_stt_panel.isVisible())
            sv.end_voice_audio_stt_tutorial_preview()
            self.assertFalse(sv.advanced_stt_panel.isVisible())

            sv.begin_voice_audio_tts_tutorial_preview()
            self.assertTrue(sv.advanced_tts_panel.isVisible())
            sv.end_voice_audio_tts_tutorial_preview()
            self.assertFalse(sv.advanced_tts_panel.isVisible())


if __name__ == "__main__":
    unittest.main()
