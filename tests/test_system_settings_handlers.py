"""Handler tests for Settings → System (Privacy, Diagnostics, License)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.app_settings import (
    KEY_MCP_INTERNET_HYBRID,
    KEY_WEB_SEARCH_AUDIT_REDACT_ENABLED,
)

if "qtawesome" not in sys.modules:
    sys.modules["qtawesome"] = MagicMock()

try:
    from PyQt6.QtWidgets import QApplication

    _PYQT_AVAILABLE = True
except ModuleNotFoundError:
    _PYQT_AVAILABLE = False

_HANDLERS_DIR = (
    Path(__file__).resolve().parents[1] / "ui" / "views" / "settings" / "handlers"
)


def _load_handler_module(name: str):
    """Load a handler module without importing ui.views.settings.handlers package."""
    path = _HANDLERS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_handlers_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if _PYQT_AVAILABLE:
    _privacy_mod = _load_handler_module("privacy_data")
    _licensing_mod = _load_handler_module("licensing")
    _diagnostics_mod = _load_handler_module("diagnostics")

    PrivacyDataHandlersMixin = _privacy_mod.PrivacyDataHandlersMixin
    DiagnosticsHandlersMixin = _diagnostics_mod.DiagnosticsHandlersMixin
    LicenseHandlersMixin = _licensing_mod.LicenseHandlersMixin

    class _WindowHost:
        def __init__(self, *, is_dark: bool = True) -> None:
            self._is_dark_theme = is_dark
            self.open_telemetry_focus = MagicMock()
            self._llm_worker = MagicMock()
            self.tool_internet_hybrid_toggle = None

    class _PrivacyHost(PrivacyDataHandlersMixin):
        def __init__(self) -> None:
            from PyQt6.QtWidgets import QLabel, QWidget

            from ui.components.toggle import PrestigeToggle

            self._window = QWidget()
            self._window_host = _WindowHost()
            self.diagnostic_log_redaction_toggles = {
                "web_search_audit": PrestigeToggle(self._window),
                "routing_debug": PrestigeToggle(self._window),
            }
            self.diagnostic_log_redaction_env_notes = {
                "web_search_audit": QLabel(self._window),
                "routing_debug": QLabel(self._window),
            }
            self.privacy_data_internet_hybrid_toggle = PrestigeToggle(self._window)
            self.status_messages: list[str] = []
            self.tier_sync_calls = 0
            self.section_sync_calls = 0
            self.web_discovery_policy_section = None

        def window(self):
            return self._window_host

        def _show_settings_file_status(
            self, message: str, *, persistent: bool = False
        ) -> None:
            self.status_messages.append(message)

        def _emit_external_settings_changed(self, *keys: str) -> None:
            self.emitted_keys = keys

        def _sync_discovery_privacy_tier_selector(self) -> None:
            self.tier_sync_calls += 1

        def _sync_privacy_data_section_ui(self) -> None:
            self.section_sync_calls += 1

    class _DiagnosticsHost(DiagnosticsHandlersMixin):
        def __init__(self) -> None:
            from PyQt6.QtWidgets import QWidget

            from ui.components.toggle import PrestigeToggle

            self._window = QWidget()
            self.diagnostic_log_recording_toggles = {
                "app_log": PrestigeToggle(self._window),
            }
            self.status_messages: list[str] = []

        def window(self):
            return self._window

        def _show_settings_file_status(
            self, message: str, *, persistent: bool = False
        ) -> None:
            self.status_messages.append(message)

    class _LicenseHost(LicenseHandlersMixin):
        def __init__(self) -> None:
            from PyQt6.QtWidgets import QLabel, QPushButton, QWidget

            self._window = _WindowHost()
            self.license_status_lbl = QLabel(parent=QWidget())
            self.remove_license_btn = QPushButton(parent=QWidget())
            self.library_pro_sync_calls = 0

        def window(self):
            return self._window

        def _sync_library_pro_features(self) -> None:
            self.library_pro_sync_calls += 1

    class TestPrivacyDataHandlers(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            cls._app = QApplication.instance() or QApplication([])

        @patch(f"{_privacy_mod.__name__}.set_web_search_audit_redact_enabled")
        @patch(
            f"{_privacy_mod.__name__}.get_web_search_audit_redact_enabled",
            return_value=False,
        )
        @patch(
            f"{_privacy_mod.__name__}.web_search_audit_redact_env_override",
            return_value=None,
        )
        def test_redaction_toggle_persists_web_search_setting(
            self,
            _env_mock,
            _get_mock,
            set_mock,
        ) -> None:
            host = _PrivacyHost()
            host._on_diagnostic_log_redaction_toggled("web_search_audit", True)
            set_mock.assert_called_once_with(True)
            self.assertTrue(
                any("redaction is now on" in msg for msg in host.status_messages)
            )

        @patch(
            f"{_privacy_mod.__name__}.routing_debug_log_redact_query_env_override",
            return_value=True,
        )
        def test_redaction_toggle_respects_launch_env_override(self, _env_mock) -> None:
            host = _PrivacyHost()
            toggle = host.diagnostic_log_redaction_toggles["routing_debug"]
            toggle.setEnabled(True)
            host._sync_diagnostic_log_redaction_toggle("routing_debug")
            self.assertFalse(toggle.isEnabled())
            self.assertTrue(toggle.isChecked())
            note = host.diagnostic_log_redaction_env_notes["routing_debug"]
            self.assertTrue(note.isVisible())

        def test_open_telemetry_discovery_delegates_to_main_window(self) -> None:
            host = _PrivacyHost()
            host._on_privacy_data_open_telemetry_discovery_clicked()
            host.window().open_telemetry_focus.assert_called_once_with("web_discovery")

        def test_open_telemetry_integrations_delegates_to_main_window(self) -> None:
            host = _PrivacyHost()
            host._on_privacy_data_open_telemetry_integrations_clicked()
            host.window().open_telemetry_focus.assert_called_once_with(
                "session_integrations"
            )

        def test_apply_external_privacy_settings_ignores_unrelated_keys(self) -> None:
            host = _PrivacyHost()
            host._apply_external_privacy_settings_changed({"qube.other.setting"})
            self.assertEqual(host.tier_sync_calls, 0)
            self.assertEqual(host.section_sync_calls, 0)

        def test_apply_external_privacy_settings_syncs_on_privacy_keys(self) -> None:
            host = _PrivacyHost()
            host._apply_external_privacy_settings_changed(
                {KEY_MCP_INTERNET_HYBRID, KEY_WEB_SEARCH_AUDIT_REDACT_ENABLED}
            )
            self.assertEqual(host.tier_sync_calls, 1)
            self.assertEqual(host.section_sync_calls, 1)

        @patch(f"{_privacy_mod.__name__}.get_mcp_internet_hybrid_enabled", return_value=False)
        @patch(f"{_privacy_mod.__name__}.set_mcp_internet_hybrid_enabled")
        def test_hybrid_internet_toggle_updates_worker_and_toolbar(
            self,
            set_mock,
            _get_mock,
        ) -> None:
            from ui.components.toggle import PrestigeToggle

            host = _PrivacyHost()
            host._window_host.tool_internet_hybrid_toggle = PrestigeToggle()
            host._window_host.tool_internet_hybrid_toggle.setChecked(False)
            host._on_privacy_data_internet_hybrid_toggled(True)
            set_mock.assert_called_once_with(True)
            host.window()._llm_worker.set_mcp_internet_hybrid.assert_called_once_with(
                True
            )
            self.assertTrue(host._window_host.tool_internet_hybrid_toggle.isChecked())
            self.assertEqual(host.emitted_keys, (KEY_MCP_INTERNET_HYBRID,))

    class TestDiagnosticsHandlers(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            cls._app = QApplication.instance() or QApplication([])

        @patch(f"{_diagnostics_mod.__name__}.open_logs_folder", return_value=True)
        def test_open_logs_folder_shows_success_status(self, _open_mock) -> None:
            host = _DiagnosticsHost()
            host._on_open_logs_folder_clicked()
            self.assertIn("Opened the logs folder", host.status_messages[0])

        @patch(f"{_diagnostics_mod.__name__}.open_logs_folder", return_value=False)
        @patch(f"{_diagnostics_mod.__name__}.PrestigeDialog")
        def test_open_logs_folder_failure_shows_dialog(self, dialog_cls, _open_mock) -> None:
            host = _DiagnosticsHost()
            dialog_cls.return_value.exec.return_value = 0
            host._on_open_logs_folder_clicked()
            dialog_cls.assert_called_once()
            self.assertEqual(host.status_messages, [])

        @patch(f"{_diagnostics_mod.__name__}.routing_debug_log_env_override", return_value=None)
        @patch(f"{_diagnostics_mod.__name__}.get_routing_debug_log_enabled", return_value=False)
        @patch(f"{_diagnostics_mod.__name__}.set_routing_debug_log_enabled")
        def test_routing_recording_toggle_persists_setting(
            self,
            set_mock,
            _get_mock,
            _env_mock,
        ) -> None:
            from ui.components.toggle import PrestigeToggle

            host = _DiagnosticsHost()
            host.diagnostic_log_recording_toggles["routing_debug"] = PrestigeToggle(
                host._window
            )
            host._on_diagnostic_log_recording_toggled("routing_debug", True)
            set_mock.assert_called_once_with(True)
            self.assertIn("Routing debug recording is now on", host.status_messages[-1])

    class TestLicenseHandlers(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            cls._app = QApplication.instance() or QApplication([])

        @patch(
            f"{_licensing_mod.__name__}.license_summary",
            return_value={"active": False, "cached": False},
        )
        @patch(f"{_licensing_mod.__name__}.sync_license_status_presentation")
        def test_refresh_license_status_disables_remove(
            self, sync_mock, _summary_mock
        ) -> None:
            host = _LicenseHost()
            host._refresh_license_status_ui()
            sync_mock.assert_called_once()
            self.assertFalse(host.remove_license_btn.isEnabled())

        @patch(
            f"{_licensing_mod.__name__}.license_summary",
            return_value={"active": True, "cached": True},
        )
        @patch(f"{_licensing_mod.__name__}.sync_license_status_presentation")
        def test_refresh_license_status_enables_remove_when_cached(
            self,
            sync_mock,
            _summary_mock,
        ) -> None:
            host = _LicenseHost()
            host._refresh_license_status_ui()
            sync_mock.assert_called_once()
            self.assertTrue(host.remove_license_btn.isEnabled())

        @patch(f"{_licensing_mod.__name__}.QFileDialog.getOpenFileName", return_value=("", ""))
        def test_import_license_cancelled_is_no_op(self, _dialog_mock) -> None:
            host = _LicenseHost()
            with patch(f"{_licensing_mod.__name__}.import_license_from_path") as import_mock:
                host._on_import_license_clicked()
                import_mock.assert_not_called()

        @patch(f"{_licensing_mod.__name__}.QFileDialog.getOpenFileName")
        @patch(f"{_licensing_mod.__name__}.import_license_from_path")
        @patch(f"{_licensing_mod.__name__}.PrestigeDialog")
        def test_import_license_success_refreshes_ui(
            self,
            dialog_cls,
            import_mock,
            file_dialog_mock,
        ) -> None:
            host = _LicenseHost()
            file_dialog_mock.return_value = ("/tmp/customer.qube-license", "")
            document = MagicMock()
            document.tier.value = "pro"
            import_mock.return_value = MagicMock(ok=True, document=document, error=None)
            dialog_cls.return_value.exec.return_value = 0

            with patch.object(host, "_refresh_license_status_ui") as refresh_mock:
                with patch.object(host, "_play_license_import_celebration") as celebrate_mock:
                    host._on_import_license_clicked()
                    refresh_mock.assert_called_once()
                    celebrate_mock.assert_called_once()
                    self.assertEqual(host.library_pro_sync_calls, 1)

        @patch("ui.components.celebration_burst.show_border_fireworks")
        @patch("PyQt6.QtCore.QTimer.singleShot", side_effect=lambda _ms, fn: fn())
        def test_play_license_import_celebration_uses_license_card(
            self, _timer_mock, fireworks_mock
        ) -> None:
            from PyQt6.QtWidgets import QWidget

            host = _LicenseHost()
            host.license_section_card = QWidget(host._window)
            host.license_section_card.show()
            host.settings_section_stack = MagicMock()
            host.settings_section_stack.currentWidget.return_value = host._window

            host._play_license_import_celebration()

            fireworks_mock.assert_called_once()
            kwargs = fireworks_mock.call_args.kwargs
            self.assertIs(kwargs["overlay_parent"], host._window)
            self.assertEqual(kwargs["duration_ms"], 3200)
            self.assertIs(fireworks_mock.call_args.args[0], host.license_section_card)

        @patch(
            f"{_licensing_mod.__name__}.license_summary",
            return_value={"active": False, "cached": False},
        )
        def test_remove_license_without_cache_syncs_pro_features_only(
            self, _summary_mock
        ) -> None:
            host = _LicenseHost()
            with patch.object(host, "_refresh_license_status_ui") as refresh_mock:
                host._on_remove_license_clicked()
                refresh_mock.assert_called_once()
                self.assertEqual(host.library_pro_sync_calls, 1)


if __name__ == "__main__":
    unittest.main()
