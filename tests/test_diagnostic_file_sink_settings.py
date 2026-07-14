"""Tests for diagnostic log file sink settings sync."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.app_log_sink import (
    attach_app_log_file_sink,
    detach_app_log_file_sink,
    is_app_log_file_sink_attached,
)
from core.llm_debug_sink import (
    attach_llm_debug_file_sink,
    detach_llm_debug_file_sink,
    is_llm_debug_file_sink_attached,
)
from core.logging_bootstrap import (
    effective_llm_debug_file_enabled,
    set_app_log_file_recording_enabled,
    set_llm_debug_log_file_recording_enabled,
    sync_diagnostic_file_sinks_from_settings,
)


class DiagnosticFileSinkSettingsTests(unittest.TestCase):
    def tearDown(self) -> None:
        detach_app_log_file_sink()
        detach_llm_debug_file_sink()

    @patch("core.logging_bootstrap.get_llm_debug_log_file_enabled", return_value=False)
    @patch("core.logging_bootstrap.get_app_log_file_enabled", return_value=False)
    @patch("core.app_log_sink.app_log_env_override", return_value=None)
    def test_sync_detaches_when_settings_disabled(
        self,
        _env_mock,
        _app_mock,
        _llm_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_path = root / "qube.log"
            llm_path = root / "llm_debug.log"
            with patch("core.app_log_sink.default_app_log_path", return_value=app_path):
                with patch(
                    "core.llm_debug_sink.default_llm_debug_log_path",
                    return_value=llm_path,
                ):
                    attach_app_log_file_sink()
                    attach_llm_debug_file_sink()
                    sync_diagnostic_file_sinks_from_settings()
                    self.assertFalse(is_app_log_file_sink_attached())
                    self.assertFalse(is_llm_debug_file_sink_attached())

    @patch("core.logging_bootstrap.set_app_log_file_enabled")
    @patch("core.logging_bootstrap.get_app_log_file_enabled", return_value=False)
    @patch("core.app_log_sink.app_log_env_override", return_value=None)
    def test_set_app_log_recording_enabled_attaches_handler(
        self,
        _env_mock,
        get_mock,
        set_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qube.log"
            with patch("core.app_log_sink.default_app_log_path", return_value=path):
                detach_app_log_file_sink()
                set_app_log_file_recording_enabled(True)
                self.assertTrue(is_app_log_file_sink_attached())
                detach_app_log_file_sink()
                set_mock.assert_called_once_with(True)
                get_mock.assert_called()

    @patch("core.logging_bootstrap.set_llm_debug_log_file_enabled")
    @patch("core.logging_bootstrap.get_llm_debug_log_file_enabled", return_value=True)
    def test_set_llm_debug_recording_disabled_detaches_handler(
        self,
        get_mock,
        set_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_debug.log"
            with patch("core.llm_debug_sink.default_llm_debug_log_path", return_value=path):
                attach_llm_debug_file_sink()
                set_llm_debug_log_file_recording_enabled(False)
                self.assertFalse(is_llm_debug_file_sink_attached())
                set_mock.assert_called_once_with(False)
                get_mock.assert_called()

    def test_app_log_env_override_skips_settings_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qube.log"
            with patch.dict("os.environ", {"QUBE_APP_LOG": "0"}, clear=False):
                with patch(
                    "core.app_log_sink.default_app_log_path",
                    return_value=path,
                ):
                    with patch(
                        "core.app_settings.set_app_log_file_enabled"
                    ) as set_mock:
                        detach_app_log_file_sink()
                        set_app_log_file_recording_enabled(True)
                        set_mock.assert_not_called()
                        self.assertFalse(is_app_log_file_sink_attached())

    def test_llm_debug_log_env_override_skips_settings_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_debug.log"
            with patch.dict("os.environ", {"QUBE_LLM_DEBUG_LOG": "0"}, clear=False):
                with patch(
                    "core.llm_debug_sink.default_llm_debug_log_path",
                    return_value=path,
                ):
                    with patch(
                        "core.app_settings.set_llm_debug_log_file_enabled"
                    ) as set_mock:
                        detach_llm_debug_file_sink()
                        set_llm_debug_log_file_recording_enabled(True)
                        set_mock.assert_not_called()
                        self.assertFalse(is_llm_debug_file_sink_attached())

    def test_effective_llm_debug_respects_env_override(self) -> None:
        with patch.dict("os.environ", {"QUBE_LLM_DEBUG_LOG": "0"}, clear=False):
            with patch(
                "core.logging_bootstrap.get_llm_debug_log_file_enabled",
                return_value=True,
            ):
                self.assertFalse(effective_llm_debug_file_enabled())


if __name__ == "__main__":
    unittest.main()
