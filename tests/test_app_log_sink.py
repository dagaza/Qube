"""Tests for general application log file sink."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.app_log_sink import (
    QubeAppLogFilter,
    app_log_enabled,
    app_log_level,
    attach_app_log_file_sink,
    detach_app_log_file_sink_for_tests,
)


class AppLogSinkTests(unittest.TestCase):
    def tearDown(self) -> None:
        detach_app_log_file_sink_for_tests()

    def test_app_log_enabled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(app_log_enabled())
        with patch.dict("os.environ", {"QUBE_APP_LOG": "0"}, clear=True):
            self.assertFalse(app_log_enabled())

    def test_app_log_level_defaults_to_info(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(app_log_level(), logging.INFO)
        with patch.dict("os.environ", {"QUBE_APP_LOG_LEVEL": "DEBUG"}, clear=True):
            self.assertEqual(app_log_level(), logging.DEBUG)

    def test_filter_accepts_qube_modules(self) -> None:
        filt = QubeAppLogFilter()
        record = logging.LogRecord(
            name="Qube.Main",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        self.assertTrue(filt.filter(record))

    def test_filter_rejects_non_qube(self) -> None:
        filt = QubeAppLogFilter()
        record = logging.LogRecord(
            name="urllib3.connectionpool",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="noise",
            args=(),
            exc_info=None,
        )
        self.assertFalse(filt.filter(record))

    def test_filter_excludes_dedicated_debug_loggers(self) -> None:
        filt = QubeAppLogFilter()
        for name in ("Qube.NativeLLM.Debug", "Qube.RoutingDebug", "Qube.SkillsDebug"):
            record = logging.LogRecord(
                name=name,
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="debug-only",
                args=(),
                exc_info=None,
            )
            self.assertFalse(filt.filter(record))

    def test_attach_writes_qube_records_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qube.log"
            with patch.dict(
                "os.environ",
                {"QUBE_APP_LOG": "1", "QUBE_APP_LOG_LEVEL": "INFO"},
                clear=False,
            ):
                handler = attach_app_log_file_sink(log_path=path, level=logging.INFO)
                assert handler is not None
                logging.getLogger().setLevel(logging.DEBUG)
                logging.getLogger("Qube.Test").info("voice capture started")
                logging.getLogger("Qube.NativeLLM.Debug").info("should not appear")
                for h in logging.getLogger().handlers:
                    h.flush()
                detach_app_log_file_sink_for_tests()
            text = path.read_text(encoding="utf-8")
            self.assertIn("voice capture started", text)
            self.assertNotIn("should not appear", text)

    def test_disabled_via_env_returns_none(self) -> None:
        with patch.dict("os.environ", {"QUBE_APP_LOG": "0"}, clear=True):
            handler = attach_app_log_file_sink()
            self.assertIsNone(handler)


if __name__ == "__main__":
    unittest.main()
