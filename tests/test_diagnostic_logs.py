"""Tests for diagnostic log catalog and tail readers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.diagnostic_logs import (
    describe_log_file,
    get_diagnostic_log,
    iter_diagnostic_logs,
    read_log_tail,
)


class DiagnosticLogsTests(unittest.TestCase):
    def test_catalog_contains_expected_logs(self) -> None:
        ids = {spec.id for spec in iter_diagnostic_logs()}
        self.assertEqual(ids, {"llm_debug", "routing_debug"})

    def test_get_diagnostic_log(self) -> None:
        spec = get_diagnostic_log("llm_debug")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.title, "LLM debug log")

    def test_read_log_tail_returns_placeholder_when_missing(self) -> None:
        missing = Path("/tmp/qube-nonexistent-log-for-test.log")
        text = read_log_tail(missing)
        self.assertIn("(no file yet:", text)

    def test_read_log_tail_limits_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.log"
            path.write_text("\n".join(f"line {i}" for i in range(10)), encoding="utf-8")
            text = read_log_tail(path, max_lines=3)
            self.assertIn("earlier line(s) omitted", text)
            self.assertIn("line 9", text)
            self.assertNotIn("line 0", text)

    def test_describe_log_file_reports_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.log"
            path.write_text("hello", encoding="utf-8")
            desc = describe_log_file(path)
            self.assertIn("B", desc)
            self.assertIn("updated", desc)

    def test_describe_log_file_missing(self) -> None:
        self.assertEqual(
            describe_log_file(Path("/tmp/qube-missing-log-for-test.log")),
            "Not created yet",
        )

    @patch("core.routing_debug_sink.logs_dir")
    @patch("core.llm_debug_sink.logs_dir")
    def test_default_paths_live_under_logs_dir(
        self, llm_logs_dir_mock, routing_logs_dir_mock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_logs_dir_mock.return_value = root
            routing_logs_dir_mock.return_value = root
            llm = get_diagnostic_log("llm_debug")
            routing = get_diagnostic_log("routing_debug")
            assert llm is not None and routing is not None
            self.assertEqual(llm.path_fn(), root / "llm_debug.log")
            self.assertEqual(routing.path_fn(), root / "routing_debug.log")


if __name__ == "__main__":
    unittest.main()
