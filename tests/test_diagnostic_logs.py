"""Tests for diagnostic log catalog and tail readers."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.diagnostic_logs import (
    DiagnosticLogSpec,
    clear_diagnostic_log,
    describe_log_file,
    get_diagnostic_log,
    iter_diagnostic_logs,
    read_log_tail,
)


class DiagnosticLogsTests(unittest.TestCase):
    def test_catalog_contains_expected_logs(self) -> None:
        ids = {spec.id for spec in iter_diagnostic_logs()}
        self.assertEqual(
            ids,
            {"app_log", "llm_debug", "routing_debug", "web_search_audit", "skills_debug"},
        )

    def test_routing_log_supports_recording_toggle(self) -> None:
        spec = get_diagnostic_log("routing_debug")
        assert spec is not None
        self.assertTrue(spec.supports_recording_toggle)
        self.assertNotIn("QUBE_", spec.description)

    def test_get_diagnostic_log(self) -> None:
        spec = get_diagnostic_log("app_log")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.title, "Application log")
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

    def test_skills_debug_log_spec(self) -> None:
        spec = get_diagnostic_log("skills_debug")
        assert spec is not None
        self.assertTrue(spec.supports_recording_toggle)
        self.assertIn("skill", spec.description.lower())
        self.assertIn("AI & Models", spec.note)

    def test_llm_debug_log_spec_notes_file_recording_only(self) -> None:
        spec = get_diagnostic_log("llm_debug")
        assert spec is not None
        self.assertIn("file recording", spec.note.lower())

    def test_web_search_audit_log_spec(self) -> None:
        spec = get_diagnostic_log("web_search_audit")
        assert spec is not None
        self.assertTrue(spec.supports_recording_toggle)
        self.assertIn("web search", spec.description.lower())
        self.assertIn("web_search.log", str(spec.path_fn()))

    @patch("core.app_log_sink.logs_dir")
    @patch("core.routing_debug_sink.logs_dir")
    @patch("core.llm_debug_sink.logs_dir")
    def test_default_paths_live_under_logs_dir(
        self, llm_logs_dir_mock, routing_logs_dir_mock, app_logs_dir_mock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_logs_dir_mock.return_value = root
            routing_logs_dir_mock.return_value = root
            app_logs_dir_mock.return_value = root
            app = get_diagnostic_log("app_log")
            llm = get_diagnostic_log("llm_debug")
            routing = get_diagnostic_log("routing_debug")
            assert app is not None and llm is not None and routing is not None
            self.assertEqual(app.path_fn(), root / "qube.log")
            self.assertEqual(llm.path_fn(), root / "llm_debug.log")
            self.assertEqual(routing.path_fn(), root / "routing_debug.log")

    def test_clear_diagnostic_log_truncates_active_handler(self) -> None:
        from logging.handlers import RotatingFileHandler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.log"
            handler = RotatingFileHandler(
                path,
                maxBytes=1024,
                backupCount=2,
                encoding="utf-8",
            )
            root = logging.getLogger()
            root.addHandler(handler)
            try:
                logging.info("line one")
                handler.flush()
                backup = Path(f"{path}.1")
                backup.write_text("old backup", encoding="utf-8")

                spec = DiagnosticLogSpec(
                    id="test_clear",
                    title="Test log",
                    description="",
                    path_fn=lambda: path,
                )
                result = clear_diagnostic_log(spec)

                self.assertTrue(result.success)
                self.assertEqual(path.read_text(encoding="utf-8"), "")
                self.assertFalse(backup.exists())
            finally:
                root.removeHandler(handler)
                handler.close()

    def test_clear_diagnostic_log_without_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orphan.log"
            path.write_text("stale content", encoding="utf-8")

            spec = DiagnosticLogSpec(
                id="test_orphan",
                title="Orphan log",
                description="",
                path_fn=lambda: path,
            )
            result = clear_diagnostic_log(spec)

            self.assertTrue(result.success)
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_iter_diagnostic_logs_by_category_audit(self) -> None:
        from core.diagnostic_logs import iter_diagnostic_logs_by_category

        audit_ids = {spec.id for spec in iter_diagnostic_logs_by_category("audit")}
        self.assertEqual(audit_ids, {"llm_debug", "routing_debug", "web_search_audit"})

    def test_iter_diagnostic_logs_by_category_technical(self) -> None:
        from core.diagnostic_logs import iter_diagnostic_logs_by_category

        technical_ids = {
            spec.id for spec in iter_diagnostic_logs_by_category("technical")
        }
        self.assertEqual(technical_ids, {"app_log", "skills_debug"})


if __name__ == "__main__":
    unittest.main()
