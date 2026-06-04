"""Tests for opt-in raw vs presented completion logging."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from core.completion_output_trace import (
    CompletionOutputSnapshot,
    build_completion_output_trace_payload,
    completion_output_trace_enabled,
    log_completion_output_trace,
)


class CompletionOutputTraceTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(completion_output_trace_enabled())

    def test_enabled_with_env(self) -> None:
        with patch.dict(os.environ, {"QUBE_LOG_RAW_COMPLETION": "1"}, clear=False):
            self.assertTrue(completion_output_trace_enabled())

    def test_payload_includes_all_stages(self) -> None:
        snap = CompletionOutputSnapshot(
            engine_mode="internal",
            raw_text="<thinking>plan</thinking>We need to explain. Hello.",
            after_harmony_parser="Hello.",
            after_worker_filters="Hello.",
            streamed_incremental="Hello",
            worker_return_text="Hello.",
            engine_end_text="<thinking>plan</thinking>We need to explain. Hello.",
            retry_replaced=False,
        )
        payload = build_completion_output_trace_payload(
            session_id="sess-1",
            snapshot=snap,
            presented_text="Hello.",
        )
        self.assertEqual(payload["event"], "llm_completion_output_trace")
        self.assertEqual(payload["session_id"], "sess-1")
        self.assertEqual(payload["engine_mode"], "internal")
        self.assertIn("raw_text->after_harmony_parser", payload["stages_changed"])
        self.assertFalse(payload["raw_equals_presented"])
        self.assertGreater(payload["removed_char_count"], 0)
        self.assertTrue(payload["stream_incremental_diverged_from_filters"])

    def test_truncation_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {"QUBE_LOG_RAW_COMPLETION_MAX_CHARS": "5"},
            clear=False,
        ):
            snap = CompletionOutputSnapshot(
                engine_mode="external",
                raw_text="0123456789",
                worker_return_text="01234",
            )
            payload = build_completion_output_trace_payload(
                session_id="",
                snapshot=snap,
                presented_text="01234",
            )
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["raw_text"], "01234")
        self.assertTrue(payload["raw_text_truncated"])
        self.assertEqual(payload["raw_text_full_len"], 10)

    def test_log_emits_json_and_summary(self) -> None:
        snap = CompletionOutputSnapshot(
            engine_mode="internal",
            raw_text="raw",
            worker_return_text="final",
        )
        with patch.dict(os.environ, {"QUBE_LOG_RAW_COMPLETION": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_completion_output_trace(
                    session_id="abc",
                    snapshot=snap,
                    presented_text="final",
                )
        messages = [r.getMessage() for r in captured.records]
        self.assertTrue(any("[CompletionOutputTrace]" in m for m in messages))
        json_line = next(m for m in messages if m.startswith("{"))
        parsed = json.loads(json_line)
        self.assertEqual(parsed["event"], "llm_completion_output_trace")
        self.assertEqual(parsed["raw_text"], "raw")
        self.assertEqual(parsed["presented_text"], "final")

    def test_log_noop_when_disabled(self) -> None:
        snap = CompletionOutputSnapshot(engine_mode="internal", raw_text="x")
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.completion_output_trace.logger") as mock_logger:
                log_completion_output_trace(
                    session_id="abc",
                    snapshot=snap,
                    presented_text="x",
                )
        mock_logger.info.assert_not_called()

    def test_log_noop_without_snapshot(self) -> None:
        with patch.dict(os.environ, {"QUBE_LOG_RAW_COMPLETION": "1"}, clear=False):
            with patch("core.completion_output_trace.logger") as mock_logger:
                log_completion_output_trace(
                    session_id="abc",
                    snapshot=None,
                    presented_text="x",
                )
        mock_logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
