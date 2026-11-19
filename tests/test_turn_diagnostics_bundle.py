"""Tests for turn diagnostics bundle builder and forensics payload."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.database import DatabaseManager
from core.knowledge.ui_sources_payload import (
    decode_forensics_payload,
    encode_sources_payload,
)
from core.turn_diagnostics_bundle import build_turn_diagnostics_bundle
from core.turn_forensics import (
    build_turn_forensics_snapshot,
    parse_diagnostics_tool_id,
)


class TestTurnDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(str(Path(self._tmpdir.name) / "test.db"))
        self.sid = self.db.create_session("Diagnostics test")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_parse_diagnostics_tool_id(self) -> None:
        self.assertEqual(parse_diagnostics_tool_id("diagnostics"), (None, 0))
        self.assertEqual(
            parse_diagnostics_tool_id("diagnostics::abc-123"),
            ("abc-123", 0),
        )
        self.assertEqual(
            parse_diagnostics_tool_id("diagnostics::abc-123::2"),
            ("abc-123", 2),
        )

    def test_forensics_roundtrip_in_sources_payload(self) -> None:
        forensics = build_turn_forensics_snapshot(
            routing_turn_id=7,
            effective_route="none",
            strategy="chat",
            composer_attachments=[],
        )
        raw = encode_sources_payload([], forensics=forensics)
        assert raw is not None
        decoded = decode_forensics_payload(raw)
        assert decoded is not None
        self.assertEqual(decoded.get("routing_turn_id"), 7)

    def test_build_bundle_for_session_with_assistant(self) -> None:
        user_id = self.db.add_message(self.sid, "user", "What is Kathmandu?")
        self.assertTrue(user_id)
        forensics = build_turn_forensics_snapshot(
            routing_turn_id=1,
            effective_route="none",
            strategy="attachment_tool_internet",
        )
        payload = encode_sources_payload(
            [{"id": 1, "type": "web", "filename": "example.com"}],
            forensics=forensics,
        )
        aid = self.db.add_message(
            self.sid,
            "assistant",
            "Kathmandu is the capital of Nepal. [1]",
            sources_json=payload,
        )
        bundle = build_turn_diagnostics_bundle(
            self.db,
            session_id=self.sid,
            target_assistant_message_id=aid,
            routing_debug_buffer=None,
        )
        self.assertEqual(bundle.get("target_assistant_message_id"), aid)
        self.assertIn("target_turn", bundle)
        self.assertEqual(
            bundle.get("orchestration", {}).get("forensics", {}).get("routing_turn_id"),
            1,
        )
        orch = bundle.get("orchestration") or {}
        self.assertEqual(orch.get("routing", {}).get("status"), "recorded")
        src_summary = (bundle.get("retrieval") or {}).get("sources_summary") or {}
        self.assertEqual(src_summary.get("count"), 1)
        cites = bundle.get("citations") or {}
        self.assertEqual(cites.get("verdict"), "bracket_tokens_match_sources")

    def test_citation_verdict_when_no_brackets_in_text(self) -> None:
        sid = self.db.create_session("Cit test")
        payload = encode_sources_payload(
            [{"id": 1, "type": "web", "url": "https://example.com", "title": "Ex"}]
        )
        aid = self.db.add_message(
            sid,
            "assistant",
            "Answer with no numeric cite tokens.",
            sources_json=payload,
        )
        bundle = build_turn_diagnostics_bundle(
            self.db, session_id=sid, target_assistant_message_id=aid
        )
        self.assertEqual(
            (bundle.get("citations") or {}).get("verdict"),
            "sources_attached_no_bracket_tokens_in_stored_text",
        )
        self.assertTrue((bundle.get("logging_state") or {}).get("routing_debug_log_enabled") is not None)


class TestTurnDiagnosticsPromptFormat(unittest.TestCase):
    def test_format_context_wraps_json(self) -> None:
        from core.turn_diagnostics_prompts import format_turn_diagnostics_context

        text = format_turn_diagnostics_context({"schema_version": 1, "x": 1})
        self.assertIn("TURN DIAGNOSTICS BUNDLE", text)
        self.assertIn('"schema_version"', text)