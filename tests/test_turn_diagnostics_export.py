"""Tests for diagnostics export and settings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import core.settings_store as settings_store_module
from core.app_settings import (
    get_diagnostics_technical_detail_enabled,
    set_diagnostics_technical_detail_enabled,
)
from core.settings_store import SettingsStore, reset_settings_store_for_tests
from core.turn_diagnostics_export import build_issue_report_snippet


class TestTurnDiagnosticsExport(unittest.TestCase):
    def test_issue_report_snippet_contains_ids(self) -> None:
        bundle = {
            "app_version": "1.0.0",
            "session_id_short": "abcd1234…",
            "target_assistant_message_id": "msg-1",
            "target_turn": {
                "user_excerpt": "Why?",
                "assistant_excerpt": "Because.",
            },
            "orchestration": {
                "routing": {
                    "status": "recorded",
                    "record": {"route": "none", "strategy": "chat", "summary": "ok"},
                }
            },
            "inference_snapshot": {"engine_mode": "internal"},
        }
        text = build_issue_report_snippet(bundle)
        self.assertIn("msg-1", text)
        self.assertIn("none", text)

    def test_technical_detail_setting_roundtrip(self) -> None:
        reset_settings_store_for_tests()
        with tempfile.TemporaryDirectory() as tmp:
            settings_store_module._store = SettingsStore(
                user_path=Path(tmp) / "settings.json"
            )
            set_diagnostics_technical_detail_enabled(True)
            self.assertTrue(get_diagnostics_technical_detail_enabled())
            set_diagnostics_technical_detail_enabled(False)
            self.assertFalse(get_diagnostics_technical_detail_enabled())
        reset_settings_store_for_tests()


if __name__ == "__main__":
    unittest.main()