"""Tests for typed SearchOutcome helpers."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.search_outcome import (  # noqa: E402
    SearchOutcome,
    SearchOutcomeKind,
    attach_search_outcome,
    build_search_outcome_from_ddg,
    format_search_outcome_explain_text,
    format_search_outcome_summary_line,
    search_outcome_from_relevance_diag,
)


class TestSearchOutcome(unittest.TestCase):
    def test_build_bot_challenge_from_inspection(self) -> None:
        rows = [
            {
                "title": "Internet search blocked: DuckDuckGo bot challenge (try again later).",
                "snippet": "",
            }
        ]
        inspection = {
            "response_kind": "bot_challenge",
            "http_status": 202,
            "parsed_rows": 0,
            "bot_challenge_signals": ("http_202", "anomaly_modal"),
        }
        outcome = build_search_outcome_from_ddg(rows, inspection, candidate_count=0)
        self.assertEqual(outcome.kind, SearchOutcomeKind.BOT_CHALLENGE)
        self.assertEqual(outcome.http_status, 202)
        self.assertIn("http_202", outcome.bot_challenge_signals)

    def test_build_serp_success_with_candidates(self) -> None:
        rows = [
            {
                "title": "Birds",
                "snippet": "Dust bathing",
                "url": "https://example.org/birds",
            }
        ]
        outcome = build_search_outcome_from_ddg(rows, {"response_kind": "serp"}, candidate_count=1)
        self.assertEqual(outcome.kind, SearchOutcomeKind.SERP_SUCCESS)
        self.assertEqual(outcome.candidate_count, 1)

    def test_round_trip_via_relevance_diag(self) -> None:
        outcome = SearchOutcome(
            kind=SearchOutcomeKind.EMPTY_PARSE,
            http_status=200,
            parsed_rows=0,
            failure_sentinel_reason="ddg_empty_parse",
        )
        diag = attach_search_outcome({}, outcome)
        restored = search_outcome_from_relevance_diag(diag)
        assert restored is not None
        self.assertEqual(restored.kind, SearchOutcomeKind.EMPTY_PARSE)
        self.assertEqual(restored.failure_sentinel_reason, "ddg_empty_parse")

    def test_format_summary_line(self) -> None:
        outcome = SearchOutcome(
            kind=SearchOutcomeKind.BOT_CHALLENGE,
            http_status=202,
            bot_challenge_signals=("http_202", "anomaly_modal"),
        )
        line = format_search_outcome_summary_line(outcome)
        assert line is not None
        self.assertIn("Search:", line)
        self.assertIn("bot challenge", line.lower())
        self.assertIn("http=202", line)

    def test_format_explain_text_includes_recovery(self) -> None:
        outcome = SearchOutcome(
            kind=SearchOutcomeKind.NETWORK_ERROR,
            recovery_hint="Check network connectivity and retry.",
        )
        text = format_search_outcome_explain_text(outcome)
        self.assertIn("network_error", text)
        self.assertIn("Check network connectivity", text)


if __name__ == "__main__":
    unittest.main()
