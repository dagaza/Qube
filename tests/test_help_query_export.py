"""Tests for @help query log export (§13.4)."""

from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_query_export import (  # noqa: E402
    aggregate_help_queries,
    export_help_query_report,
    parse_help_log_line,
)


class HelpQueryExportTests(unittest.TestCase):
    def test_parse_help_log_line(self) -> None:
        payload = {
            "event": "help_query",
            "query": "Where are GPU layers?",
            "retrieved_doc_ids": ["features.settings.ai_models"],
            "canonical_id": "features.settings.ai-models.gpu-layers",
        }
        line = f"INFO Qube.Help [Help] {json.dumps(payload)}"
        parsed = parse_help_log_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["query"], "Where are GPU layers?")

    def test_aggregate_and_backlog(self) -> None:
        events = [
            {
                "event": "help_query",
                "query": "Where are GPU layers?",
                "retrieved_doc_ids": ["features.settings.ai_models"],
                "canonical_id": "x",
            },
            {
                "event": "help_query",
                "query": "where are gpu layers",
                "retrieved_doc_ids": [],
            },
            {
                "event": "help_query",
                "query": "Where are GPU layers?",
                "retrieved_doc_ids": ["features.settings.ai_models"],
            },
        ]
        report = export_help_query_report(events)
        self.assertEqual(report["total_events"], 3)
        self.assertEqual(report["unique_queries"], 1)
        self.assertGreaterEqual(len(report["doc_backlog"]), 1)
        aggregates = aggregate_help_queries(events)
        self.assertEqual(aggregates[0].count, 3)
        self.assertEqual(aggregates[0].empty_retrieval_count, 1)


if __name__ == "__main__":
    unittest.main()
