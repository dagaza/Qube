"""Tests for retrieval record persistence."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.bundle_builder import build_general_web_bundle  # noqa: E402
from core.knowledge.retrieval_records import (  # noqa: E402
    RetrievalContextFingerprint,
    save_retrieval_record,
)


class TestRetrievalRecords(unittest.TestCase):
    def test_save_retrieval_record_delegates_to_database_manager(self) -> None:
        db = MagicMock()
        bundle = build_general_web_bundle(
            query_raw="birds",
            query_resolved="birds",
            kept_rows=[
                {
                    "title": "Birds",
                    "snippet": "Dust bathing behavior",
                    "url": "https://example.com/birds",
                }
            ],
            rejected_count=0,
            latency_ms=42.0,
        )
        fingerprint = RetrievalContextFingerprint(
            query_raw="birds",
            query_resolved="birds",
            knowledge_service="general_web",
            preset_id=None,
            adapter_filter=(),
            retrieval_profile="balanced",
            connector_config_hashes=(),
        )

        save_retrieval_record(
            db,
            request_id="req-123",
            bundle=bundle,
            context_fingerprint=fingerprint,
            session_id="sess-1",
            turn_id=7,
        )

        db.save_retrieval_record.assert_called_once()
        kwargs = db.save_retrieval_record.call_args.kwargs
        self.assertEqual(kwargs["request_id"], "req-123")
        self.assertEqual(kwargs["bundle_id"], bundle.bundle_id)
        self.assertEqual(kwargs["session_id"], "sess-1")
        self.assertEqual(kwargs["turn_id"], 7)
        self.assertEqual(kwargs["evidence_count"], 1)

    def test_save_retrieval_record_noop_when_db_missing(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="birds",
            query_resolved="birds",
            kept_rows=[],
            rejected_count=0,
            latency_ms=0.0,
        )
        fingerprint = RetrievalContextFingerprint(
            query_raw="birds",
            query_resolved="birds",
            knowledge_service="general_web",
            preset_id=None,
            adapter_filter=(),
            retrieval_profile="balanced",
            connector_config_hashes=(),
        )
        save_retrieval_record(
            None,
            request_id="req-123",
            bundle=bundle,
            context_fingerprint=fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
