"""Tests for retrieval observability (schema v2)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.bundle_builder import build_general_web_bundle  # noqa: E402
from core.knowledge.observability import (  # noqa: E402
    RETRIEVAL_TRACE_EVENT,
    build_retrieval_trace,
    serialize_retrieval_trace,
)


class TestRetrievalObservability(unittest.TestCase):
    def test_serialize_retrieval_trace(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="birds dust bath",
            query_resolved="birds dust bath",
            kept_rows=[{"title": "Birds", "snippet": "Dust baths."}],
            rejected_count=0,
            latency_ms=42.0,
        )
        trace = build_retrieval_trace(
            bundle,
            relevance_diag={"web_results_raw_count": 1, "web_results_kept_count": 1},
            session_id="sess-1",
            turn_id=7,
        )
        payload = serialize_retrieval_trace(trace, sources=bundle.sources)
        self.assertEqual(payload["event"], RETRIEVAL_TRACE_EVENT)
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["knowledge_service"], "general_web")
        self.assertEqual(payload["bundle_id"], bundle.bundle_id)
        self.assertEqual(len(payload["sources"]), 1)


if __name__ == "__main__":
    unittest.main()
