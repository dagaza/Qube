"""Unit tests for branch graph helpers."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.graph.branch_copy import build_branch_session_graph
from core.knowledge.graph.build import build_graph_from_bundle, merge_graphs
from core.knowledge.bundle_builder import build_general_web_bundle


class TestKnowledgeGraphBranchCopy(unittest.TestCase):
    def test_build_branch_session_graph_filters_bundles(self) -> None:
        b1 = build_general_web_bundle(
            query_raw="one",
            query_resolved="one",
            kept_rows=[{"title": "S1", "snippet": "x", "url": "https://a.test"}],
            rejected_count=0,
            latency_ms=1.0,
        )
        b2 = build_general_web_bundle(
            query_raw="two",
            query_resolved="two",
            kept_rows=[{"title": "S2", "snippet": "y", "url": "https://b.test"}],
            rejected_count=0,
            latency_ms=1.0,
        )
        parent = merge_graphs(
            build_graph_from_bundle(b1, message_id="msg-a"),
            build_graph_from_bundle(b2, message_id="msg-b"),
        )

        class _Db:
            @staticmethod
            def get_session_knowledge_graph_json(_sid: str) -> str | None:
                from core.knowledge.graph.build import graph_to_json

                return graph_to_json(parent)

        prefix = [
            {"id": "msg-a", "role": "assistant", "evidence_bundle_id": b1.bundle_id},
        ]
        graph = build_branch_session_graph(
            _Db(),
            source_session_id="sess",
            prefix_messages=prefix,
            message_id_map={"msg-a": "msg-a-new"},
        )
        assert graph is not None
        queries = [n for n in graph["nodes"] if n.get("kind") == "query"]
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["bundle_id"], b1.bundle_id)
        self.assertEqual(queries[0]["message_id"], "msg-a-new")


if __name__ == "__main__":
    unittest.main()
