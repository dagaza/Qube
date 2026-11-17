"""Knowledge graph copy when branching conversations."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.conversation_branch import branch_session
from core.database import DatabaseManager
from core.knowledge.bundle_builder import build_general_web_bundle
from core.knowledge.graph.service import export_session_graph, record_bundle_in_session_graph


def _web_bundle(query: str) -> object:
    return build_general_web_bundle(
        query_raw=query,
        query_resolved=query,
        kept_rows=[
            {
                "title": f"Source for {query}",
                "snippet": "Evidence snippet.",
                "url": "https://example.org/doc",
                "_web_token_overlap": 0.5,
            }
        ],
        rejected_count=0,
        latency_ms=12.0,
    )


class ConversationBranchGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(str(Path(self._tmpdir.name) / "test.db"))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_branch_copies_graph_for_prefix_bundles_only(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        source_id = self.db.create_session("Graph parent", folder_id=folder_id)

        b1 = _web_bundle("first query")
        b2 = _web_bundle("second query")
        self.db.add_message(source_id, "user", "Q1")
        a1 = self.db.add_message(source_id, "assistant", "A1", evidence_bundle_id=b1.bundle_id)
        record_bundle_in_session_graph(
            self.db, session_id=source_id, bundle=b1, message_id=a1
        )
        self.db.add_message(source_id, "user", "Q2")
        a2 = self.db.add_message(source_id, "assistant", "A2", evidence_bundle_id=b2.bundle_id)
        record_bundle_in_session_graph(
            self.db, session_id=source_id, bundle=b2, message_id=a2
        )

        parent_graph = export_session_graph(self.db, source_id)
        self.assertEqual(
            len([n for n in parent_graph["nodes"] if n.get("kind") == "query"]),
            2,
        )

        result = branch_session(
            self.db,
            source_session_id=source_id,
            through_message_id=a1,
        )
        child_graph = export_session_graph(self.db, result.new_session_id)
        query_nodes = [n for n in child_graph["nodes"] if n.get("kind") == "query"]
        self.assertEqual(len(query_nodes), 1)
        self.assertEqual(str(query_nodes[0].get("bundle_id")), b1.bundle_id)

        child_messages = self.db.get_session_history(result.new_session_id)
        new_a1 = next(m for m in child_messages if m["role"] == "assistant")
        self.assertEqual(str(query_nodes[0].get("message_id")), new_a1["id"])

    def test_branch_full_thread_copies_all_bundles(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        source_id = self.db.create_session("Full graph", folder_id=folder_id)
        b1 = _web_bundle("alpha")
        b2 = _web_bundle("beta")
        self.db.add_message(source_id, "user", "U")
        a1 = self.db.add_message(source_id, "assistant", "A1", evidence_bundle_id=b1.bundle_id)
        record_bundle_in_session_graph(
            self.db, session_id=source_id, bundle=b1, message_id=a1
        )
        self.db.add_message(source_id, "user", "U2")
        a2 = self.db.add_message(source_id, "assistant", "A2", evidence_bundle_id=b2.bundle_id)
        record_bundle_in_session_graph(
            self.db, session_id=source_id, bundle=b2, message_id=a2
        )

        result = branch_session(
            self.db,
            source_session_id=source_id,
            through_message_id=a2,
        )
        child_graph = export_session_graph(self.db, result.new_session_id)
        query_nodes = [n for n in child_graph["nodes"] if n.get("kind") == "query"]
        self.assertEqual(len(query_nodes), 2)


if __name__ == "__main__":
    unittest.main()
