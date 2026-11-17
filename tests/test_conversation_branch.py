"""Tests for conversation branching (session prefix copy)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.conversation_branch import BranchError, branch_session
from core.database import DatabaseManager
from core.knowledge.ui_sources_payload import encode_sources_payload


class ConversationBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(str(Path(self._tmpdir.name) / "test.db"))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_branch_copies_inclusive_prefix(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        source_id = self.db.create_session("Parent chat", folder_id=folder_id)
        self.db.add_message(source_id, "user", "U1")
        self.db.add_message(source_id, "assistant", "A1")
        self.db.add_message(source_id, "user", "U2")
        a2_id = self.db.add_message(source_id, "assistant", "A2")
        self.db.add_message(source_id, "user", "U3")

        result = branch_session(
            self.db,
            source_session_id=source_id,
            through_message_id=a2_id,
        )

        self.assertEqual(result.copied_message_count, 4)
        parent = self.db.get_session_history(source_id)
        child = self.db.get_session_history(result.new_session_id)
        self.assertEqual(len(parent), 5)
        self.assertEqual(len(child), 4)
        self.assertEqual([m["content"] for m in child], ["U1", "A1", "U2", "A2"])

        meta = self.db.get_session(result.new_session_id)
        assert meta is not None
        self.assertEqual(meta.get("branched_from_session_id"), source_id)
        self.assertEqual(meta.get("branched_from_message_id"), a2_id)
        self.assertTrue(str(meta.get("title") or "").startswith("Branch:"))

    def test_branch_preserves_sources_json_bytes(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        source_id = self.db.create_session("Citations", folder_id=folder_id)
        self.db.add_message(source_id, "user", "Q")
        payload = encode_sources_payload(
            [{"id": "1", "title": "Doc", "snippet": "x"}],
            transparency={"summary": "test"},
        )
        assistant_id = self.db.add_message(
            source_id,
            "assistant",
            "Answer [1]",
            sources_json=payload,
        )

        result = branch_session(
            self.db,
            source_session_id=source_id,
            through_message_id=assistant_id,
        )
        raw_child = self.db.list_session_messages_raw(result.new_session_id)
        assistant_rows = [r for r in raw_child if r["role"] == "assistant"]
        self.assertEqual(len(assistant_rows), 1)
        self.assertEqual(assistant_rows[0]["sources_json"], payload)

        decoded = self.db.get_session_history(result.new_session_id)
        self.assertEqual(len(decoded[1].get("sources") or []), 1)

    def test_branch_unknown_message_raises(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        source_id = self.db.create_session("Empty branch", folder_id=folder_id)
        self.db.add_message(source_id, "user", "Hi")
        with self.assertRaises(BranchError):
            branch_session(
                self.db,
                source_session_id=source_id,
                through_message_id="not-a-real-id",
            )

    def test_parent_unchanged_after_child_send(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        source_id = self.db.create_session("Stable parent", folder_id=folder_id)
        u1 = self.db.add_message(source_id, "user", "One")
        a1 = self.db.add_message(source_id, "assistant", "Two")

        result = branch_session(
            self.db,
            source_session_id=source_id,
            through_message_id=a1,
        )
        self.db.add_message(result.new_session_id, "user", "New branch turn")

        parent = self.db.get_session_history(source_id)
        self.assertEqual([m["id"] for m in parent], [u1, a1])
        self.assertEqual(len(self.db.get_session_history(result.new_session_id)), 3)


if __name__ == "__main__":
    unittest.main()
