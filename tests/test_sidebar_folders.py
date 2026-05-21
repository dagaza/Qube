"""Unit tests for sidebar folder schema and DatabaseManager APIs."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.database import DatabaseManager


class SidebarFolderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(str(Path(self._tmpdir.name) / "test_qube.db"))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_fresh_db_creates_main_folders(self) -> None:
        conv = self.db.list_conversation_folders()
        lib = self.db.list_library_folders()
        self.assertEqual(len(conv), 1)
        self.assertEqual(len(lib), 1)
        self.assertEqual(conv[0]["name"], "Main")
        self.assertEqual(lib[0]["name"], "Main")
        self.assertTrue(conv[0]["is_system"])
        self.assertTrue(lib[0]["is_system"])

    def test_backfill_existing_sessions_and_documents(self) -> None:
        main_conv = self.db.get_main_conversation_folder_id()
        main_lib = self.db.get_main_library_folder_id()

        with self.db._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)",
                ("legacy-session", "Legacy"),
            )
            conn.execute(
                """
                INSERT INTO documents (id, filename, file_size_kb, chunk_count)
                VALUES (?, ?, ?, ?)
                """,
                ("legacy-doc", "old.pdf", 1.0, 1),
            )
            conn.commit()
            self.db._ensure_main_folders_and_backfill(conn)

        with self.db._get_connection() as conn:
            sess = conn.execute(
                "SELECT folder_id FROM sessions WHERE id = ?", ("legacy-session",)
            ).fetchone()
            doc = conn.execute(
                "SELECT folder_id FROM documents WHERE filename = ?", ("old.pdf",)
            ).fetchone()
        self.assertEqual(sess["folder_id"], main_conv)
        self.assertEqual(doc["folder_id"], main_lib)

    def test_create_rename_collapse_folder(self) -> None:
        folder_id = self.db.create_conversation_folder("Work")
        self.assertIsNotNone(folder_id)
        names = [f["name"] for f in self.db.list_conversation_folders()]
        self.assertIn("Work", names)

        self.assertTrue(self.db.rename_conversation_folder(folder_id, "Projects"))
        updated = next(
            f for f in self.db.list_conversation_folders() if f["id"] == folder_id
        )
        self.assertEqual(updated["name"], "Projects")

        self.assertTrue(self.db.set_conversation_folder_collapsed(folder_id, True))
        updated = next(
            f for f in self.db.list_conversation_folders() if f["id"] == folder_id
        )
        self.assertTrue(updated["is_collapsed"])

    def test_move_session_between_folders(self) -> None:
        folder_b = self.db.create_conversation_folder("B")
        session_id = self.db.create_session("Chat A")
        main_id = self.db.get_main_conversation_folder_id()

        _, grouped = self.db.get_sessions_for_sidebar_by_folder()
        self.assertTrue(any(s["id"] == session_id for s in grouped[main_id]))

        self.assertTrue(self.db.move_session_to_folder(session_id, folder_b))
        _, grouped = self.db.get_sessions_for_sidebar_by_folder()
        self.assertTrue(any(s["id"] == session_id for s in grouped[folder_b]))

    def test_move_document_between_folders(self) -> None:
        folder_b = self.db.create_library_folder("B")
        self.db.add_document_metadata("doc.txt", 2.0, 3)
        main_id = self.db.get_main_library_folder_id()

        _, grouped = self.db.get_documents_for_sidebar_by_folder()
        self.assertTrue(any(d["filename"] == "doc.txt" for d in grouped[main_id]))

        self.assertTrue(self.db.move_document_to_folder("doc.txt", folder_b))
        _, grouped = self.db.get_documents_for_sidebar_by_folder()
        self.assertTrue(any(d["filename"] == "doc.txt" for d in grouped[folder_b]))

    def test_main_folder_delete_blocked(self) -> None:
        main_id = self.db.get_main_conversation_folder_id()
        self.db.create_session("Keep me")
        self.assertFalse(self.db.delete_conversation_folder(main_id))

        lib_main = self.db.get_main_library_folder_id()
        self.db.add_document_metadata("keep.pdf", 1.0, 1)
        ok, _ = self.db.delete_library_folder(lib_main)
        self.assertFalse(ok)

    def test_cascade_delete_conversation_folder(self) -> None:
        folder_id = self.db.create_conversation_folder("Temp")
        session_id = self.db.create_session("To delete", folder_id=folder_id)
        self.db.add_message(session_id, "user", "hello")

        self.assertTrue(self.db.delete_conversation_folder(folder_id))
        self.assertIsNone(self.db.get_session_folder_id(session_id))
        with self.db._get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_cascade_delete_library_folder_returns_filenames(self) -> None:
        folder_id = self.db.create_library_folder("Temp")
        self.db.add_document_metadata("a.txt", 1.0, 1, folder_id=folder_id)
        self.db.add_document_metadata("b.txt", 2.0, 2, folder_id=folder_id)

        ok, filenames = self.db.delete_library_folder(folder_id)
        self.assertTrue(ok)
        self.assertEqual(set(filenames), {"a.txt", "b.txt"})
        with self.db._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        self.assertEqual(count, 0)

    def test_search_includes_folder_fields(self) -> None:
        conv_folder = self.db.create_conversation_folder("Archive")
        session_id = self.db.create_session("Find me", folder_id=conv_folder)
        self.db.add_message(session_id, "user", "needle")

        results = self.db.get_sessions_for_sidebar_search("needle")
        hit = next(r for r in results if r["id"] == session_id)
        self.assertEqual(hit["folder_id"], conv_folder)

        lib_folder = self.db.create_library_folder("Docs")
        self.db.add_document_metadata("needle.pdf", 1.0, 1, folder_id=lib_folder)
        docs = self.db.get_library_documents_for_sidebar_search("needle")
        self.assertTrue(any(d["filename"] == "needle.pdf" for d in docs))


if __name__ == "__main__":
    unittest.main()
