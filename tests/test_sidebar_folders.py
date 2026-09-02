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
        self.assertEqual(len(lib), 2)
        self.assertEqual(conv[0]["name"], "Main")
        lib_names = {f["name"] for f in lib}
        self.assertEqual(lib_names, {"Main", "Qube"})
        self.assertTrue(conv[0]["is_system"])
        for folder in lib:
            self.assertTrue(folder["is_system"])
        main = next(f for f in lib if f["name"] == "Main")
        qube = next(f for f in lib if f["name"] == "Qube")
        self.assertTrue(main["allows_user_ingest"])
        self.assertFalse(qube["allows_user_ingest"])
        self.assertEqual(main["folder_key"], "main")
        self.assertEqual(qube["folder_key"], "qube")
        self.assertFalse(main["is_collapsed"])
        self.assertTrue(qube["is_collapsed"])

    def test_qube_folder_stays_expanded_after_user_toggle(self) -> None:
        qube_id = self.db.get_qube_library_folder_id()
        self.assertTrue(self.db.set_library_folder_collapsed(qube_id, False))
        qube = next(
            f for f in self.db.list_library_folders() if f["id"] == qube_id
        )
        self.assertFalse(qube["is_collapsed"])

        # Re-init should not re-collapse once user expanded the folder.
        path = str(Path(self._tmpdir.name) / "test_qube.db")
        self.db = DatabaseManager(path)
        qube = next(
            f for f in self.db.list_library_folders() if f["id"] == qube_id
        )
        self.assertFalse(qube["is_collapsed"])

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
        lib_qube = self.db.get_qube_library_folder_id()
        self.db.add_document_metadata("keep.pdf", 1.0, 1)
        ok, _ = self.db.delete_library_folder(lib_main)
        self.assertFalse(ok)
        ok, _ = self.db.delete_library_folder(lib_qube)
        self.assertFalse(ok)

    def test_qube_folder_blocks_user_ingest_and_moves(self) -> None:
        main_id = self.db.get_main_library_folder_id()
        qube_id = self.db.get_qube_library_folder_id()
        self.assertFalse(self.db.library_folder_allows_user_ingest(qube_id))
        self.assertTrue(self.db.library_folder_allows_user_ingest(main_id))

        self.db.add_document_metadata("user.txt", 1.0, 1, folder_id=main_id)
        self.assertFalse(self.db.move_document_to_folder("user.txt", qube_id))
        self.assertIsNone(self.db.create_library_folder("Qube"))
        self.assertFalse(self.db.rename_library_folder(qube_id, "Renamed"))

    def test_qube_managed_documents_migrate_from_main(self) -> None:
        main_id = self.db.get_main_library_folder_id()
        qube_id = self.db.get_qube_library_folder_id()
        self.db.add_document_metadata("qube/preferences.md", 1.0, 2, folder_id=main_id)
        self.db.add_document_metadata("notes.pdf", 1.0, 1, folder_id=main_id)

        with self.db._get_connection() as conn:
            self.db._migrate_qube_managed_documents_to_qube_folder(conn, main_id, qube_id)
            conn.commit()

        _, grouped = self.db.get_documents_for_sidebar_by_folder()
        qube_names = {d["filename"] for d in grouped[qube_id]}
        main_names = {d["filename"] for d in grouped[main_id]}
        self.assertIn("qube/preferences.md", qube_names)
        self.assertNotIn("qube/preferences.md", main_names)
        self.assertIn("notes.pdf", main_names)

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

    def test_composer_file_picker_excludes_qube_help_corpus(self) -> None:
        main_id = self.db.get_main_library_folder_id()
        qube_id = self.db.get_qube_library_folder_id()
        self.db.add_document_metadata(
            "qube/documentation/features/library.md",
            1.0,
            5,
            folder_id=qube_id,
        )
        self.db.add_document_metadata(
            "my-report.pdf",
            120.0,
            10,
            folder_id=main_id,
        )
        docs = self.db.get_user_library_documents_for_composer()
        filenames = {d["filename"] for d in docs}
        self.assertIn("my-report.pdf", filenames)
        self.assertNotIn("qube/documentation/features/library.md", filenames)

    def test_pin_and_unpin_session(self) -> None:
        session_id = self.db.create_session("Pinned chat")
        self.assertTrue(self.db.set_session_pinned(session_id, True))

        _, grouped = self.db.get_sessions_for_sidebar_by_folder()
        main_id = self.db.get_main_conversation_folder_id()
        hit = next(s for s in grouped[main_id] if s["id"] == session_id)
        self.assertTrue(hit["is_pinned"])

        self.assertTrue(self.db.set_session_pinned(session_id, False))
        _, grouped = self.db.get_sessions_for_sidebar_by_folder()
        hit = next(s for s in grouped[main_id] if s["id"] == session_id)
        self.assertFalse(hit["is_pinned"])

    def test_pin_and_unpin_document(self) -> None:
        self.db.add_document_metadata("pinned.txt", 1.0, 1)
        self.assertTrue(self.db.set_document_pinned("pinned.txt", True))

        _, grouped = self.db.get_documents_for_sidebar_by_folder()
        main_id = self.db.get_main_library_folder_id()
        hit = next(d for d in grouped[main_id] if d["filename"] == "pinned.txt")
        self.assertTrue(hit["is_pinned"])

        self.assertTrue(self.db.set_document_pinned("pinned.txt", False))
        _, grouped = self.db.get_documents_for_sidebar_by_folder()
        hit = next(d for d in grouped[main_id] if d["filename"] == "pinned.txt")
        self.assertFalse(hit["is_pinned"])

    def test_pinned_items_sort_before_unpinned(self) -> None:
        items = [
            {"title": "Alpha", "is_pinned": 0, "updated_at": "2026-01-01"},
            {"title": "Beta", "is_pinned": 1, "updated_at": "2026-01-02"},
            {"title": "Gamma", "is_pinned": 0, "updated_at": "2026-01-03"},
        ]
        pinned = [it for it in items if it.get("is_pinned")]
        unpinned = [it for it in items if not it.get("is_pinned")]
        pinned.sort(key=lambda it: str(it.get("title") or "").lower())
        unpinned.sort(key=lambda it: str(it.get("title") or "").lower())
        ordered = pinned + unpinned
        self.assertEqual([it["title"] for it in ordered], ["Beta", "Alpha", "Gamma"])

    def test_get_all_library_document_filenames_returns_full_set(self) -> None:
        main_id = self.db.get_main_library_folder_id()
        for idx in range(25):
            self.db.add_document_metadata(
                f"bulk-{idx:02d}.txt",
                1.0,
                1,
                folder_id=main_id,
            )

        names = self.db.get_all_library_document_filenames()
        self.assertEqual(len(names), 25)
        self.assertIn("bulk-00.txt", names)
        self.assertIn("bulk-24.txt", names)

    def test_add_document_metadata_skips_duplicate_filename(self) -> None:
        main_id = self.db.get_main_library_folder_id()
        first_id = self.db.add_document_metadata(
            "once.txt",
            1.0,
            1,
            folder_id=main_id,
        )
        second_id = self.db.add_document_metadata(
            "once.txt",
            9.0,
            9,
            folder_id=main_id,
        )

        self.assertIsNotNone(first_id)
        self.assertEqual(second_id, first_id)
        self.assertEqual(
            len([name for name in self.db.get_all_library_document_filenames() if name == "once.txt"]),
            1,
        )

    def test_dedupe_library_document_metadata_keeps_richest_row(self) -> None:
        main_id = self.db.get_main_library_folder_id()
        self.db.add_document_metadata(
            "eval_kubernetes_notes.md",
            0.0,
            0,
            folder_id=main_id,
            allow_duplicate=True,
        )
        self.db.add_document_metadata(
            "eval_kubernetes_notes.md",
            0.0,
            0,
            folder_id=main_id,
            allow_duplicate=True,
        )
        real_id = self.db.add_document_metadata(
            "eval_kubernetes_notes.md",
            0.35,
            1,
            folder_id=main_id,
            allow_duplicate=True,
        )

        removed = self.db.dedupe_library_document_metadata()
        self.assertEqual(removed, 2)

        with self.db._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, file_size_kb, chunk_count FROM documents WHERE filename = ?",
                ("eval_kubernetes_notes.md",),
            ).fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["id"]), real_id)
        self.assertEqual(rows[0]["chunk_count"], 1)
        self.assertEqual(rows[0]["file_size_kb"], 0.35)


if __name__ == "__main__":
    unittest.main()
