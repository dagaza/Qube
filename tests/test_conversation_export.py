"""Unit tests for conversation Markdown / ZIP export."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.conversation_export import (
    export_conversation_markdown,
    export_folder_zip,
    format_conversation_markdown,
    sanitize_export_filename,
)
from core.database import DatabaseManager


class ConversationExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(str(Path(self._tmpdir.name) / "test.db"))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_sanitize_export_filename(self) -> None:
        self.assertEqual(sanitize_export_filename("  Hello / World  "), "Hello _ World")
        self.assertEqual(sanitize_export_filename(""), "Untitled")

    def test_format_conversation_markdown(self) -> None:
        md = format_conversation_markdown(
            "Test Chat",
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )
        self.assertIn("# Test Chat", md)
        self.assertIn("## User", md)
        self.assertIn("Hi", md)
        self.assertIn("## Assistant", md)
        self.assertIn("Hello", md)

    def test_export_single_conversation(self) -> None:
        folder_id = self.db.get_main_conversation_folder_id()
        session_id = self.db.create_session("My Title", folder_id=folder_id)
        self.db.add_message(session_id, "user", "Question")
        self.db.add_message(session_id, "assistant", "Answer")

        out = Path(self._tmpdir.name) / "My Title.md"
        self.assertTrue(export_conversation_markdown(self.db, session_id, out))
        text = out.read_text(encoding="utf-8")
        self.assertIn("# My Title", text)
        self.assertIn("Question", text)
        self.assertIn("Answer", text)

    def test_export_folder_zip(self) -> None:
        folder_id = self.db.create_conversation_folder("Backup")
        s1 = self.db.create_session("Alpha", folder_id=folder_id)
        s2 = self.db.create_session("Beta", folder_id=folder_id)
        self.db.add_message(s1, "user", "a")
        self.db.add_message(s2, "user", "b")

        zip_path = Path(self._tmpdir.name) / "Backup.zip"
        count = export_folder_zip(self.db, folder_id, zip_path)
        self.assertEqual(count, 2)
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, ["Alpha.md", "Beta.md"])
            self.assertIn("a", zf.read("Alpha.md").decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
