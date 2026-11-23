"""Tests for per-session sticky library file scope."""

import unittest
from unittest.mock import MagicMock, patch

from core.composer_attachments import ComposerAttachment
from core.session_file_scope import (
    clear_sticky_library_filename,
    get_sticky_library_filename,
    merge_session_sticky_file_into_attachments,
    set_sticky_library_filename,
    sticky_filename_from_routing,
)


class TestSessionFileScope(unittest.TestCase):
    def setUp(self) -> None:
        clear_sticky_library_filename("sess-a")
        clear_sticky_library_filename("sess-b")

    def test_sticky_filename_from_routing_prefers_file(self) -> None:
        routing = [
            ComposerAttachment(kind="tool", id="library", label="Library"),
            ComposerAttachment(kind="file", id="report.pdf", label="report.pdf"),
        ]
        self.assertEqual(sticky_filename_from_routing(routing), "report.pdf")
        self.assertIsNone(sticky_filename_from_routing([]))

    def test_memory_cache_roundtrip(self) -> None:
        set_sticky_library_filename("sess-a", "notes.pdf")
        self.assertEqual(get_sticky_library_filename("sess-a"), "notes.pdf")
        clear_sticky_library_filename("sess-a")
        self.assertIsNone(get_sticky_library_filename("sess-a"))

    def test_db_backed_get_and_set(self) -> None:
        db = MagicMock()
        db.get_session_sticky_library_filename.return_value = "ledger.xlsx"
        self.assertEqual(
            get_sticky_library_filename("sess-b", db=db),
            "ledger.xlsx",
        )
        db.get_session_sticky_library_filename.assert_called_once_with("sess-b")

        set_sticky_library_filename("sess-b", "memo.txt", db=db)
        db.set_session_sticky_library_filename.assert_called_once_with(
            "sess-b", "memo.txt"
        )

        clear_sticky_library_filename("sess-b", db=db)
        db.set_session_sticky_library_filename.assert_called_with("sess-b", None)

    def test_merge_adds_sticky_when_no_file_attachment(self) -> None:
        set_sticky_library_filename("sess-a", "policy.pdf")
        merged = merge_session_sticky_file_into_attachments(
            [ComposerAttachment(kind="tool", id="library", label="Library")],
            "sess-a",
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].kind, "file")
        self.assertEqual(merged[0].id, "policy.pdf")

    def test_merge_skips_when_turn_already_has_file(self) -> None:
        set_sticky_library_filename("sess-a", "other.pdf")
        existing = ComposerAttachment(kind="file", id="chosen.pdf", label="chosen.pdf")
        merged = merge_session_sticky_file_into_attachments([existing], "sess-a")
        self.assertEqual(merged, [existing])

    def test_merge_noop_without_sticky(self) -> None:
        rows = [
            ComposerAttachment(kind="tool", id="web", label="Web"),
        ]
        merged = merge_session_sticky_file_into_attachments(rows, "sess-a")
        self.assertEqual(merged, rows)

    @patch(
        "core.app_settings.get_composer_keep_document_attached_for_chat",
        return_value=False,
    )
    def test_merge_honors_keep_attached_setting(self, _mock: MagicMock) -> None:
        set_sticky_library_filename("sess-a", "policy.pdf")
        rows = [ComposerAttachment(kind="tool", id="library", label="Library")]
        merged = merge_session_sticky_file_into_attachments(rows, "sess-a")
        self.assertEqual(merged, rows)


if __name__ == "__main__":
    unittest.main()