"""Guards for Qube-managed Library documents (help corpus, memory tier exports)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.database import DatabaseManager
from core.help_corpus_manifest import help_doc_source
from core.library_folder_policy import (
    is_protected_library_document,
    is_qube_managed_document_filename,
)


class ProtectedLibraryDocumentTests(unittest.TestCase):
    def test_protected_alias_matches_qube_managed(self) -> None:
        sample = "qube/documentation/faq/install-linux.md"
        self.assertTrue(is_qube_managed_document_filename(sample))
        self.assertTrue(is_protected_library_document(sample))
        self.assertFalse(is_protected_library_document("reports/notes.md"))

    def test_delete_metadata_refuses_protected_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(db_path=os.path.join(tmp, "t.db"))
            source = help_doc_source("00-index.md")
            db.add_document_metadata(source, 1.0, 3, folder_id=db.get_qube_library_folder_id())
            self.assertFalse(db.delete_document_metadata(source))
            row = db.get_documents_for_sidebar_by_folder()
            _folders, grouped = row
            qube_id = db.get_qube_library_folder_id()
            names = [d["filename"] for d in grouped.get(qube_id, [])]
            self.assertIn(source, names)
            self.assertTrue(db.delete_document_metadata(source, force=True))
            row = db.get_documents_for_sidebar_by_folder()
            _folders, grouped = row
            names = [d["filename"] for d in grouped.get(qube_id, [])]
            self.assertNotIn(source, names)


if __name__ == "__main__":
    unittest.main()