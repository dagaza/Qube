"""Tests for per-document Library ingest modes."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.database import DatabaseManager
from core.library_ingest_modes import (
    INGEST_MODE_PRECISION,
    INGEST_MODE_STANDARD,
    is_precision_ingest_mode,
    normalize_ingest_mode,
)
from core.library_pro_features import resolve_import_ingest_mode


class IngestModeHelpersTests(unittest.TestCase):
    def test_normalize_ingest_mode(self) -> None:
        self.assertEqual(normalize_ingest_mode(None), INGEST_MODE_STANDARD)
        self.assertEqual(normalize_ingest_mode("precision"), INGEST_MODE_PRECISION)
        self.assertEqual(normalize_ingest_mode("standard"), INGEST_MODE_STANDARD)

    def test_resolve_import_without_license(self) -> None:
        self.assertEqual(
            resolve_import_ingest_mode(INGEST_MODE_PRECISION),
            INGEST_MODE_STANDARD,
        )


class DocumentIngestModePersistenceTests(unittest.TestCase):
    def test_add_document_metadata_stores_ingest_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            db = DatabaseManager(db_path)
            doc_id = db.add_document_metadata(
                "contract.pdf",
                120.0,
                12,
                ingest_mode=INGEST_MODE_PRECISION,
            )
            self.assertIsNotNone(doc_id)
            docs = db.get_library_documents(limit=10)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["ingest_mode"], INGEST_MODE_PRECISION)
            self.assertTrue(is_precision_ingest_mode(docs[0]["ingest_mode"]))


if __name__ == "__main__":
    unittest.main()
