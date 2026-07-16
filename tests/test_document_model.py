"""Tests for canonical Document model (M3)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.document.types import (  # noqa: E402
    Document,
    DocumentMetadata,
    DocumentSection,
    DocumentTable,
)


class TestDocumentModel(unittest.TestCase):
    def test_document_total_text_chars(self) -> None:
        doc = Document(
            url="https://example.com",
            title="Title",
            sections=[
                DocumentSection(heading="H1", level=1, text="Hello"),
                DocumentSection(heading="H2", level=2, text="World"),
            ],
        )
        self.assertEqual(doc.total_text_chars, 10)

    def test_document_metadata_fields(self) -> None:
        meta = DocumentMetadata(
            extractor_name="TrafilaturaExtractor",
            extractor_version="1.0.0",
            extractor_confidence=0.3,
            fetch_tier="http",
            language="en",
        )
        doc = Document(url="https://example.com", title="T", metadata=meta)
        self.assertEqual(doc.metadata.extractor_name, "TrafilaturaExtractor")
        self.assertEqual(doc.metadata.language, "en")

    def test_document_table_shape(self) -> None:
        table = DocumentTable(
            caption="Specs",
            headers=("Name", "Value"),
            rows=(("Weight", "2 kg"),),
        )
        doc = Document(
            url="https://example.com",
            title="Product",
            tables=[table],
        )
        self.assertEqual(doc.tables[0].headers, ("Name", "Value"))


if __name__ == "__main__":
    unittest.main()
