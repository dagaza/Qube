"""Tests for Phase 2 shared Document IR + structural chunking."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "ebooklib" not in sys.modules:
    ebooklib_mod = types.ModuleType("ebooklib")
    epub_mod = types.ModuleType("ebooklib.epub")
    epub_mod.read_epub = lambda *_a, **_k: None
    ebooklib_mod.epub = epub_mod
    ebooklib_mod.ITEM_DOCUMENT = 9
    sys.modules["ebooklib"] = ebooklib_mod
    sys.modules["ebooklib.epub"] = epub_mod

from core.chunking.embed_context import library_chunk_embed_text
from core.chunking.ingest_pipeline import chunk_document_for_ingest
from core.knowledge.document.builders.library_builder import (
    build_document_from_markdown,
    build_document_from_path,
)
from core.knowledge.document.builders.plain_text_sections import split_plain_text_sections
from core.knowledge.document.types import Document, DocumentSection


class PlainTextSectionTests(unittest.TestCase):
    def test_detects_all_caps_heading(self) -> None:
        text = "INTRODUCTION\n\nBody paragraph one.\n\nDETAILS\n\nBody paragraph two."
        sections = split_plain_text_sections(text)
        headings = [heading for heading, _level, _body in sections if heading]
        self.assertIn("INTRODUCTION", headings)
        self.assertIn("DETAILS", headings)

    def test_detects_numbered_heading(self) -> None:
        text = "1. Setup\n\nInstall dependencies.\n\n2. Configure\n\nEdit settings."
        sections = split_plain_text_sections(text)
        self.assertGreaterEqual(len(sections), 2)


class MarkdownDocumentBuilderTests(unittest.TestCase):
    def test_build_document_splits_headings(self) -> None:
        text = "# Title\n\n## Alpha\nAlpha body.\n\n## Beta\nBeta body."
        document = build_document_from_markdown(text, title="Doc")
        headings = [section.heading for section in document.sections]
        self.assertIn("Alpha", headings)
        self.assertIn("Beta", headings)


class IngestPipelineTests(unittest.TestCase):
    def test_breadcrumb_tracks_heading_hierarchy(self) -> None:
        document = Document(
            url="guide.md",
            title="Guide",
            sections=[
                DocumentSection(heading="Chapter 1", level=1, text="Intro text."),
                DocumentSection(
                    heading="Section 1.1",
                    level=2,
                    text="Detailed steps for setup and configuration.",
                ),
            ],
        )
        records = chunk_document_for_ingest(document, max_chars=500)
        self.assertTrue(records)
        nested = next((r for r in records if r.heading == "Section 1.1"), None)
        self.assertIsNotNone(nested)
        assert nested is not None
        self.assertIn("Chapter 1", nested.breadcrumb)
        self.assertIn("Section 1.1", nested.breadcrumb)

    def test_embed_uses_breadcrumb(self) -> None:
        embed = library_chunk_embed_text(
            "guide.md",
            "Body",
            section_heading="Section 1.1",
            breadcrumb="Chapter 1 > Section 1.1",
        )
        self.assertIn("Section: Chapter 1 > Section 1.1", embed)


class PdfDocumentBuilderTests(unittest.TestCase):
    def test_pdf_document_has_page_spans_and_single_section(self) -> None:
        class _FakePage:
            def __init__(self, text: str) -> None:
                self._text = text

            def get_text(self) -> str:
                return self._text

        class _FakeDoc:
            def __init__(self, pages: list[_FakePage]) -> None:
                self._pages = pages

            def __iter__(self):
                return iter(self._pages)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        fitz_mod = types.ModuleType("fitz")
        fitz_mod.open = lambda _path: _FakeDoc(
            [
                _FakePage("Start of paragraph"),
                _FakePage("continuation on page two."),
            ]
        )
        sys.modules["fitz"] = fitz_mod

        if "core.knowledge.document.builders.library_builder" in sys.modules:
            del sys.modules["core.knowledge.document.builders.library_builder"]

        from core.knowledge.document.builders import library_builder

        document = library_builder._build_pdf_document(Path("sample.pdf"))
        self.assertEqual(len(document.sections), 1)
        self.assertIn("continuation", document.sections[0].text)
        spans = document.structured_data.get("page_spans") or []
        self.assertEqual(len(spans), 2)

        records = chunk_document_for_ingest(document, max_chars=200)
        self.assertTrue(records)
        if records[0].page_start is not None:
            self.assertGreaterEqual(records[0].page_start, 1)


class WebChunkerCompatTests(unittest.TestCase):
    def test_section_chunker_reexport_unchanged_defaults(self) -> None:
        from core.knowledge.fetch.section_chunker import (
            DEFAULT_MAX_SECTION_CHARS,
            chunk_document,
        )

        document = Document(
            url="https://example.com",
            title="Article",
            sections=[
                DocumentSection(
                    heading="Body",
                    level=2,
                    text="Short section.",
                )
            ],
        )
        chunks = chunk_document(document, max_section_chars=DEFAULT_MAX_SECTION_CHARS)
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks[0].text), DEFAULT_MAX_SECTION_CHARS)


if __name__ == "__main__":
    unittest.main()
