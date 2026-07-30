"""Tests for Phase 1 Library chunking helpers."""

from __future__ import annotations

import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Stub optional parser deps so ``rag.parsers`` loads in minimal CI envs.
if "ebooklib" not in sys.modules:
    ebooklib_mod = types.ModuleType("ebooklib")
    epub_mod = types.ModuleType("ebooklib.epub")
    epub_mod.read_epub = lambda *_a, **_k: None
    ebooklib_mod.epub = epub_mod
    ebooklib_mod.ITEM_DOCUMENT = 9
    sys.modules["ebooklib"] = ebooklib_mod
    sys.modules["ebooklib.epub"] = epub_mod

from pathlib import Path

from core.chunking.embed_context import heading_from_chunk, library_chunk_embed_text
from core.chunking.library_chunking import chunk_library_text


class EmbedContextTests(unittest.TestCase):
    def test_library_embed_prefix_includes_document_and_section(self) -> None:
        chunk = "## Installation\nRun the installer."
        embed = library_chunk_embed_text("guide.md", chunk)
        self.assertIn("Document: guide.md", embed)
        self.assertIn("Section: Installation", embed)
        self.assertIn("Run the installer.", embed)

    def test_library_embed_includes_document_even_without_heading(self) -> None:
        chunk = "Plain paragraph without headings."
        embed = library_chunk_embed_text("notes.txt", chunk)
        self.assertIn("Document: notes.txt", embed)
        self.assertIn(chunk, embed)
        self.assertNotIn("Section:", embed)

    def test_heading_from_chunk(self) -> None:
        self.assertEqual(
            heading_from_chunk("### Linux packages\nInstall deps."),
            "Linux packages",
        )


class LibraryChunkingTests(unittest.TestCase):
    def test_markdown_preserves_heading_sections(self) -> None:
        text = "## GPU layers\nAdjust offload.\n\n## Chat template\nPick a family."
        chunks = chunk_library_text(text, path_suffix=".md")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("GPU layers" in chunk for chunk in chunks))
        self.assertTrue(any("Chat template" in chunk for chunk in chunks))

    def test_plain_text_uses_sliding_window(self) -> None:
        text = "word " * 400
        chunks = chunk_library_text(text, path_suffix=".txt", max_chars=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))


class PdfParserTests(unittest.TestCase):
    def test_parse_file_delegates_to_document_builder(self) -> None:
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
            [_FakePage("Page one."), _FakePage("Page two.")]
        )
        sys.modules["fitz"] = fitz_mod

        for mod in (
            "core.knowledge.document.builders.library_builder",
            "rag.parsers",
        ):
            if mod in sys.modules:
                del sys.modules[mod]

        from rag.parsers import parse_file

        sections = parse_file(Path("sample.pdf"))
        self.assertEqual(len(sections), 1)
        self.assertIn("Page one.", sections[0])
        self.assertIn("Page two.", sections[0])


if __name__ == "__main__":
    unittest.main()
