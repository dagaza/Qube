"""Tests for Phase 3 persisted chunk metadata and breadcrumb SOURCE headers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = types.ModuleType("lancedb")
if "pyarrow" not in sys.modules:
    pa = types.ModuleType("pyarrow")

    def _noop(*_args, **_kwargs):
        return None

    pa.schema = _noop
    pa.field = _noop
    pa.list_ = _noop
    pa.float32 = _noop
    pa.utf8 = _noop
    pa.int32 = _noop
    sys.modules["pyarrow"] = pa

if "ebooklib" not in sys.modules:
    ebooklib_mod = types.ModuleType("ebooklib")
    epub_mod = types.ModuleType("ebooklib.epub")
    epub_mod.read_epub = lambda *_a, **_k: None
    ebooklib_mod.epub = epub_mod
    ebooklib_mod.ITEM_DOCUMENT = 9
    sys.modules["ebooklib"] = ebooklib_mod
    sys.modules["ebooklib.epub"] = epub_mod

from core.chunking.chunk_metadata import (
    chunk_record_to_meta_json,
    format_rag_source_header,
    parse_meta_json,
)
from core.chunking.library_preview import (
    build_library_preview,
    build_library_preview_plain,
    preview_section_label,
)
from core.chunking.structure_chunker import ChunkRecord
from mcp import rag_tool


class ChunkMetadataTests(unittest.TestCase):
    def test_chunk_record_round_trip(self) -> None:
        record = ChunkRecord(
            body="Install steps",
            heading="Installation",
            heading_level=2,
            breadcrumb="Setup > Linux > Installation",
            section_index=1,
            chunk_index=0,
            total_chunks=3,
            page_start=2,
            page_end=3,
        )
        raw = chunk_record_to_meta_json(record)
        meta = parse_meta_json(raw)
        self.assertEqual(meta["breadcrumb"], "Setup > Linux > Installation")
        self.assertEqual(meta["heading"], "Installation")
        self.assertEqual(meta["page_start"], 2)

    def test_empty_meta_when_no_structure(self) -> None:
        record = ChunkRecord(
            body="plain",
            heading=None,
            heading_level=0,
            breadcrumb="",
            section_index=0,
            chunk_index=0,
        )
        self.assertEqual(chunk_record_to_meta_json(record), "")

    def test_format_rag_source_header(self) -> None:
        header = format_rag_source_header(
            "guide.md",
            {"breadcrumb": "GPU Setup > Linux"},
        )
        self.assertEqual(header, "guide.md — § GPU Setup > Linux")


class LibraryPreviewTests(unittest.TestCase):
    def test_preview_section_label_includes_page_range(self) -> None:
        label = preview_section_label(
            {"breadcrumb": "Ch 1 > Intro", "page_start": 2, "page_end": 4}
        )
        self.assertEqual(label, "Ch 1 > Intro (pp. 2–4)")

    def test_build_library_preview_plain_legacy(self) -> None:
        text = build_library_preview_plain(
            [
                {"chunk_id": 1, "text": "Second"},
                {"chunk_id": 0, "text": "First"},
            ]
        )
        self.assertEqual(text, "First\n\nSecond")

    def test_build_library_preview_html_with_breadcrumbs(self) -> None:
        rows = [
            {
                "chunk_id": 0,
                "text": "Intro body.",
                "meta_json": '{"breadcrumb":"Setup > Linux"}',
            },
            {
                "chunk_id": 1,
                "text": "More in same section.",
                "meta_json": '{"breadcrumb":"Setup > Linux"}',
            },
            {
                "chunk_id": 2,
                "text": "Install steps.",
                "meta_json": '{"breadcrumb":"Setup > Linux > Installation"}',
            },
        ]
        content, is_html = build_library_preview(
            rows,
            breadcrumb_color="#888",
            body_color="#eee",
            font_pt=12.0,
        )
        self.assertTrue(is_html)
        self.assertEqual(content.count("§ Setup &gt; Linux</p>"), 1)
        self.assertIn("§ Setup &gt; Linux &gt; Installation", content)
        self.assertIn("Intro body.", content)
        self.assertIn("Install steps.", content)

    def test_build_library_preview_falls_back_without_metadata(self) -> None:
        content, is_html = build_library_preview(
            [{"chunk_id": 0, "text": "Legacy chunk.", "meta_json": ""}],
            breadcrumb_color="#888",
            body_color="#eee",
        )
        self.assertFalse(is_html)
        self.assertEqual(content, "Legacy chunk.")


class _FakeQuery:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def limit(self, _n: int) -> "_FakeQuery":
        return self

    def where(self, _clause: str) -> "_FakeQuery":
        return self

    def to_list(self) -> list[dict]:
        return self._rows


class _FakeTable:
    def __init__(self, vector_rows: list[dict]) -> None:
        self._vector_rows = vector_rows

    def search(self, query, query_type: str | None = None) -> _FakeQuery:
        if query_type == "fts":
            return _FakeQuery([])
        return _FakeQuery(self._vector_rows)


class _FakeStore:
    def __init__(self, vector_rows: list[dict]) -> None:
        self.table = _FakeTable(vector_rows)


class RagSourceHeaderTests(unittest.TestCase):
    def test_rag_search_source_block_includes_breadcrumb(self) -> None:
        meta = json.dumps({"breadcrumb": "Chapter 1 > Intro"})
        store = _FakeStore(
            [
                {
                    "source": "manual.pdf",
                    "text": "Overview paragraph.",
                    "_distance": 0.1,
                    "chunk_id": 0,
                    "meta_json": meta,
                }
            ]
        )
        result = rag_tool.rag_search(
            "overview",
            np.zeros(4, dtype=np.float32),
            store,
        )
        self.assertIn("manual.pdf — § Chapter 1 > Intro", result["llm_context"])


class DocumentStoreMetaJsonTests(unittest.TestCase):
    def test_normalize_chunk_row_adds_empty_meta_json(self) -> None:
        from rag.store import _normalize_chunk_row

        row = _normalize_chunk_row(
            {"vector": [0.1], "text": "hi", "source": "a.txt", "chunk_id": 0}
        )
        self.assertIn("meta_json", row)
        self.assertEqual(row["meta_json"], "")

    def test_normalize_chunk_row_serializes_dict(self) -> None:
        from rag.store import _normalize_chunk_row

        row = _normalize_chunk_row(
            {
                "vector": [0.1],
                "text": "hi",
                "source": "a.txt",
                "chunk_id": 0,
                "meta_json": {"breadcrumb": "A > B"},
            }
        )
        self.assertEqual(json.loads(row["meta_json"])["breadcrumb"], "A > B")


if __name__ == "__main__":
    unittest.main()
