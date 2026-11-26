"""Plan S41 (V65): unscoped ``rag_search`` leaves memory rows out.

Memories are rows of the same ``documents`` table as library chunks, so the
unscoped search (plain-chat RAG and ``@library``) used to return them as
documents: raw memory JSON in the context and in the sources. They are now
dropped from both channels, beside the help rows, before the relevance floor
and fusion. Memories are searched by ``memory_search``.

The first three tests fail on the old code.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.rag_tool import rag_search  # noqa: E402

MEMORY_SOURCE = "qube_memory::preference::coffee"
MEMORY_TEXT = json.dumps({"content": "The user takes their coffee as a flat white.", "category": "preference"})


class _FakeQuery:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def where(self, _clause, *_a, **_k):
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return [dict(r) for r in self._rows]


class _FakeTable:
    def __init__(self, vector_rows, fts_rows=None):
        self._vector_rows = vector_rows
        self._fts_rows = fts_rows or []

    def search(self, query, query_type=None, **_kwargs):
        return _FakeQuery(self._fts_rows if query_type == "fts" else self._vector_rows)


class _FakeStore:
    def __init__(self, vector_rows, fts_rows=None):
        self.table = _FakeTable(vector_rows, fts_rows)


def _row(source: str, text: str, distance: float | None = None, chunk_id: int = 0) -> dict:
    row = {"source": source, "text": text, "chunk_id": chunk_id}
    if distance is not None:
        row["_distance"] = distance
    return row


def _sources(out: dict) -> list[str]:
    return [str(s.get("filename") or s.get("source") or "") for s in out.get("sources") or []]


DOC = "lease_notes.md"
DOC_TEXT = "Pets are allowed with the landlord's written consent."


class TestRagLeavesMemoryRowsOut(unittest.TestCase):
    def test_vector_channel_memory_row_is_dropped(self):
        store = _FakeStore([_row(MEMORY_SOURCE, MEMORY_TEXT, 0.20), _row(DOC, DOC_TEXT, 0.40)])
        out = rag_search("can I keep a cat", np.zeros(8, dtype=np.float32), store, top_k=5)
        self.assertEqual(_sources(out), [DOC])
        self.assertNotIn("flat white", out["llm_context"])

    def test_fts_channel_memory_row_is_dropped(self):
        store = _FakeStore(
            [_row(DOC, DOC_TEXT, 0.40)],
            fts_rows=[_row(MEMORY_SOURCE, MEMORY_TEXT), _row(DOC, DOC_TEXT)],
        )
        out = rag_search("coffee flat white", np.zeros(8, dtype=np.float32), store, top_k=5)
        self.assertEqual(_sources(out), [DOC])

    def test_memory_row_no_longer_keeps_the_vector_channel_alive(self):
        # T4.1: when every vector candidate is under the floor, FTS hits are dropped too.
        # A memory row above the floor used to count as a vector candidate and let the
        # FTS-only chunk through (V65, seen on Fast and Power).
        store = _FakeStore(
            [_row(MEMORY_SOURCE, MEMORY_TEXT, 0.20), _row(DOC, DOC_TEXT, 0.90)],
            fts_rows=[_row(DOC, DOC_TEXT, chunk_id=1)],
        )
        out = rag_search("cat clause", np.zeros(8, dtype=np.float32), store, top_k=5)
        self.assertEqual(out["sources"], [])

    def test_document_rows_unchanged(self):
        store = _FakeStore([_row(DOC, DOC_TEXT, 0.40), _row("budget.md", "Rent is 1,250 a month.", 0.50)])
        out = rag_search("rent", np.zeros(8, dtype=np.float32), store, top_k=5)
        self.assertEqual(sorted(_sources(out)), ["budget.md", DOC])


if __name__ == "__main__":
    unittest.main()
