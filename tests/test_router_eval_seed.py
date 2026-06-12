"""Tests for router-eval fixture seeding."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_eval_seed import (
    default_eval_lancedb_dir,
    is_eval_library_seeded,
    list_library_fixture_paths,
    load_memory_fixtures,
    seed_router_eval_library,
)


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            vec = np.zeros(768, dtype=np.float32)
            vec[0] = min(1.0, len(text) / 1000.0)
            out.append(vec)
        return np.array(out, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class _FakeTable:
    def __init__(self):
        self.rows: list[dict] = []
        self._names = ["documents"]

    def add(self, chunks: list[dict]) -> None:
        self.rows.extend(chunks)

    def table_names(self):
        return self._names

    def create_fts_index(self, *_args, **_kwargs) -> None:
        return None

    def search(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def where(self, *_args, **_kwargs):
        return self

    def to_list(self):
        return []


class _FakeStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.table = _FakeTable()
        self._sources: set[str] = set()

    def source_exists(self, source: str) -> bool:
        return source in self._sources

    def get_all_indexed_sources(self) -> list[str]:
        return sorted(self._sources)

    def add_chunks(self, chunks: list[dict]) -> None:
        self.table.add(chunks)
        for row in chunks:
            self._sources.add(str(row["source"]))

    def delete_document(self, source: str) -> None:
        self._sources.discard(source)
        self.table.rows = [r for r in self.table.rows if r.get("source") != source]


class RouterEvalSeedTests(unittest.TestCase):
    def test_fixtures_present(self) -> None:
        paths = list_library_fixture_paths()
        self.assertGreaterEqual(len(paths), 8)
        memories = load_memory_fixtures()
        self.assertGreaterEqual(len(memories), 15)

    def test_seed_writes_library_and_memory_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_dir = Path(td) / "lancedb"
            store = _FakeStore(db_dir)
            with mock.patch(
                "core.router_eval_seed._is_safe_force_purge_dir", return_value=True
            ):
                summary = seed_router_eval_library(store, _FakeEmbedder(), force=True)
            self.assertFalse(summary["skipped"])
            self.assertGreater(summary["library_chunks"], 0)
            self.assertGreater(summary["memory_rows"], 0)
            self.assertTrue(is_eval_library_seeded(db_dir))

            sources = store.get_all_indexed_sources()
            self.assertTrue(any(s.startswith("eval_") for s in sources))
            self.assertTrue(any(s.startswith("qube_memory::") for s in sources))

    def test_seed_skips_when_manifest_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_dir = Path(td) / "lancedb"
            store = _FakeStore(db_dir)
            with mock.patch(
                "core.router_eval_seed._is_safe_force_purge_dir", return_value=True
            ):
                seed_router_eval_library(store, _FakeEmbedder(), force=True)
            before = len(store.table.rows)
            summary = seed_router_eval_library(store, _FakeEmbedder(), force=False)
            self.assertTrue(summary["skipped"])
            self.assertEqual(len(store.table.rows), before)

    def test_default_eval_lancedb_under_eval(self) -> None:
        path = default_eval_lancedb_dir()
        self.assertEqual(path.parent.name, "eval")


if __name__ == "__main__":
    unittest.main()
