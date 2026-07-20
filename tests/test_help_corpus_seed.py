"""Tests for bundled help corpus seeding."""

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

from core.help_corpus_manifest import help_doc_source, load_manifest
from core.help_corpus_seed import (
    seed_help_corpus,
    should_seed_help_corpus,
)


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = min(1.0, len(text) / 1000.0)
            out.append(vec)
        return np.array(out, dtype=np.float32)


class _FakeStore:
    def __init__(self) -> None:
        self._sources: set[str] = set()
        self.rows: list[dict] = []

    def source_exists(self, source: str) -> bool:
        return source in self._sources

    def get_all_indexed_sources(self) -> list[str]:
        return sorted(self._sources)

    def add_chunks(self, chunks: list[dict]) -> None:
        self.rows.extend(chunks)
        for row in chunks:
            self._sources.add(str(row["source"]))

    def delete_document(self, source: str) -> None:
        self._sources.discard(source)
        self.rows = [row for row in self.rows if row.get("source") != source]


class _FakeDb:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.qube_folder_id = "folder-qube"

    def get_qube_library_folder_id(self) -> str:
        return self.qube_folder_id

    def add_document_metadata(
        self,
        filename: str,
        file_size_kb: float,
        chunk_count: int,
        folder_id: str | None = None,
        summary_blurb: str | None = None,
    ) -> None:
        self.docs[filename] = {
            "filename": filename,
            "file_size_kb": file_size_kb,
            "chunk_count": chunk_count,
            "folder_id": folder_id,
            "summary_blurb": summary_blurb,
        }

    def delete_document_metadata(self, filename: str) -> None:
        self.docs.pop(filename, None)


class HelpCorpusSeedTests(unittest.TestCase):
    def test_should_seed_when_no_state(self) -> None:
        manifest = load_manifest()
        with mock.patch(
            "core.help_corpus_seed.user_help_corpus_state_path",
            return_value=Path("/nonexistent/help_corpus_state.json"),
        ):
            need, reason = should_seed_help_corpus(manifest)
        self.assertTrue(need)
        self.assertIn("no prior", reason)

    def test_seed_indexes_bundled_index(self) -> None:
        manifest = load_manifest()
        rel = str(manifest["documents"][0]["path"])
        expected_source = help_doc_source(rel)

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            state_path = Path(td) / "help_corpus_state.json"
            with mock.patch(
                "core.help_corpus_seed.user_help_corpus_state_path",
                return_value=state_path,
            ):
                store = _FakeStore()
                db = _FakeDb()
                summary = seed_help_corpus(store, _FakeEmbedder(), db, force=True)

            self.assertFalse(summary["skipped"])
            self.assertGreater(summary["chunks"], 0)
            self.assertIn(expected_source, store.get_all_indexed_sources())
            self.assertIn(expected_source, db.docs)
            self.assertEqual(db.docs[expected_source]["folder_id"], db.qube_folder_id)
            self.assertTrue(state_path.is_file())

    def test_seed_skips_when_state_matches(self) -> None:
        import hashlib

        from core.help_corpus_manifest import bundled_help_locale_dir

        manifest = load_manifest()
        root = bundled_help_locale_dir("en")
        doc_hashes: dict[str, str] = {}
        for doc in manifest["documents"]:
            rel = str(doc["path"])
            source = help_doc_source(rel)
            composed = root / rel
            doc_hashes[source] = hashlib.sha256(composed.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            state_path = Path(td) / "help_corpus_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema": "qube.help_corpus_state.v1",
                        "corpus_version": manifest["corpus_version"],
                        "doc_hashes": doc_hashes,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "core.help_corpus_seed.user_help_corpus_state_path",
                return_value=state_path,
            ):
                store = _FakeStore()
                for source in doc_hashes:
                    store._sources.add(source)
                db = _FakeDb()
                summary = seed_help_corpus(store, _FakeEmbedder(), db, force=False)

            self.assertTrue(summary["skipped"])

    def test_should_seed_when_corpus_version_changes(self) -> None:
        manifest = load_manifest()
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            state_path = Path(td) / "help_corpus_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema": "qube.help_corpus_state.v1",
                        "corpus_version": "0.9.0",
                        "doc_hashes": {},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "core.help_corpus_seed.user_help_corpus_state_path",
                return_value=state_path,
            ):
                need, reason = should_seed_help_corpus(manifest)
            self.assertTrue(need)
            self.assertIn("corpus_version changed", reason)

    def test_seed_on_upgrade_persists_new_corpus_version(self) -> None:
        """Simulate app upgrade: old corpus_version in state → seed → new state."""
        manifest = load_manifest()
        expected_version = str(manifest["corpus_version"])

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            state_path = Path(td) / "help_corpus_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema": "qube.help_corpus_state.v1",
                        "corpus_version": "0.9.0",
                        "doc_hashes": {},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "core.help_corpus_seed.user_help_corpus_state_path",
                return_value=state_path,
            ):
                store = _FakeStore()
                db = _FakeDb()
                summary = seed_help_corpus(store, _FakeEmbedder(), db, force=False)

            self.assertFalse(summary["skipped"])
            self.assertEqual(summary["corpus_version"], expected_version)
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["corpus_version"], expected_version)
            self.assertGreater(summary["chunks"], 0)


if __name__ == "__main__":
    unittest.main()
