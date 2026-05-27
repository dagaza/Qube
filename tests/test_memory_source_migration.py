"""Tests for pre-T3.4 memory source migration (Option A + C)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.memory_source_migration import (
    is_unnamespaced_legacy_source,
    legacy_namespaced_source,
    migrate_legacy_memory_sources,
)


class LegacySourceDetectionTests(unittest.TestCase):
    def test_two_segment_source_is_legacy(self):
        self.assertTrue(is_unnamespaced_legacy_source("qube_memory::preference"))
        self.assertTrue(is_unnamespaced_legacy_source("qube_memory::knowledge"))

    def test_three_segment_tier_source_is_not_legacy(self):
        self.assertFalse(is_unnamespaced_legacy_source("qube_memory::preference::identity"))
        self.assertFalse(is_unnamespaced_legacy_source("qube_memory::knowledge::knowledge"))
        self.assertFalse(is_unnamespaced_legacy_source("qube_memory::episode::sess-1"))
        self.assertFalse(is_unnamespaced_legacy_source("qube_memory::legacy::preference"))

    def test_legacy_namespaced_source(self):
        self.assertEqual(
            legacy_namespaced_source("preference"),
            "qube_memory::legacy::preference",
        )


class _FakeTable:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self.add_calls: list[list[dict]] = []
        self.delete_calls: list[str] = []

    def search(self):
        return self

    def select(self, _fields):
        return self

    def where(self, _clause: str):
        return self

    def limit(self, _n: int):
        return self

    def to_list(self):
        return list(self._rows)

    def delete(self, clause: str):
        self.delete_calls.append(clause)
        if "_rowid = 1" in clause:
            self._rows = [r for r in self._rows if r.get("_rowid") != 1]
        elif "id = 'row-1'" in clause:
            self._rows = [r for r in self._rows if r.get("id") != "row-1"]

    def add(self, records: list[dict]):
        self.add_calls.append(records)


class _FakeStore:
    def __init__(self, rows: list[dict]) -> None:
        self.table = _FakeTable(rows)


class MigrateLegacyMemorySourcesTests(unittest.TestCase):
    def test_migrates_unnamespaced_rows(self):
        rows = [{
            "_rowid": 1,
            "text": '{"type":"fact","content":"likes tea"}',
            "vector": [0.1, 0.2],
            "source": "qube_memory::preference",
            "chunk_id": 0,
        }]
        store = _FakeStore(rows)
        count = migrate_legacy_memory_sources(store)
        self.assertEqual(count, 1)
        self.assertEqual(len(store.table.delete_calls), 1)
        self.assertEqual(len(store.table.add_calls), 1)
        self.assertEqual(
            store.table.add_calls[0][0]["source"],
            "qube_memory::legacy::preference",
        )

    def test_skips_already_tiered_rows(self):
        rows = [{
            "_rowid": 2,
            "text": '{"type":"fact","content":"fact"}',
            "vector": [0.1],
            "source": "qube_memory::knowledge::knowledge",
            "chunk_id": 0,
        }]
        store = _FakeStore(rows)
        count = migrate_legacy_memory_sources(store)
        self.assertEqual(count, 0)
        self.assertEqual(store.table.delete_calls, [])


if __name__ == "__main__":
    unittest.main()
