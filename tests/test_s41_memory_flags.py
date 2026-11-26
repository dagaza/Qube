"""Plan S41 (V66): ``memory_search`` with ``apply_mmr`` / ``apply_core_memory_gate``.

``memory_search`` takes boolean arguments with the same names as the policy
functions it imported, so inside the function ``apply_mmr(...)`` and
``apply_core_memory_gate(...)`` called a bool. Every MEMORY-route, narrative
and core-memory search with a candidate raised ``TypeError``, and the outer
``except`` returned nothing. The worker always passes one of the two flags on
those routes, and no test passed either flag as True.

Each row below is far from every threshold (semantic floor, soft cutoff,
core-memory score and margin), so the outcome depends only on the policy
functions. The first four tests fail on the shadowed code (the fourth because
the gate never ran: the search logs the error and returns nothing).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.memory_tool import memory_search  # noqa: E402


def _row(rowid: int, content: str, *, distance: float, confidence: float, tier: str = "preference") -> dict:
    payload = {"content": content, "confidence": confidence, "category": tier, "strength": 1, "decay": 1.0}
    return {
        "_rowid": rowid,
        "text": json.dumps(payload),
        "source": f"qube_memory::{tier}::{tier}",
        "_distance": distance,
    }


class _FakeQuery:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def select(self, *_cols):
        return self

    def where(self, _clause, *_a, **_k):
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return [dict(r) for r in self._rows]


class _FakeTable:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def search(self, query=None, query_type=None, **_kwargs):
        return _FakeQuery([] if query_type == "fts" else self._rows)


class _FakeStore:
    def __init__(self, rows: list[dict]):
        self.table = _FakeTable(rows)


def _search(rows: list[dict], **flags) -> list[str]:
    out = memory_search(
        "how do i take my coffee",  # lowercase: no proper-noun gate
        np.zeros(8, dtype=np.float32),
        _FakeStore(rows),
        include_preference=True,
        include_context=True,
        **flags,
    )
    return [s["content"] for s in out.get("memory_sources") or []]


COFFEE = "The user takes their coffee as a flat white with oat milk."
COFFEE_DUP = "The user takes their coffee as a flat white with oat milk!"
TEA = "The user drinks green tea in the afternoon."
BIKE = "The user commutes by bike."


class TestMemorySearchFlags(unittest.TestCase):
    def test_mmr_runs_and_demotes_a_near_duplicate(self):
        rows = [
            _row(1, COFFEE, distance=0.10, confidence=0.9),
            _row(2, COFFEE_DUP, distance=0.11, confidence=0.9),
            _row(3, TEA, distance=0.40, confidence=0.6),
        ]
        with self.assertNoLogs("Qube.MemoryTool", level="ERROR"):
            kept = _search(rows, apply_mmr=True)
        self.assertEqual(kept, [COFFEE, TEA, COFFEE_DUP])

    def test_core_memory_gate_keeps_a_clear_top_hit(self):
        # Scores 0.835 and 0.44: the top is above the 0.45 floor and leads by 0.395 (margin 0.08).
        rows = [
            _row(1, COFFEE, distance=0.10, confidence=0.9),
            _row(2, BIKE, distance=0.60, confidence=0.3),
        ]
        with self.assertNoLogs("Qube.MemoryTool", level="ERROR"):
            kept = _search(rows, top_k=3, apply_core_memory_gate=True)
        self.assertEqual(kept, [COFFEE, BIKE])

    def test_both_flags_together(self):
        rows = [
            _row(1, COFFEE, distance=0.10, confidence=0.9),
            _row(2, BIKE, distance=0.60, confidence=0.3),
        ]
        with self.assertNoLogs("Qube.MemoryTool", level="ERROR"):
            kept = _search(rows, apply_mmr=True, apply_core_memory_gate=True)
        self.assertEqual(kept, [COFFEE, BIKE])

    def test_core_memory_gate_still_suppresses_a_tight_margin(self):
        # Scores 0.835 and 0.830: a 0.005 lead, under the 0.08 margin.
        rows = [
            _row(1, COFFEE, distance=0.10, confidence=0.9),
            _row(2, TEA, distance=0.11, confidence=0.9),
        ]
        with self.assertNoLogs("Qube.MemoryTool", level="ERROR"):
            kept = _search(rows, top_k=3, apply_core_memory_gate=True)
        self.assertEqual(kept, [])

    def test_flags_off_unchanged(self):
        rows = [
            _row(1, COFFEE, distance=0.10, confidence=0.9),
            _row(2, BIKE, distance=0.60, confidence=0.3),
        ]
        self.assertEqual(_search(rows), [COFFEE, BIKE])


if __name__ == "__main__":
    unittest.main()
