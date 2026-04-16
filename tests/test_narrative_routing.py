"""T3.2 — narrative routing through memory_tool.

Exercises ``prefer_episode`` in ``mcp.memory_tool.memory_search``:

- without the flag, atomic-fact rows can outrank an episode summary
- with the flag, the episode summary wins the ranking tie-breaker
- the episode row gets an inline ``[EPISODE]`` label so the LLM can
  identify it once ``NARRATIVE_RECALL_SYSTEM_SUFFIX`` is in scope
- the proper-noun gate is bypassed for episode rows when the caller
  opted into ``prefer_episode=True``
"""
from __future__ import annotations

import json
import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)


def _episode_row(
    rid: str, *, content: str, distance: float, confidence: float = 0.85,
    session_id: str = "sess-ABC",
) -> dict:
    payload = {
        "type": "fact",
        "category": "episode",
        "content": content,
        "confidence": confidence,
        "strength": 1,
        "decay": 1.0,
        "topics": ["refactor"],
        "links_to_document_ids": [],
    }
    return {
        "id": rid,
        "_distance": distance,
        "text": json.dumps(payload),
        "source": f"qube_memory::episode::{session_id}",
        "vector": [0.1, 0.2, 0.3],
    }


def _fact_row(
    rid: str, *, content: str, distance: float, confidence: float = 0.90,
    category: str = "knowledge",
) -> dict:
    payload = {
        "type": "fact",
        "category": category,
        "content": content,
        "confidence": confidence,
        "strength": 1,
        "decay": 1.0,
        "links_to_document_ids": [],
    }
    return {
        "id": rid,
        "_distance": distance,
        "text": json.dumps(payload),
        "source": f"qube_memory::{category}",
        "vector": [0.1, 0.2, 0.3],
    }


class _Search:
    def __init__(self, rows):
        self.rows = rows

    def where(self, *_a, **_kw):
        return self

    def limit(self, *_a, **_kw):
        return self

    def to_list(self):
        return list(self.rows)


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def search(self, *_a, **_kw):
        return _Search(self._rows)


class _Store:
    def __init__(self, rows):
        self.table = _Table(rows)


class TestNarrativeRouting(unittest.TestCase):
    def setUp(self):
        # Import lazily so the stub PyQt6 modules from other tests don't
        # interfere (memory_tool doesn't import Qt, but belt-and-braces).
        from mcp import memory_tool

        self.memory_tool = memory_tool

    def test_episode_outranks_fact_with_prefer_episode(self):
        # Set up rows where the atomic fact is *slightly* closer in
        # embedding space than the episode summary. Without the flag,
        # the fact wins. With prefer_episode, the episode wins.
        rows = [
            _episode_row(
                "ep-1",
                content="User worked on the memory enrichment refactor this session.",
                distance=0.30,
            ),
            _fact_row(
                "fact-1",
                content="User prefers dark roast coffee.",
                distance=0.25,
            ),
        ]
        store = _Store(rows)

        baseline = self.memory_tool.memory_search(
            query="recap my session",
            query_vector=[0.1, 0.2, 0.3],
            store=store,
            top_k=5,
        )
        boosted = self.memory_tool.memory_search(
            query="recap my session",
            query_vector=[0.1, 0.2, 0.3],
            store=store,
            top_k=5,
            prefer_episode=True,
        )

        # Baseline: the fact beats the episode (or at least appears first).
        baseline_sources = baseline.get("memory_sources", [])
        self.assertTrue(baseline_sources)
        self.assertEqual(baseline_sources[0].get("memory_id"), "fact-1")

        # Boosted: episode wins.
        boosted_sources = boosted.get("memory_sources", [])
        self.assertTrue(boosted_sources)
        self.assertEqual(boosted_sources[0].get("memory_id"), "ep-1")

        # Boosted context must carry the inline [EPISODE] label so the
        # LLM (following NARRATIVE_RECALL_SYSTEM_SUFFIX) can target it.
        self.assertIn("[EPISODE]", boosted.get("memory_context", ""))

    def test_prefer_episode_bypasses_proper_noun_gate_for_episodes(self):
        # Query is strongly entity-scoped ("Dr. Evelyn") — a generic
        # episode summary has no proper-noun overlap with that query, so
        # normally it would be dropped by the proper-noun gate. With
        # prefer_episode=True that drop is skipped for episode rows only.
        rows = [
            _episode_row(
                "ep-2",
                content="User recapped a discussion about the memory refactor.",
                distance=0.40,
            ),
            _fact_row(
                "fact-2",
                content="User likes classical music.",
                distance=0.35,
            ),
        ]
        store = _Store(rows)

        out = self.memory_tool.memory_search(
            query="Tell me about Dr. Evelyn",
            query_vector=[0.1, 0.2, 0.3],
            store=store,
            top_k=5,
            prefer_episode=True,
        )
        ids = [s.get("memory_id") for s in out.get("memory_sources", [])]
        self.assertIn(
            "ep-2", ids,
            "episode row should survive the proper-noun gate under prefer_episode",
        )
        self.assertNotIn(
            "fact-2", ids,
            "atomic fact without proper-noun overlap must still be dropped",
        )


if __name__ == "__main__":
    unittest.main()
