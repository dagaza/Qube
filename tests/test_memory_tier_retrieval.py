"""Unit tests for the T3.4 tier-aware WHERE builder in ``mcp.memory_tool``.

Stubs out the LanceDB ``store.table.search(...)`` chain so the test can
capture the ``where_clause`` string ``memory_search`` actually pushes to
the backend for every combination of tier flags.
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp import memory_tool


# ------------------------------------------------------------
# Fake LanceDB chain: .search(vec).where(clause).limit(n).to_list()
# Captures the last where clause on the parent ``fake_table``.
# ------------------------------------------------------------
class _FakeQuery:
    def __init__(self, parent: "_FakeTable") -> None:
        self._parent = parent

    def where(self, clause: str) -> "_FakeQuery":
        self._parent.last_where = clause
        self._parent.where_calls.append(clause)
        return self

    def limit(self, _n: int) -> "_FakeQuery":
        return self

    def to_list(self) -> list[dict[str, Any]]:
        # An empty result list short-circuits memory_search past the
        # ranking loop — we only care that the right WHERE was built.
        return []


class _FakeTable:
    def __init__(self) -> None:
        self.last_where: str | None = None
        self.where_calls: list[str] = []

    def search(self, _vec):
        return _FakeQuery(self)


class _FakeStore:
    def __init__(self) -> None:
        self.table = _FakeTable()


class WhereClauseBuilderTests(unittest.TestCase):
    def _run(
        self,
        *,
        include_preference: bool = True,
        include_knowledge: bool = False,
        include_episode: bool = False,
        include_context: bool = True,
        prefer_episode: bool = False,
    ) -> str:
        store = _FakeStore()
        memory_tool.memory_search(
            "hello",
            np.zeros(4, dtype=np.float32),
            store,
            top_k=3,
            include_preference=include_preference,
            include_knowledge=include_knowledge,
            include_episode=include_episode,
            include_context=include_context,
            prefer_episode=prefer_episode,
        )
        self.assertIsNotNone(store.table.last_where)
        return store.table.last_where  # type: ignore[return-value]

    # -------------------- single-tier cases --------------------

    def test_preferences_only(self):
        clause = self._run(
            include_preference=True,
            include_knowledge=False,
            include_episode=False,
            include_context=False,
        )
        self.assertIn("qube_memory::preference::%", clause)
        self.assertNotIn("knowledge", clause)
        self.assertNotIn("episode", clause)

    def test_knowledge_only(self):
        clause = self._run(
            include_preference=False,
            include_knowledge=True,
            include_episode=False,
            include_context=False,
        )
        self.assertIn("qube_memory::knowledge::%", clause)
        self.assertNotIn("preference", clause)

    def test_episode_only(self):
        clause = self._run(
            include_preference=False,
            include_knowledge=False,
            include_episode=True,
            include_context=False,
        )
        self.assertIn("qube_memory::episode::%", clause)

    def test_context_includes_legacy_prefix(self):
        """Context pulls legacy ``qube_memory::<category>`` rows in too."""
        clause = self._run(
            include_preference=False,
            include_knowledge=False,
            include_episode=False,
            include_context=True,
        )
        self.assertIn("qube_memory::context::%", clause)
        # Legacy wildcard must also be present so pre-T3.4 installs
        # keep retrieving their existing rows.
        self.assertIn("qube_memory::%", clause)

    # -------------------- combinations --------------------

    def test_default_chat_turn_is_pref_plus_context(self):
        """Default every turn: preferences + context (no knowledge/episode)."""
        clause = self._run()  # defaults
        self.assertIn("preference::%", clause)
        self.assertIn("context::%", clause)
        self.assertNotIn("knowledge::%", clause)
        self.assertNotIn("episode::%", clause)

    def test_recall_memory_turn_adds_knowledge(self):
        clause = self._run(
            include_preference=True,
            include_knowledge=True,
            include_episode=False,
            include_context=True,
        )
        self.assertIn("preference::%", clause)
        self.assertIn("knowledge::%", clause)
        self.assertIn("context::%", clause)
        self.assertNotIn("episode::%", clause)

    def test_narrative_turn_adds_episode(self):
        clause = self._run(
            include_preference=True,
            include_knowledge=True,
            include_episode=True,
            include_context=True,
            prefer_episode=True,
        )
        self.assertIn("preference::%", clause)
        self.assertIn("knowledge::%", clause)
        self.assertIn("episode::%", clause)
        self.assertIn("context::%", clause)

    def test_prefer_episode_forces_episode_even_if_flag_false(self):
        """prefer_episode=True should ensure episode rows are reachable
        even when include_episode is left at its default (False)."""
        clause = self._run(
            include_preference=False,
            include_knowledge=False,
            include_episode=False,
            include_context=False,
            prefer_episode=True,
        )
        self.assertIn("episode::%", clause)

    # -------------------- degenerate input --------------------

    def test_all_flags_off_falls_back_to_catchall(self):
        """With every flag off, fall back to the catch-all to avoid
        silently returning zero rows."""
        clause = self._run(
            include_preference=False,
            include_knowledge=False,
            include_episode=False,
            include_context=False,
        )
        self.assertEqual(clause, "source LIKE 'qube_memory::%'")


if __name__ == "__main__":
    unittest.main()
