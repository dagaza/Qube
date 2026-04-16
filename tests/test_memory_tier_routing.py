"""Contract tests for the LLMWorker -> memory_tool tier flag mapping (T3.4).

The LLMWorker's ``_execute_llm_turn`` chooses a set of tier flags for
``memory_search`` per execution route (CHAT / MEMORY / HYBRID /
narrative). Directly instantiating LLMWorker is heavy (pulls the whole
audio / qt / engine stack), so instead we encode the intended route ->
flags mapping as a fixture and assert it (a) produces the right
``WHERE`` clause in ``memory_tool`` and (b) matches the flag set the
llm_worker source file statically passes on each branch.

This guards against silent regressions when someone edits llm_worker
and forgets that MEMORY turns must still surface knowledge tier rows.
"""
from __future__ import annotations

import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp import memory_tool


# ------------------------------------------------------------
# Fake LanceDB chain — same shape as test_memory_tier_retrieval.
# ------------------------------------------------------------
class _FakeQuery:
    def __init__(self, parent):
        self._parent = parent

    def where(self, clause):
        self._parent.last_where = clause
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return []


class _FakeTable:
    def __init__(self):
        self.last_where = None

    def search(self, _vec):
        return _FakeQuery(self)


class _FakeStore:
    def __init__(self):
        self.table = _FakeTable()


# ------------------------------------------------------------
# Canonical route -> flags contract (§3.3 of the plan).
# ------------------------------------------------------------
ROUTE_FLAGS: dict[str, dict[str, bool]] = {
    "CHAT": dict(
        include_preference=True,
        include_knowledge=False,
        include_episode=False,
        include_context=True,
        prefer_episode=False,
    ),
    "MEMORY": dict(
        include_preference=True,
        include_knowledge=True,
        include_episode=False,
        include_context=True,
        prefer_episode=False,
    ),
    "HYBRID": dict(
        include_preference=True,
        include_knowledge=True,
        include_episode=False,
        include_context=True,
        prefer_episode=False,
    ),
    "NARRATIVE": dict(
        include_preference=True,
        include_knowledge=True,
        include_episode=True,
        include_context=True,
        prefer_episode=True,
    ),
}


class TierRoutingContractTests(unittest.TestCase):
    def _run_memory_search(self, flags: dict) -> str:
        store = _FakeStore()
        memory_tool.memory_search(
            "hello",
            np.zeros(4, dtype=np.float32),
            store,
            top_k=3,
            **flags,
        )
        self.assertIsNotNone(store.table.last_where)
        return store.table.last_where

    def test_chat_turn_queries_preferences_and_context_only(self):
        clause = self._run_memory_search(ROUTE_FLAGS["CHAT"])
        self.assertIn("preference::%", clause)
        self.assertIn("context::%", clause)
        self.assertNotIn("knowledge::%", clause)
        self.assertNotIn("episode::%", clause)

    def test_memory_turn_includes_knowledge(self):
        clause = self._run_memory_search(ROUTE_FLAGS["MEMORY"])
        self.assertIn("preference::%", clause)
        self.assertIn("knowledge::%", clause)
        self.assertIn("context::%", clause)
        self.assertNotIn("episode::%", clause)

    def test_hybrid_turn_includes_knowledge(self):
        clause = self._run_memory_search(ROUTE_FLAGS["HYBRID"])
        self.assertIn("preference::%", clause)
        self.assertIn("knowledge::%", clause)
        self.assertIn("context::%", clause)
        self.assertNotIn("episode::%", clause)

    def test_narrative_turn_includes_all_tiers(self):
        clause = self._run_memory_search(ROUTE_FLAGS["NARRATIVE"])
        self.assertIn("preference::%", clause)
        self.assertIn("knowledge::%", clause)
        self.assertIn("episode::%", clause)
        self.assertIn("context::%", clause)


class LLMWorkerSourceContractTests(unittest.TestCase):
    """Static check: the llm_worker source file still passes tier flags
    on every branch that invokes ``memory_search``.

    Parses the source text instead of importing the module (which would
    pull the whole Qt / audio / engine stack). The goal is to fail CI if
    someone rewrites the route block and accidentally drops the tier
    keywords — not to reproduce the full routing logic here.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "workers", "llm_worker.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_every_memory_search_call_passes_include_preference(self):
        """Every ``memory_search(...)`` call must pass ``include_preference``."""
        # Extract each memory_search(...) block.
        blocks = re.findall(
            r"memory_search\(\s*(?P<body>.*?)\n\s*\)",
            self.src,
            flags=re.DOTALL,
        )
        self.assertGreaterEqual(
            len(blocks), 1,
            "llm_worker.py should call memory_search at least once",
        )
        for body in blocks:
            self.assertIn(
                "include_preference",
                body,
                msg=f"memory_search call missing include_preference:\n{body}",
            )
            self.assertIn(
                "include_context",
                body,
                msg=f"memory_search call missing include_context:\n{body}",
            )

    def test_chat_path_passes_include_knowledge_false(self):
        """The plain-CHAT (route=NONE) branch must pass
        ``include_knowledge=False`` — knowledge rows should not leak
        into ordinary chat turns."""
        # Slice the source around ``execution_route == "NONE"`` block.
        match = re.search(
            r"execution_route == \"NONE\".*?memory_search\(\s*(?P<body>.*?)\n\s*\)",
            self.src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "No memory_search call found in the execution_route=='NONE' branch; "
            "plain CHAT turns must still run a preferences-only retrieval.",
        )
        body = match.group("body")
        self.assertIn("include_knowledge=False", body)
        self.assertIn("include_episode=False", body)
        self.assertIn("include_preference=True", body)

    def test_memory_or_hybrid_path_passes_include_knowledge_true(self):
        """The MEMORY/HYBRID branch must pass ``include_knowledge=True``."""
        match = re.search(
            r"execution_route in \[\"MEMORY\", \"HYBRID\"\].*?memory_search\(\s*(?P<body>.*?)\n\s*\)",
            self.src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "Could not find the MEMORY/HYBRID memory_search call.",
        )
        body = match.group("body")
        self.assertIn("include_knowledge=True", body)


if __name__ == "__main__":
    unittest.main()
