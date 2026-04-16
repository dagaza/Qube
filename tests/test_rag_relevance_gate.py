"""Unit tests for the T4.1 RAG hard semantic-relevance gate
(`mcp.rag_tool.MIN_RAG_SEMANTIC_SCORE`) and the corresponding
llm_worker post-retrieval route downgrade.

These tests deliberately avoid importing heavy modules (qt / audio /
native engine) — the RAG tool is pure and can be exercised directly
with a fake LanceDB store, and the llm_worker downgrade is verified
via a static source-text check in the same style as
``test_memory_tier_routing.LLMWorkerSourceContractTests``.
"""
from __future__ import annotations

import os
import re
import sys
import types
import unittest
from typing import Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ------------------------------------------------------------
# Minimal lancedb / pyarrow stubs so ``rag.store`` (imported
# transitively by ``mcp.rag_tool``) can load in a headless test
# environment without the real native libs. ``rag_tool`` does not
# touch any lancedb symbol — it only operates on the fake table we
# inject — so the stubs can be essentially empty.
# ------------------------------------------------------------
if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = types.ModuleType("lancedb")
if "pyarrow" not in sys.modules:
    pa = types.ModuleType("pyarrow")

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    pa.schema = _noop
    pa.field = _noop
    pa.list_ = _noop
    pa.float32 = _noop
    pa.utf8 = _noop
    pa.int32 = _noop
    sys.modules["pyarrow"] = pa


from mcp import rag_tool


# ============================================================
# Fake LanceDB table:
#   .search(x).limit(n).to_list()  ← vector  (x = np.ndarray)
#   .search(x, query_type="fts").limit(n).to_list()  ← FTS (x = str)
# The fake returns different canned lists depending on whether it was
# called with a numpy vector (vector channel) or a string (FTS channel).
# ============================================================
class _FakeQuery:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def limit(self, _n: int) -> "_FakeQuery":
        return self

    def to_list(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeTable:
    def __init__(
        self,
        vector_rows: list[dict[str, Any]],
        fts_rows: list[dict[str, Any]] | None = None,
        fts_raises: bool = False,
    ) -> None:
        self._vector_rows = vector_rows
        self._fts_rows = fts_rows or []
        self._fts_raises = fts_raises

    def search(self, query, query_type: str | None = None) -> _FakeQuery:
        if query_type == "fts":
            if self._fts_raises:
                raise RuntimeError("FTS unavailable")
            return _FakeQuery(self._fts_rows)
        return _FakeQuery(self._vector_rows)


class _FakeStore:
    def __init__(
        self,
        vector_rows: list[dict[str, Any]],
        fts_rows: list[dict[str, Any]] | None = None,
        fts_raises: bool = False,
    ) -> None:
        self.table = _FakeTable(vector_rows, fts_rows, fts_raises)


# ============================================================
# Helpers
# ============================================================
def _chunk(source: str, text: str, distance: float) -> dict[str, Any]:
    """Build a fake RAG row roughly shaped like a LanceDB result."""
    return {
        "source": source,
        "text": text,
        "_distance": distance,
        "chunk_id": 0,
    }


# ============================================================
# Tests
# ============================================================
class RagRelevanceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "Why is the sky blue?"
        self.qvec = np.zeros(4, dtype=np.float32)

    def test_floor_constant_matches_design(self) -> None:
        """Regression guard: the floor must stay in the conservative
        0.20 – 0.50 band (mirrors memory_tool's MIN_SEMANTIC_SCORE
        design window). Loosening past 0.50 would start dropping
        legitimately related chunks; tightening below 0.20 would let
        the sky-blue-vs-Project-Omega regression back in."""
        self.assertGreaterEqual(rag_tool.MIN_RAG_SEMANTIC_SCORE, 0.20)
        self.assertLessEqual(rag_tool.MIN_RAG_SEMANTIC_SCORE, 0.50)

    def test_below_floor_chunk_is_dropped(self) -> None:
        """Irrelevant chunk (semantic << floor) must never reach
        llm_context / sources."""
        # distance 1.2 → semantic_score = max(0, 1 - 1.2) = 0.0
        store = _FakeStore(
            vector_rows=[_chunk("Project Omega", "Blue Jay migration study", 1.2)]
        )
        result = rag_tool.rag_search(self.query, self.qvec, store)
        self.assertEqual(result["llm_context"], "")
        self.assertEqual(result["sources"], [])

    def test_above_floor_chunk_is_kept(self) -> None:
        """Relevant chunk (semantic >> floor) passes through
        unchanged."""
        # distance 0.2 → semantic_score = 0.8
        store = _FakeStore(
            vector_rows=[_chunk("atmosphere.pdf", "Rayleigh scattering explains...", 0.2)]
        )
        result = rag_tool.rag_search(self.query, self.qvec, store)
        self.assertIn("Rayleigh scattering", result["llm_context"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "atmosphere.pdf")

    def test_mixed_set_filters_only_below_floor(self) -> None:
        """Above-floor chunks survive; below-floor chunks are
        dropped. Order is preserved among survivors."""
        store = _FakeStore(
            vector_rows=[
                _chunk("A.pdf", "topically relevant A", 0.2),  # 0.8 keep
                _chunk("B.pdf", "topically unrelated B", 1.1),  # 0.0 drop
                _chunk("C.pdf", "topically relevant C", 0.4),  # 0.6 keep
            ]
        )
        result = rag_tool.rag_search(self.query, self.qvec, store, top_k=5)
        names = [s["filename"] for s in result["sources"]]
        self.assertIn("A.pdf", names)
        self.assertIn("C.pdf", names)
        self.assertNotIn("B.pdf", names)

    def test_all_vectors_below_floor_also_suppresses_fts(self) -> None:
        """When every vector candidate fails the gate, FTS-only hits
        are dropped too — lexical matches without semantic
        corroboration are brittle (FTS matching 'blue' in a Blue Jay
        study when the user asked about sky color)."""
        store = _FakeStore(
            vector_rows=[
                _chunk("Project Omega", "Blue Jay migration", 1.2),  # drop
                _chunk("Project Omega", "Dr Vance chief researcher", 1.3),  # drop
            ],
            fts_rows=[
                # FTS has no _distance — would survive an overly-lenient
                # filter and slip into the prompt. Must be dropped when
                # the vector gate killed everything.
                {"source": "Project Omega", "text": "blue jay", "chunk_id": 0},
            ],
        )
        result = rag_tool.rag_search(self.query, self.qvec, store)
        self.assertEqual(result["llm_context"], "")
        self.assertEqual(result["sources"], [])

    def test_missing_distance_is_treated_as_unassessable_and_kept(self) -> None:
        """Defensive: if a vector backend returns rows without
        ``_distance`` (unusual but possible), we keep them rather
        than silently dropping all retrieval."""
        store = _FakeStore(
            vector_rows=[
                {"source": "X.md", "text": "no-distance row", "chunk_id": 0},
            ]
        )
        result = rag_tool.rag_search(self.query, self.qvec, store)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "X.md")

    def test_empty_vector_channel_keeps_fts_fallback(self) -> None:
        """If the vector search was simply unavailable (not "all
        dropped by the gate" but "nothing came back at all"), we
        keep FTS as a legitimate fallback — the gate must not make
        RAG strictly worse than before for lexical-only deployments."""
        store = _FakeStore(
            vector_rows=[],  # vector channel empty
            fts_rows=[
                {"source": "FTS.md", "text": "lexical hit", "chunk_id": 0},
            ],
        )
        result = rag_tool.rag_search(self.query, self.qvec, store)
        names = [s["filename"] for s in result["sources"]]
        self.assertIn("FTS.md", names)

    def test_floor_boundary_score_is_kept(self) -> None:
        """A chunk exactly at the floor (semantic == MIN) passes.
        Only strictly-below-floor chunks are dropped."""
        # distance such that semantic_score == MIN_RAG_SEMANTIC_SCORE
        boundary_distance = 1.0 - rag_tool.MIN_RAG_SEMANTIC_SCORE
        store = _FakeStore(
            vector_rows=[_chunk("edge.md", "right at floor", boundary_distance)]
        )
        result = rag_tool.rag_search(self.query, self.qvec, store)
        self.assertEqual(len(result["sources"]), 1)


class LLMWorkerDowngradeContractTests(unittest.TestCase):
    """Static contract check: the llm_worker source must contain the
    T4.1 post-retrieval route downgrade block, and it must sit
    AFTER the telemetry log (so telemetry records the original
    executed route) and BEFORE the system-prompt build.

    Parses the source text instead of importing the module (which
    would pull Qt / audio / native engine deps). The goal is to
    fail CI if someone rewrites the retrieval block and loses the
    downgrade — not to reproduce the full routing logic here.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "workers", "llm_worker.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_downgrade_block_exists(self) -> None:
        # The downgrade flips execution_route to "NONE" when
        # all_ui_sources is empty on a retrieval route. Assert both
        # the predicate and the assignment exist.
        self.assertRegex(
            self.src,
            r'execution_route\s+in\s+\(\s*"MEMORY"\s*,\s*"RAG"\s*,\s*"HYBRID"\s*\)',
            "Expected a tuple-membership check for retrieval routes "
            "in the downgrade block.",
        )
        self.assertRegex(
            self.src,
            r'not\s+all_ui_sources',
            "Expected the downgrade to be guarded by ``not all_ui_sources``.",
        )
        self.assertRegex(
            self.src,
            r'execution_route\s*=\s*"NONE"',
            "Expected the downgrade to reassign execution_route to NONE.",
        )

    def test_downgrade_runs_after_telemetry(self) -> None:
        """Telemetry must log the actually-executed route. The
        downgrade must therefore sit AFTER the telemetry block."""
        telemetry_idx = self.src.find("self.telemetry.log(")
        self.assertGreater(telemetry_idx, 0, "telemetry log call not found")
        # Locate our downgrade assignment.
        downgrade_match = re.search(
            r'execution_route\s*=\s*"NONE"',
            self.src,
        )
        self.assertIsNotNone(downgrade_match, "downgrade assignment not found")
        self.assertGreater(
            downgrade_match.start(), telemetry_idx,
            "T4.1 downgrade must run after telemetry so the original "
            "route is still recorded for router tuning.",
        )

    def test_downgrade_runs_before_system_prompt_build(self) -> None:
        """The downgrade must run before the system-prompt branch
        (``elif execution_route in ["RAG", "HYBRID", "MEMORY"]``)
        so that an empty-retrieval turn lands on the base prompt."""
        downgrade_match = re.search(
            r'execution_route\s*=\s*"NONE"',
            self.src,
        )
        self.assertIsNotNone(downgrade_match)
        prompt_branch_idx = self.src.find(
            'elif execution_route in ["RAG", "HYBRID", "MEMORY"]'
        )
        self.assertGreater(prompt_branch_idx, 0)
        self.assertLess(
            downgrade_match.start(), prompt_branch_idx,
            "T4.1 downgrade must run before the system-prompt build "
            "so the prompt branch sees the updated route.",
        )


if __name__ == "__main__":
    unittest.main()
