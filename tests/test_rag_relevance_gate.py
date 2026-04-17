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

    # Distinctive log line emitted by the post-retrieval downgrade
    # block. Shared by every test in this class as a structural
    # anchor so we don't accidentally find the earlier proactive
    # WEB-veto assignment (which uses its own log line).
    _POST_RETRIEVAL_ANCHOR = "All retrieval channels empty after relevance"

    def test_downgrade_block_exists(self) -> None:
        # The downgrade flips execution_route to "NONE" when
        # all_ui_sources is empty on a retrieval route. Assert the
        # tuple (now widened to include WEB / INTERNET so the
        # ``[W]`` hallucination regression can't come back), the
        # guard, and the assignment anchored to the downgrade log.
        self.assertRegex(
            self.src,
            r'execution_route\s+in\s+\(\s*"MEMORY"\s*,\s*"RAG"\s*,'
            r'\s*"HYBRID"\s*,\s*"WEB"\s*,\s*"INTERNET"\s*\)',
            "Expected a tuple-membership check including WEB / "
            "INTERNET in the downgrade block — WEB turns with no "
            "sources must also downgrade to NONE so the 'You have "
            "live web results' system prompt cannot fire without "
            "any actual sources.",
        )
        self.assertRegex(
            self.src,
            r'not\s+all_ui_sources',
            "Expected the downgrade to be guarded by ``not all_ui_sources``.",
        )
        # Anchor the NONE assignment to the distinctive post-retrieval
        # downgrade log so this test isn't satisfied by the earlier
        # proactive WEB-veto ``execution_route = "NONE"`` assignment.
        # ``[\s\S]*?`` is used instead of ``.*?`` because the default
        # regex ``.`` does not match newlines and ``re.search`` (which
        # ``assertRegex`` calls under the hood) does not expose the
        # ``re.DOTALL`` flag.
        self.assertRegex(
            self.src,
            re.escape(self._POST_RETRIEVAL_ANCHOR)
            + r'[\s\S]*?execution_route\s*=\s*"NONE"',
            "Expected the ``execution_route = 'NONE'`` assignment "
            "immediately after the post-retrieval downgrade log line.",
        )

    def test_downgrade_runs_after_telemetry(self) -> None:
        """Telemetry must log the actually-executed route. The
        post-retrieval downgrade must therefore sit AFTER the
        telemetry block."""
        telemetry_idx = self.src.find("self.telemetry.log(")
        self.assertGreater(telemetry_idx, 0, "telemetry log call not found")
        # Anchor to the distinctive post-retrieval log line — a plain
        # ``execution_route = "NONE"`` match would now find the earlier
        # proactive WEB-veto assignment instead.
        post_retrieval_idx = self.src.find(self._POST_RETRIEVAL_ANCHOR)
        self.assertGreater(
            post_retrieval_idx, 0,
            f"post-retrieval downgrade log line "
            f"{self._POST_RETRIEVAL_ANCHOR!r} not found.",
        )
        self.assertGreater(
            post_retrieval_idx, telemetry_idx,
            "T4.1 post-retrieval downgrade must run after telemetry "
            "so the original executed route is still recorded for "
            "router tuning.",
        )

    def test_downgrade_runs_before_system_prompt_build(self) -> None:
        """The post-retrieval downgrade must run before the
        system-prompt branch (``elif execution_route in ["RAG",
        "HYBRID", "MEMORY"]``) so that an empty-retrieval turn
        lands on the base prompt."""
        post_retrieval_idx = self.src.find(self._POST_RETRIEVAL_ANCHOR)
        self.assertGreater(post_retrieval_idx, 0)
        prompt_branch_idx = self.src.find(
            'elif execution_route in ["RAG", "HYBRID", "MEMORY"]'
        )
        self.assertGreater(prompt_branch_idx, 0)
        self.assertLess(
            post_retrieval_idx, prompt_branch_idx,
            "T4.1 downgrade must run before the system-prompt build "
            "so the prompt branch sees the updated route.",
        )

    def test_web_downgrade_marks_skip_enrichment(self) -> None:
        """When the empty-retrieval downgrade fires on a WEB /
        INTERNET turn the worker must also call
        ``_mark_skip_enrichment("web_route_no_sources")`` so the
        thin "I can't check live data" reply isn't mined for user
        facts by the enrichment worker — mirroring the existing
        ``web_tool_failure`` behaviour on the sentinel path."""
        # The ``skip_enrichment`` call must sit inside the
        # post-retrieval downgrade block (anchored to the downgrade
        # log) and specifically be guarded by a ``WEB`` / ``INTERNET``
        # check so the MEMORY / RAG / HYBRID paths don't get the
        # same enrichment skip (those turns are legitimate and
        # should still enrich).
        self.assertRegex(
            self.src,
            re.escape(self._POST_RETRIEVAL_ANCHOR)
            + r'[\s\S]*?execution_route\s+in\s+\(\s*"WEB"\s*,\s*"INTERNET"\s*\)'
            + r'[\s\S]*?_mark_skip_enrichment\(\s*"web_route_no_sources"\s*\)',
            "Expected the empty-WEB downgrade path to mark "
            "``skip_enrichment(\"web_route_no_sources\")`` so the "
            "thin reply isn't enriched into user-fact memories.",
        )


class LLMWorkerWebVetoContractTests(unittest.TestCase):
    """Static contract check for the proactive WEB-route veto added
    alongside the T4.1 downgrade widening.

    The cognitive router internally promotes ``route = "web"`` as
    soon as ``_score_web_intent`` clears its threshold (keywords
    like "weather" / "today"). That value flows through
    ``execution_route = decision["route"].upper()`` BEFORE the
    manual/auto/force triggers are evaluated. Without a proactive
    veto, a query like "what's the weather in Copenhagen today?"
    arrives at the system-prompt branch already pinned to WEB even
    when the user has disabled the internet tool — which then fires
    the "You have live web results" prompt against an empty source
    set and causes the small LLM to hallucinate a ``[W]`` citation.

    These tests guard against the veto block being removed or
    weakened.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "workers", "llm_worker.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    # The veto's INFO log string is split across multiple adjacent
    # string literals in ``llm_worker.py`` (for readability); pick
    # the first literal as a stable anchor for source searches.
    _VETO_LOG_ANCHOR = "Cognitive router picked WEB but internet"

    def test_proactive_web_veto_log_line_exists(self) -> None:
        """The veto block emits a distinctive INFO log line so the
        fix is greppable in ``logs/llm_debug.log``."""
        self.assertIn(
            self._VETO_LOG_ANCHOR,
            self.src,
            "Expected the proactive WEB-veto INFO log line — silent "
            "config changes make future regressions invisible.",
        )

    def test_proactive_web_veto_checks_tool_disabled(self) -> None:
        """The veto predicate must gate on
        ``not self.mcp_internet_enabled`` — i.e. the user has
        explicitly turned off the internet tool."""
        # Anchor to the veto log line so this test isn't satisfied
        # by an unrelated ``mcp_internet_enabled`` check elsewhere.
        anchor_idx = self.src.find(self._VETO_LOG_ANCHOR)
        self.assertGreater(anchor_idx, 0)
        # Take a window AROUND the anchor (the predicate is a few
        # lines above the log line). 400 chars is plenty for the
        # if-block + preceding condition lines.
        window_start = max(0, anchor_idx - 600)
        window = self.src[window_start:anchor_idx]
        self.assertIn(
            "not self.mcp_internet_enabled",
            window,
            "Expected the proactive veto to be guarded by "
            "``not self.mcp_internet_enabled`` so it only fires "
            "when the user has disabled the internet tool.",
        )

    def test_proactive_web_veto_respects_force_flag(self) -> None:
        """The veto must NOT fire when ``force_web`` is set (the
        user clicked the Web button on the chat UI), nor when a
        manual web trigger / auto trigger fired — those paths have
        already legitimately earned the WEB route."""
        anchor_idx = self.src.find(self._VETO_LOG_ANCHOR)
        self.assertGreater(anchor_idx, 0)
        window_start = max(0, anchor_idx - 600)
        window = self.src[window_start:anchor_idx]
        for flag in ("not force_web", "not manual_web", "not auto_web"):
            self.assertIn(
                flag,
                window,
                f"Expected the proactive veto predicate to include "
                f"``{flag}`` so it only fires when no legitimate "
                f"web-intent path claimed this turn.",
            )

    def test_proactive_web_veto_runs_before_post_retrieval_downgrade(
        self,
    ) -> None:
        """The proactive veto must run BEFORE the §2.75 post-retrieval
        downgrade so it can short-circuit the WEB tool-execution
        block entirely — we don't want to waste a web search call
        on a route the user has disabled."""
        veto_idx = self.src.find(self._VETO_LOG_ANCHOR)
        downgrade_idx = self.src.find(
            "All retrieval channels empty after relevance"
        )
        self.assertGreater(veto_idx, 0)
        self.assertGreater(downgrade_idx, 0)
        self.assertLess(
            veto_idx, downgrade_idx,
            "Proactive WEB veto must run before the post-retrieval "
            "downgrade so the WEB tool-execution block is skipped, "
            "not retried.",
        )


if __name__ == "__main__":
    unittest.main()
