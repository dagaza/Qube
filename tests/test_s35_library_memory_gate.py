"""Plan S35 (D55-D59): the cross-encoder gate for library and memory recall.

Plain-chat library search, ``@library`` and memory recall keep what the S3
cross-encoder scores >= -6 (D55) in place of the embedding floors; memories
are scored in the first person (D56). When candidates were found and none
passed, the reply says the documents (or saved memories) don't cover it (D57);
a partial state is recorded only (D58). Without the reranker, or on a scoped
search, today's floors apply unchanged (fallback-first).

Fake scorers stand in for the model; the replay tests use the recorded scores
(``eval/fixtures/library_memory/rerank_scores.json``).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import library_memory_eval as lme  # noqa: E402
from core.knowledge.ranking import library_memory_gate as gate  # noqa: E402
from core.knowledge.ranking import library_memory_shadow as shadow  # noqa: E402
from core.knowledge.ranking import rerank  # noqa: E402
from mcp.memory_tool import memory_search  # noqa: E402
from mcp.rag_tool import rag_search  # noqa: E402

MODES = ("balanced", "fast", "power")
FIXTURE_DIR = ROOT / "eval" / "fixtures" / "library_memory"
BASELINE_DIR = ROOT / "eval" / "library_memory_corpus" / "baselines"
CURRENT_GATED_BASELINE = "lm_s27_gated"  # S35's gate on today's captures and recording (S27, D64; lm_s35_* is S35's record)
FALLBACK_BASELINE = "lm_s27"  # the floors: production without the reranker (S27)


def _scorer_from(table: dict[str, float], default: float = -11.0, calls: list | None = None):
    """Score a passage by the first table key it contains."""

    def score(question: str, docs: list[str]) -> list[float]:
        if calls is not None:
            calls.append((question, list(docs)))
        return [next((v for k, v in table.items() if k in d), default) for d in docs]

    return score


class _GateCase(unittest.TestCase):
    TABLE: dict[str, float] = {}

    def setUp(self):
        shadow.end_turn()
        self.calls: list = []
        rerank.set_reranker_scorer_for_tests(_scorer_from(self.TABLE, calls=self.calls))

    def tearDown(self):
        shadow.end_turn()
        rerank.set_reranker_scorer_for_tests(None)


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a):
        return self

    def where(self, *_a, **_k):
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return [dict(r) for r in self._rows]


class _Store:
    def __init__(self, vector_rows, fts_rows=()):
        self.table = SimpleNamespace(
            search=lambda q=None, query_type=None, **_k: _Query(list(fts_rows) if query_type == "fts" else vector_rows)
        )

    def reconstruct_document(self, _source):
        return ""


def _chunk(source, text, cid, distance=None):
    row = {"source": source, "text": text, "chunk_id": cid}
    if distance is not None:
        row["_distance"] = distance
    return row


def _memory(rowid, content, distance, tier="preference"):
    payload = {"content": content, "confidence": 0.8, "category": tier, "strength": 1, "decay": 1.0}
    return {"_rowid": rowid, "text": json.dumps(payload), "source": f"qube_memory::{tier}::{rowid}", "_distance": distance}


Q = np.zeros(8, dtype=np.float32)
LEASE = "lease_notes.md"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestGatePassages(unittest.TestCase):
    def tearDown(self):
        rerank.set_reranker_scorer_for_tests(None)

    def test_thresholds_are_d55(self):
        self.assertEqual((gate.LIBRARY_THRESHOLDS.keep, gate.LIBRARY_THRESHOLDS.strong), (-6.0, 2.0))
        self.assertEqual((gate.MEMORY_RECALL_THRESHOLDS.keep, gate.MEMORY_RECALL_THRESHOLDS.strong), (-6.0, 2.0))
        for path in shadow.SHADOW_PATHS:
            self.assertEqual(shadow.CALIBRATED_KEEP[path], -6.0)

    def test_without_the_reranker_the_gate_does_not_run(self):
        with patch.object(rerank, "unavailable_reason", return_value=rerank.SKIP_MODEL_MISSING):
            self.assertIsNone(gate.gate_passages("can I keep a cat", ["a"], gate.LIBRARY_THRESHOLDS))

    def test_keeps_rows_at_or_above_keep_best_first(self):
        rerank.set_reranker_scorer_for_tests(lambda _q, docs: [-7.0, 3.0, -6.0, 3.0, -2.0][: len(docs)])
        res = gate.gate_passages("can I keep a cat", ["a", "b", "c", "d", "e"], gate.LIBRARY_THRESHOLDS)
        self.assertEqual(res.kept_indices, (1, 3, 4, 2))  # ties keep input order; -6 itself is kept
        self.assertEqual(res.state, "sufficient")  # two rows >= 2
        self.assertEqual(res.diag["dropped"], 1)
        self.assertNotIn("question", res.diag)

    def test_states(self):
        for scores, state in (([-1.0, -3.0], "partial"), ([-6.5, -11.0], "insufficient"), ([2.5], "partial")):
            with self.subTest(scores=scores):
                rerank.set_reranker_scorer_for_tests(lambda _q, docs, s=scores: s[: len(docs)])
                self.assertEqual(gate.gate_passages("q text", ["x"] * len(scores), gate.LIBRARY_THRESHOLDS).state, state)
        self.assertEqual(gate.gate_passages("q text", [], gate.LIBRARY_THRESHOLDS).state, "none")

    def test_memories_are_scored_in_the_first_person(self):
        self.assertEqual(
            shadow.memory_passage("The user's sister Hana is a  paediatric nurse."), "My sister Hana is a paediatric nurse."
        )
        self.assertEqual(shadow.memory_passage("The user follows a vegetarian diet."), "I follows a vegetarian diet.")
        self.assertEqual(shadow.memory_passage("Kofi reports to the user."), "Kofi reports to me.")
        self.assertEqual(shadow.memory_passage("the username and the users"), "the username and the users")


# ---------------------------------------------------------------------------
# rag_search
# ---------------------------------------------------------------------------


class TestRagSearchGate(_GateCase):
    TABLE = {"One cat": 5.0, "cat leaves": 2.5, "deposit": -5.5, "Rent": -9.0}
    VECTOR = [
        _chunk(LEASE, "## Deposit\n\nNo pet deposit is required.", 2, 0.45),
        _chunk("budget.md", "## Fixed costs\n\nRent is due on the 1st.", 0, 0.20),  # passes the floor
        _chunk(LEASE, "## Pets\n\nOne cat is allowed.", 1, 0.75),  # below the 0.30 floor
        _chunk("qube_memory::preference::x", json.dumps({"content": "The user has One cat."}), 0, 0.1),  # not a document
    ]
    FTS = [_chunk(LEASE, "## Ending the tenancy\n\nThe cat leaves with the tenant.", 6)]

    def _run(self, **kw):
        return rag_search("can I keep a cat", Q, _Store(self.VECTOR, self.FTS), top_k=5, **kw)

    def test_the_gate_replaces_the_floor(self):
        out = self._run(relevance_gate=True)
        self.assertEqual(
            [s["chunk_id"] for s in out["sources"]],
            [f"{LEASE}::1", f"{LEASE}::6", f"{LEASE}::2"],  # by score; the sub-floor chunk is first
        )
        self.assertEqual([s["id"] for s in out["sources"]], [1, 2, 3])
        self.assertEqual(out["sufficiency"], "sufficient")
        self.assertNotIn("Rent is due", out["llm_context"])
        diag = out["relevance_gate"]
        self.assertEqual((diag["scored"], diag["survivors"], diag["kept"]), (4, 3, 3))  # the memory row is not scored
        self.assertFalse(any("qube_memory" in s["filename"] for s in out["sources"]))
        self.assertEqual(diag["kept_scores"], [5.0, 2.5, -5.5])
        # D2: the dropped chunk is listed for the inspector only.
        self.assertEqual([r["title"] for r in diag["rerank_dropped"]], ["budget.md"])

    def test_nothing_above_keep_is_insufficient(self):
        rerank.set_reranker_scorer_for_tests(lambda _q, docs: [-8.0] * len(docs))
        out = self._run(relevance_gate=True)
        self.assertEqual((out["sources"], out["llm_context"], out["sufficiency"]), ([], "", "insufficient"))

    def test_top_k_cuts_the_survivors(self):
        out = rag_search("can I keep a cat", Q, _Store(self.VECTOR, self.FTS), top_k=2, relevance_gate=True)
        self.assertEqual([s["chunk_id"] for s in out["sources"]], [f"{LEASE}::1", f"{LEASE}::6"])
        self.assertEqual(out["relevance_gate"]["kept"], 2)

    def test_fallback_first_without_the_reranker(self):
        floors = self._run()
        with patch.object(rerank, "unavailable_reason", return_value=rerank.SKIP_MODEL_MISSING):
            out = self._run(relevance_gate=True)
        self.assertEqual(out["sources"], floors["sources"])
        self.assertEqual(out["llm_context"], floors["llm_context"])
        self.assertEqual(out["sufficiency"], "unscored")
        self.assertEqual(out["relevance_gate"]["skipped_reason"], rerank.SKIP_MODEL_MISSING)
        self.assertNotIn("sufficiency", floors)  # no gate asked: the pre-S35 result shape

    def test_a_scoped_search_keeps_the_floors(self):
        floors = rag_search("can I keep a cat", Q, _Store(self.VECTOR, self.FTS), source_filter=LEASE)
        out = rag_search("can I keep a cat", Q, _Store(self.VECTOR, self.FTS), source_filter=LEASE, relevance_gate=True)
        self.assertEqual(out["sources"], floors["sources"])
        self.assertEqual((out["sufficiency"], out["relevance_gate"]["skipped_reason"]), ("unscored", "scoped"))
        self.assertEqual(self.calls, [])

    def test_a_gated_search_offers_its_own_scores(self):
        self.assertTrue(shadow.begin_turn())
        out = self._run(relevance_gate=True, shadow_path=shadow.PATH_LIBRARY_CHAT)
        (job,) = shadow.end_turn()
        self.assertEqual(len(self.calls), 1)
        rec = shadow.score_job(job)
        self.assertEqual(len(self.calls), 1)  # not scored twice
        self.assertTrue(rec["gated"])
        self.assertEqual((rec["kept"], rec["kept_scores"]), (len(out["sources"]), [5.0, 2.5, -5.5]))
        self.assertEqual((rec["would_keep"], rec["would_abstain"]), (3, False))


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------


class TestMemorySearchGate(_GateCase):
    TABLE = {"My cat": 4.0, "metric": -7.0, "sourdough": -5.0, "Dr. Amara": 1.0}
    ROWS = [
        _memory(1, "The user's cat is called Miso.", 0.60, "knowledge"),  # below the 0.35 floor
        _memory(2, "The user prefers metric units.", 0.20),
        _memory(3, "The user bakes sourdough.", 0.30),
    ]

    def _run(self, rows=None, query="what is my cat called", **kw):
        kwargs = dict(include_knowledge=True, apply_mmr=True, apply_temporal_decay=True)
        kwargs.update(kw)
        return memory_search(query, Q, _Store(rows or self.ROWS), **kwargs)

    def test_recall_keeps_what_the_gate_keeps(self):
        out = self._run(relevance_gate=True)
        self.assertEqual(
            [s["content"] for s in out["memory_sources"]], ["The user's cat is called Miso.", "The user bakes sourdough."]
        )
        self.assertEqual(out["sufficiency"], "partial")
        self.assertEqual(out["relevance_gate"]["kept_scores"], [4.0, -5.0])
        # Scored in the first person; shown as stored.
        self.assertIn("My cat is called Miso.", self.calls[0][1])
        self.assertIn("The user's cat is called Miso.", out["memory_context"])

    def test_nothing_above_keep_is_insufficient(self):
        rerank.set_reranker_scorer_for_tests(lambda _q, docs: [-9.0] * len(docs))
        out = self._run(relevance_gate=True)
        self.assertEqual((out["memory_sources"], out["sufficiency"]), ([], "insufficient"))

    def test_the_proper_noun_check_stays(self):
        rows = [_memory(4, "Dr. Amara Osei is the user's GP.", 0.2)]
        out = self._run(rows, query="Tell me about Dr. Evelyn.", relevance_gate=True)
        self.assertEqual(out["memory_sources"], [])
        self.assertEqual(out["sufficiency"], "insufficient")

    def test_core_memory_and_narrative_are_not_gated(self):
        for kw in (dict(apply_core_memory_gate=True, top_k=3), dict(prefer_episode=True, include_episode=True)):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                self.calls.clear()
                floors = self._run(**kw)
                out = self._run(relevance_gate=True, **kw)
                self.assertEqual(out["memory_sources"], floors["memory_sources"])
                self.assertEqual(out["relevance_gate"]["skipped_reason"], "not_recall")
                self.assertEqual(self.calls, [])

    def test_fallback_first_without_the_reranker(self):
        floors = self._run()
        with patch.object(rerank, "unavailable_reason", return_value=rerank.SKIP_DISABLED):
            out = self._run(relevance_gate=True)
        self.assertEqual(out["memory_sources"], floors["memory_sources"])
        self.assertEqual(out["sufficiency"], "unscored")


class TestMergedSearches(unittest.TestCase):
    def test_the_expanded_search_keeps_the_gate_state(self):
        from core.dual_query_retrieval import merge_memory_search_results, merge_rag_search_results

        src = {"id": 1, "filename": "a.md", "content": "x", "type": "rag", "chunk_id": "a.md::1"}
        empty = {"llm_context": "", "sources": [], "sufficiency": "insufficient", "relevance_gate": {"scored": 3}}
        kept = {"llm_context": "x", "sources": [dict(src)], "sufficiency": "partial", "relevance_gate": {"scored": 2}}
        self.assertEqual(merge_rag_search_results(empty, dict(empty))["sufficiency"], "insufficient")
        self.assertEqual(merge_rag_search_results(empty, kept)["sufficiency"], "partial")
        self.assertNotIn("sufficiency", merge_rag_search_results({"sources": []}, {"sources": [dict(src)]}))
        mem = {"memory_context": "", "memory_sources": [], "sufficiency": "insufficient"}
        self.assertEqual(merge_memory_search_results(mem, dict(mem))["sufficiency"], "insufficient")


# ---------------------------------------------------------------------------
# @library: the bundle and the inspector
# ---------------------------------------------------------------------------


class TestInternalCorpus(unittest.TestCase):
    def _run(self, rows, out):
        from core.knowledge.pipeline_internal_corpus import InternalCorpusEvidencePipeline
        from core.knowledge.types import RetrievalBudget, RetrievalContext

        def fake_search(*_a, gate_out=None, **_k):
            gate_out.update(out)
            return rows, [dict(r) for r in rows]

        ctx = RetrievalContext(
            query="Which refrigerant does it use?",
            semantic_query="Which refrigerant does it use?",
            budget=RetrievalBudget(max_results=3),
            library_store=object(),
            query_vector=Q,
        )
        with patch("core.knowledge.pipeline_internal_corpus.search_library_chunks", side_effect=fake_search):
            return InternalCorpusEvidencePipeline().run(ctx)

    def test_insufficient_gives_an_empty_insufficient_bundle(self):
        bundle, diag, _audit = self._run(
            [],
            {
                "sufficiency": "insufficient",
                "relevance_gate": {
                    "state": "insufficient",
                    "scored": 4,
                    "dropped": 4,
                    "threshold_keep": -6.0,
                    "top_scores": [-7.2],
                    "rerank_dropped": [{"title": "hx9_manual.md", "url": None, "adapter": "lancedb_library", "score": -7.2}],
                },
            },
        )
        self.assertEqual((bundle.sufficiency, bundle.stop_reason, bundle.sources), ("insufficient", "relevance_filtered", ()))
        self.assertEqual(diag["sufficiency"], "insufficient")
        self.assertEqual(diag["rerank_dropped"][0]["title"], "hx9_manual.md")
        self.assertNotIn("rerank_dropped", diag["rerank"])

    def test_kept_rows_record_the_state(self):
        rows = [{"title": "a.md", "snippet": "x", "source": "a.md", "chunk_id": "a.md::1"}]
        bundle, diag, _audit = self._run(rows, {"sufficiency": "partial", "relevance_gate": {"state": "partial", "dropped": 2}})
        self.assertEqual((bundle.sufficiency, len(bundle.sources), bundle.rejected_count), ("partial", 1, 2))

    def test_without_a_gate_the_bundle_is_unscored(self):
        bundle, diag, _audit = self._run([], {"sufficiency": "unscored", "relevance_gate": {"skipped_reason": "scoped"}})
        self.assertEqual((bundle.sufficiency, bundle.stop_reason), ("unscored", "no_evidence"))

    def test_the_inspector_lists_the_dropped_chunks(self):
        from core.knowledge.retrieval_trace_reader import format_relevance_gate_lines

        lines = format_relevance_gate_lines(
            {
                "sufficiency": "insufficient",
                "relevance_diag": {
                    "rerank": {"scored": 4, "survivors": 0, "threshold_keep": -6.0, "latency_ms": 90.0},
                    "rerank_dropped": [{"title": "hx9_manual.md", "adapter": "lancedb_library", "score": -7.2}],
                },
            }
        )
        self.assertTrue(any("Considered, not used (1)" in ln for ln in lines))
        self.assertTrue(any("hx9_manual.md" in ln and "-7.20" in ln for ln in lines))


# ---------------------------------------------------------------------------
# The wording (D57) and no partial note (D58)
# ---------------------------------------------------------------------------


class TestWording(unittest.TestCase):
    def _sys(self, **kw):
        from core.prompt_blocks import build_prompt_blocks, compose_system_prompt

        base = dict(execution_route="NONE", explicit_remember_active=False, has_retrieval_sources=False, current_date="")
        base.update(kw)
        blocks = build_prompt_blocks(**base)
        return blocks, compose_system_prompt(blocks)

    def test_each_insufficient_turn_gets_its_own_line(self):
        from core.memory_filters import (
            EXPLICIT_WEB_EMPTY_SUFFIX,
            LIBRARY_AND_MEMORY_INSUFFICIENT_SUFFIX,
            LIBRARY_INSUFFICIENT_SUFFIX,
            MEMORY_INSUFFICIENT_SUFFIX,
            RETRIEVAL_INSUFFICIENT_SUFFIX,
        )

        cases = [
            (dict(execution_route="WEB", retrieval_insufficient=True, knowledge_service="internal_corpus"), LIBRARY_INSUFFICIENT_SUFFIX),
            (dict(library_insufficient=True), LIBRARY_INSUFFICIENT_SUFFIX),
            (dict(memory_insufficient=True), MEMORY_INSUFFICIENT_SUFFIX),
            (dict(library_insufficient=True, memory_insufficient=True), LIBRARY_AND_MEMORY_INSUFFICIENT_SUFFIX),
        ]
        every = (
            LIBRARY_INSUFFICIENT_SUFFIX,
            MEMORY_INSUFFICIENT_SUFFIX,
            LIBRARY_AND_MEMORY_INSUFFICIENT_SUFFIX,
            RETRIEVAL_INSUFFICIENT_SUFFIX,
            EXPLICIT_WEB_EMPTY_SUFFIX,
        )
        for kw, expected in cases:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                blocks, sys_p = self._sys(**kw)
                self.assertTrue(blocks.no_sources_mode)
                for suffix in every:
                    if suffix is expected:
                        self.assertIn(suffix.strip(), sys_p)
                    else:
                        self.assertNotIn(suffix.strip()[:80], sys_p)

    def test_the_documents_wording_is_d57(self):
        from core.memory_filters import LIBRARY_INSUFFICIENT_SUFFIX, MEMORY_INSUFFICIENT_SUFFIX

        self.assertIn('"Your documents don\'t cover this."', LIBRARY_INSUFFICIENT_SUFFIX)
        self.assertIn("labelled clearly as general knowledge", LIBRARY_INSUFFICIENT_SUFFIX)
        self.assertIn("searched the web", LIBRARY_INSUFFICIENT_SUFFIX)  # ... must not be claimed
        self.assertIn('"I don\'t have that saved."', MEMORY_INSUFFICIENT_SUFFIX)
        self.assertIn("Do NOT guess or invent personal details", MEMORY_INSUFFICIENT_SUFFIX)

    def test_the_web_wording_is_unchanged_elsewhere(self):
        from core.memory_filters import RETRIEVAL_INSUFFICIENT_SUFFIX

        _, sys_p = self._sys(execution_route="WEB", retrieval_insufficient=True, knowledge_service="general_web")
        self.assertIn(RETRIEVAL_INSUFFICIENT_SUFFIX.strip(), sys_p)

    def test_worker_wiring(self):
        src = (ROOT / "workers" / "llm_worker.py").read_text(encoding="utf-8")
        self.assertIn("relevance_gate=not prefer_episode,", src)
        self.assertIn("relevance_gate=True,  # S35 (D55); a file-scoped search keeps the floors", src)
        core_call = src[src.index("top_k=3,\n                    apply_core_memory_gate=True") :][:600]
        self.assertNotIn("relevance_gate", core_call)
        self.assertIn("library_insufficient=library_insufficient,", src)
        self.assertIn("memory_insufficient=memory_insufficient,", src)
        # D58: @library records a partial state without the partial note.
        partial = src[src.index("retrieval_partial = bool(") :][:500]
        self.assertIn("!= SERVICE_INTERNAL_CORPUS", partial)
        library = (ROOT / "core" / "knowledge" / "adapters" / "lancedb_library.py").read_text(encoding="utf-8")
        self.assertIn("relevance_gate=True,", library)


# ---------------------------------------------------------------------------
# Replays: the gate on S33's corpus, all three embedders
# ---------------------------------------------------------------------------


def _baseline(stage: str, mode: str) -> dict:
    return lme.load_json(BASELINE_DIR / f"{stage}_{mode}.json")


class TestGatedReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = lme.load_corpus()
        cls.captures = {m: lme.load_json(FIXTURE_DIR / f"captured_{m}.json") for m in MODES}
        cls.recorded = lme.load_json(lme.DEFAULT_RERANK_SCORES)

    def test_gated_replays_match_the_baseline(self):
        for mode, cap in self.captures.items():
            with self.subTest(mode=mode):
                report = lme.build_report(cap, self.corpus, stage="S35", rerank_scores=self.recorded)
                self.assertEqual(json.loads(json.dumps(report)), _baseline(CURRENT_GATED_BASELINE, mode))

    def test_without_the_reranker_the_gated_runs_replay_the_floors(self):
        # Fallback-first: the worker's gate flag with no reranker gives today's floors.
        with patch.object(rerank, "unavailable_reason", return_value=rerank.SKIP_MODEL_MISSING):
            for mode, cap in self.captures.items():
                base = _baseline(FALLBACK_BASELINE, mode)
                for case in self.corpus["cases"]:
                    for run in lme.case_runs(case):
                        with self.subTest(mode=mode, case=case["id"], run=run):
                            kept, state = lme.replay_run_state(cap, case, run, relevance_gate=True)
                            expected = next(r for r in base["cases"] if r["id"] == case["id"] and r["run"] == run)
                            self.assertEqual(kept, expected["kept"])
                            if run in lme.GATED_RUNS:
                                self.assertEqual(state, "unscored")

    def test_an_unrecorded_passage_refuses_the_replay(self):
        cap = self.captures["balanced"]
        recorded = json.loads(json.dumps(self.recorded))
        case = next(c for c in self.corpus["cases"] if c["id"] == "lib_omega_only_sky_blue")
        question = cap["cases"][case["id"]]["runs"][lme.RUN_LIBRARY_CHAT]["search_text"]
        recorded["scores"][question].pop(lme.rerank_pool(cap, case, lme.RUN_LIBRARY_CHAT)[0])
        with self.assertRaises(lme.ReplayMismatch):
            lme.evaluate_capture(cap, self.corpus, rerank_scores=recorded)

    def test_rule_7_against_the_floors(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                regs = lme.compare_reports(_baseline(CURRENT_GATED_BASELINE, mode), _baseline(FALLBACK_BASELINE, mode))
                self.assertEqual(regs, [])
