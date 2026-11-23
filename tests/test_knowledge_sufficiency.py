"""Scientific relevance gate and sufficiency (plan S4)."""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.ranking.rerank import (  # noqa: E402
    SKIP_DISABLED,
    set_reranker_scorer_for_tests,
)
from core.knowledge.ranking.sufficiency import (  # noqa: E402
    SCIENTIFIC_THRESHOLDS,
    Sufficiency,
    SufficiencyThresholds,
    assess_sufficiency,
    relevance_probability,
)

_T = SufficiencyThresholds(keep=0.0, strong=2.0, min_strong_for_sufficient=2)


class AssessSufficiencyTests(unittest.TestCase):
    def test_none_is_unscored(self) -> None:
        res = assess_sufficiency(None, _T)
        self.assertIs(res.state, Sufficiency.UNSCORED)
        self.assertEqual(res.kept_indices, ())
        self.assertIsNone(res.max_score)

    def test_empty_is_none(self) -> None:
        res = assess_sufficiency([], _T)
        self.assertIs(res.state, Sufficiency.NONE)

    def test_all_below_keep_is_insufficient(self) -> None:
        res = assess_sufficiency([-0.01, -5.0, -11.0], _T)
        self.assertIs(res.state, Sufficiency.INSUFFICIENT)
        self.assertEqual(res.kept_indices, ())
        self.assertAlmostEqual(res.max_score, -0.01)
        self.assertIn("none reached", res.rationale)

    def test_boundaries_are_inclusive(self) -> None:
        # keep is inclusive (0.0 kept), strong is inclusive (2.0 strong).
        res = assess_sufficiency([0.0, 2.0, 1.99], _T)
        self.assertEqual(res.kept_indices, (1, 2, 0))
        self.assertEqual(res.strong_count, 1)
        self.assertIs(res.state, Sufficiency.PARTIAL)

    def test_two_strong_is_sufficient(self) -> None:
        res = assess_sufficiency([2.0, -3.0, 7.5], _T)
        self.assertIs(res.state, Sufficiency.SUFFICIENT)
        self.assertEqual(res.kept_indices, (2, 0))
        self.assertEqual(res.strong_count, 2)

    def test_ties_keep_input_order(self) -> None:
        res = assess_sufficiency([3.0, 3.0, 3.0], _T)
        self.assertEqual(res.kept_indices, (0, 1, 2))

    def test_scientific_thresholds_are_the_calibrated_ones(self) -> None:
        self.assertEqual(
            (SCIENTIFIC_THRESHOLDS.keep, SCIENTIFIC_THRESHOLDS.strong,
             SCIENTIFIC_THRESHOLDS.min_strong_for_sufficient),
            (0.0, 2.0, 2),
        )

    def test_relevance_probability(self) -> None:
        self.assertAlmostEqual(relevance_probability(0.0), 0.5)
        self.assertGreater(relevance_probability(8.0), 0.99)
        self.assertLess(relevance_probability(-8.0), 0.01)
        self.assertGreater(relevance_probability(-1000.0), -1e-12)  # no overflow
        self.assertLessEqual(relevance_probability(1000.0), 1.0)


_SEMA_1 = {
    "title": "Semaglutide and cardiovascular outcomes",
    "snippet": "Semaglutide reduced major adverse cardiovascular events.",
    "full_text": "Semaglutide reduced major adverse cardiovascular events in adults with type 2 diabetes.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    "_adapter": "pubmed",
    "doi": "10.1000/test.1",
    "publication_date": "2024-01",
    "peer_reviewed": True,
}
_SEMA_2 = {
    "title": "Semaglutide subgroup analysis",
    "snippet": "No significant benefit for semaglutide in this cohort.",
    "full_text": "No significant benefit was observed for semaglutide cardiovascular endpoints in this cohort.",
    "url": "https://openalex.org/W123",
    "_adapter": "openalex",
    "doi": "10.1000/test.2",
    "publication_date": "2023",
    "peer_reviewed": True,
}
_ANT_COLONY = {
    "title": "Ant colony optimisation for routing",
    "snippet": "Swarm heuristics for vehicle routing.",
    "url": "https://arxiv.org/abs/0000.0001",
    "_adapter": "arxiv",
}
_ADAPTERS = ["pubmed", "openalex", "arxiv"]


def _topic_scorer(_q: str, docs: list[str]) -> list[float]:
    return [5.0 if "emaglutide" in d else -8.0 for d in docs]


def _irrelevant_scorer(_q: str, docs: list[str]) -> list[float]:
    return [-6.0 for _ in docs]


class _PipelineHarness(unittest.TestCase):
    """Runs the real scientific pipeline with a stub reranker and pinned adapters (no network)."""

    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def _run(self, rows=(_SEMA_1, _SEMA_2, _ANT_COLONY), query="semaglutide cardiovascular outcomes"):
        from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
        from core.knowledge.web_retrieval import run_v2_web_retrieval

        by_adapter = {a: [dict(r) for r in rows if r["_adapter"] == a] for a in _ADAPTERS}
        with patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None), patch(
            "core.knowledge.pipeline_scientific.set_cached_rows"
        ), patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters", return_value=list(_ADAPTERS)
        ), patch(
            "core.knowledge.pipeline_scientific.tiered_scientific_retrieval_enabled", return_value=False
        ), patch.dict(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {a: (lambda q, max_results=3, _a=a: [dict(r) for r in by_adapter[_a]]) for a in _ADAPTERS},
            clear=False,
        ):
            return run_v2_web_retrieval(
                query=query,
                semantic_query=query,
                knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            )


class ScientificPipelineGateTests(_PipelineHarness):
    def test_unscored_path_is_legacy(self) -> None:
        set_reranker_scorer_for_tests(None)
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False):
            out = self._run()
        assert out.bundle is not None
        self.assertEqual(out.bundle.sufficiency, "unscored")
        self.assertEqual(out.relevance_diag["sufficiency"], "unscored")
        self.assertEqual(out.relevance_diag["rerank"]["skipped_reason"], SKIP_DISABLED)
        self.assertNotIn("rerank_dropped", out.relevance_diag)
        # Same selection and coverage as the legacy ranker on the same rows.
        from core.knowledge.bundle_builder import _compute_scientific_coverage

        self.assertEqual(out.bundle.coverage, _compute_scientific_coverage(out.bundle.sources)[0])
        self.assertTrue(all(0.0 <= s.relevance_score <= 1.0 for s in out.bundle.sources))

    def test_unscored_selection_matches_legacy_rank_candidates(self) -> None:
        from core.knowledge.pipeline_scientific import _rank_candidates, _rank_candidates_gated
        from core.knowledge.types import RetrievalContext

        ctx = RetrievalContext(query="semaglutide outcomes", semantic_query="semaglutide outcomes")
        rows = [dict(r) for r in (_SEMA_1, _SEMA_2, _ANT_COLONY)]
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False):
            gated = _rank_candidates_gated([dict(r) for r in rows], ctx=ctx, max_results=3)
        legacy, legacy_rej = _rank_candidates([dict(r) for r in rows], ctx=ctx, max_results=3)
        self.assertFalse(gated.scored)
        self.assertEqual([r["url"] for r in gated.kept], [r["url"] for r in legacy])
        self.assertEqual([r["url"] for r in gated.rejected], [r["url"] for r in legacy_rej])

    def test_irrelevant_rows_are_dropped_and_recorded(self) -> None:
        set_reranker_scorer_for_tests(_topic_scorer)
        out = self._run()
        assert out.bundle is not None
        urls = [s.url for s in out.bundle.sources]
        self.assertNotIn(_ANT_COLONY["url"], urls)
        self.assertEqual(len(urls), 2)
        self.assertEqual(out.bundle.sufficiency, "sufficient")
        dropped = out.relevance_diag["rerank_dropped"]
        self.assertEqual([d["url"] for d in dropped], [_ANT_COLONY["url"]])
        self.assertEqual(dropped[0]["score"], -8.0)
        self.assertEqual(dropped[0]["adapter"], "arxiv")
        # relevance_score is sigmoid(logit) on scored rows.
        self.assertTrue(all(abs(s.relevance_score - relevance_probability(5.0)) < 1e-6
                            for s in out.bundle.sources))
        json.dumps(out.relevance_diag["rerank"])  # trace-serialisable

    def test_all_irrelevant_gives_insufficient_empty_bundle(self) -> None:
        set_reranker_scorer_for_tests(_irrelevant_scorer)
        out = self._run()
        b = out.bundle
        assert b is not None
        self.assertEqual(b.sources, ())
        self.assertEqual(b.sufficiency, "insufficient")
        self.assertEqual(b.stop_reason, "relevance_filtered")
        self.assertEqual(b.coverage, "none")
        self.assertEqual(b.confidence, 0.0)
        self.assertIn("none reached", b.coverage_rationale)
        self.assertEqual(len(out.relevance_diag["rerank_dropped"]), 3)
        self.assertEqual(out.relevance_diag["sufficiency"], "insufficient")
        summary = b.summary_for_skills()
        self.assertEqual(summary.sufficiency, "insufficient")
        self.assertFalse(summary.present)

    def test_partial_caps_confidence_and_marks_poor(self) -> None:
        set_reranker_scorer_for_tests(
            lambda _q, docs: [1.0 if "subgroup" in d else (5.0 if "emaglutide" in d else -8.0) for d in docs]
        )
        out = self._run()
        b = out.bundle
        assert b is not None
        self.assertEqual(b.sufficiency, "partial")  # one strong (5.0) + one weak (1.0)
        self.assertEqual(b.coverage, "poor")
        self.assertLessEqual(b.confidence, 0.55)

    def test_trace_records_sufficiency(self) -> None:
        from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace

        set_reranker_scorer_for_tests(_irrelevant_scorer)
        out = self._run()
        payload = serialize_retrieval_trace(
            build_retrieval_trace(out.bundle, relevance_diag=out.relevance_diag)
        )
        self.assertEqual(payload["sufficiency"], "insufficient")
        self.assertEqual(len(payload["relevance_diag"]["rerank_dropped"]), 3)
        json.dumps(payload)


class SlotReservationGateTests(unittest.TestCase):
    """Discipline slots may only reserve rows that passed the gate."""

    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def _gated(self, scorer, max_results=2):
        from core.knowledge.pipeline_scientific import _rank_candidates_gated
        from core.knowledge.types import RetrievalContext

        set_reranker_scorer_for_tests(scorer)
        ctx = RetrievalContext(query="semaglutide outcomes", semantic_query="semaglutide outcomes")
        return _rank_candidates_gated(
            [dict(r) for r in (_SEMA_1, _SEMA_2, _ANT_COLONY)], ctx=ctx, max_results=max_results
        )

    def test_irrelevant_arxiv_row_not_reserved(self) -> None:
        from core.knowledge.pipeline_scientific import _ensure_physics_arxiv_slot
        from core.knowledge.scientific_discipline import SCIENTIFIC_DISCIPLINE_PHYSICS

        ranking = self._gated(_topic_scorer)
        self.assertNotIn("arxiv", [r["_adapter"] for r in ranking.slot_pool])
        kept = _ensure_physics_arxiv_slot(
            ranking.kept, ranking.slot_pool, discipline=SCIENTIFIC_DISCIPLINE_PHYSICS,
            query="q", max_results=2,
        )
        self.assertNotIn("arxiv", [r["_adapter"] for r in kept])

    def test_relevant_arxiv_row_is_reserved(self) -> None:
        from core.knowledge.pipeline_scientific import _ensure_physics_arxiv_slot
        from core.knowledge.scientific_discipline import SCIENTIFIC_DISCIPLINE_PHYSICS

        ranking = self._gated(
            lambda _q, docs: [9.0 if "emaglutide" in d else 0.5 for d in docs], max_results=1
        )
        self.assertNotIn("arxiv", [r["_adapter"] for r in ranking.kept])  # budget 1, lower score
        self.assertIn("arxiv", [r["_adapter"] for r in ranking.slot_pool])  # passed keep
        kept = _ensure_physics_arxiv_slot(
            ranking.kept, ranking.slot_pool, discipline=SCIENTIFIC_DISCIPLINE_PHYSICS,
            query="q", max_results=1,
        )
        self.assertEqual([r["_adapter"] for r in kept], ["arxiv"])


class InsufficientPromptTests(unittest.TestCase):
    def _sys(self, **kw):
        from core.prompt_blocks import build_prompt_blocks, compose_system_prompt

        base = dict(
            execution_route="NONE", explicit_remember_active=False, has_retrieval_sources=False,
            knowledge_service="scientific_evidence", current_date="",
        )
        base.update(kw)
        blocks = build_prompt_blocks(**base)
        return blocks, compose_system_prompt(blocks)

    def test_insufficient_suffix_present_and_web_empty_absent(self) -> None:
        from core.memory_filters import (
            COMPOSER_WEB_EMPTY_SUFFIX,
            EXPLICIT_WEB_EMPTY_SUFFIX,
            RETRIEVAL_INSUFFICIENT_SUFFIX,
        )

        for route in ("NONE", "WEB"):
            with self.subTest(route=route):
                blocks, sys_p = self._sys(
                    execution_route=route, retrieval_insufficient=True,
                    # Even if a caller also flags web-empty, insufficient wins.
                    explicit_web_empty_results=True,
                )
                self.assertIn(RETRIEVAL_INSUFFICIENT_SUFFIX.strip(), sys_p)
                self.assertNotIn(EXPLICIT_WEB_EMPTY_SUFFIX.strip()[:60], sys_p)
                self.assertNotIn(COMPOSER_WEB_EMPTY_SUFFIX.strip()[:60], sys_p)
                self.assertTrue(blocks.no_sources_mode)

    def test_default_is_unchanged(self) -> None:
        from core.memory_filters import RETRIEVAL_INSUFFICIENT_SUFFIX

        _, sys_p = self._sys(execution_route="WEB")
        self.assertNotIn(RETRIEVAL_INSUFFICIENT_SUFFIX.strip(), sys_p)


class DroppedSourcesNeverReachModelTests(_PipelineHarness):
    """D2: gate-dropped rows are trace-only — not in sources, UI payload, prompt or ids."""

    def test_dropped_rows_absent_from_ui_sources_and_prompt(self) -> None:
        from types import SimpleNamespace

        from core.knowledge.ui_adapter import append_turn_evidence_bundle_sources
        from workers.llm_worker import LLMWorker

        set_reranker_scorer_for_tests(_topic_scorer)
        out = self._run()
        dropped_title = _ANT_COLONY["title"]
        self.assertIn(dropped_title, [d["title"] for d in out.relevance_diag["rerank_dropped"]])
        ui_sources: list = []
        append_turn_evidence_bundle_sources(ui_sources, out.bundle)
        self.assertEqual(len(ui_sources), 2)
        self.assertNotIn(dropped_title, json.dumps(ui_sources))
        fake_self = SimpleNamespace(MAX_TOTAL_RETRIEVAL_CHARS=LLMWorker.MAX_TOTAL_RETRIEVAL_CHARS)
        prompt = LLMWorker._format_sources_for_llm_prompt(fake_self, ui_sources)
        self.assertNotIn(dropped_title, prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("[2]", prompt)
        self.assertNotIn("[3]", prompt)
        self.assertEqual([s.id for s in out.bundle.sources], ["ek_1", "ek_2"])


class ConsideredNotUsedTests(_PipelineHarness):
    """D2: dropped sources render in the trace summary and the Retrieval Inspector."""

    def _payload(self):
        from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace

        set_reranker_scorer_for_tests(_topic_scorer)
        out = self._run()
        return serialize_retrieval_trace(
            build_retrieval_trace(out.bundle, relevance_diag=out.relevance_diag),
            sources=out.bundle.sources,
        )

    def test_trace_summary_lists_dropped_sources(self) -> None:
        from core.knowledge.retrieval_trace_reader import format_retrieval_trace_summary

        text = format_retrieval_trace_summary(self._payload())
        self.assertIn("Relevance gate: sufficient", text)
        self.assertIn("Considered, not used (1):", text)
        self.assertIn(f"{_ANT_COLONY['title']} [arxiv] — relevance -8.00", text)

    def test_unscored_trace_says_gate_not_applied(self) -> None:
        from core.knowledge.retrieval_trace_reader import format_relevance_gate_lines

        lines = format_relevance_gate_lines(
            {"relevance_diag": {"rerank": {"skipped_reason": "model_missing"}, "sufficiency": "unscored"}}
        )
        self.assertEqual(lines, ["Relevance gate: not applied (model_missing); legacy ranking used"])

    def test_inspector_summary_tab_shows_section(self) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])  # noqa: F841
        from ui.components.retrieval_inspector import RetrievalInspector

        inspector = RetrievalInspector()
        inspector.load(trace=self._payload())
        text = inspector._summary.toPlainText()
        self.assertIn("Considered, not used (1):", text)
        self.assertIn(_ANT_COLONY["title"], text)


class BundleFieldTests(unittest.TestCase):
    def _bundle(self, sufficiency="partial"):
        from core.knowledge.bundle_builder import build_scientific_evidence_bundle

        row = dict(_SEMA_1, _scientific_relevance=0.9, _rerank_score=3.0)
        return build_scientific_evidence_bundle(
            query_raw="q", query_resolved="q", kept_rows=[row], rejected_count=0,
            latency_ms=1.0, sufficiency=sufficiency, sufficiency_rationale="r",
        )

    def test_codec_round_trip_keeps_sufficiency(self) -> None:
        from core.knowledge.graph.bundle_codec import bundle_from_dict, bundle_to_dict

        b = self._bundle("partial")
        self.assertEqual(bundle_from_dict(bundle_to_dict(b)).sufficiency, "partial")
        legacy = bundle_to_dict(b)
        legacy.pop("sufficiency")
        self.assertEqual(bundle_from_dict(legacy).sufficiency, "unscored")

    def test_conflict_copy_keeps_sufficiency(self) -> None:
        from core.knowledge.pipeline_scientific import _bundle_with_conflicts

        b = self._bundle("sufficient")
        out = _bundle_with_conflicts(b, ())
        self.assertEqual(out.sufficiency, "sufficient")
        self.assertIn("material_conflict", out.warnings)
        self.assertEqual(
            {f.name for f in dataclasses.fields(out)} - {"warnings", "conflicts"},
            {f.name for f in dataclasses.fields(b)} - {"warnings", "conflicts"},
        )

    def test_unscored_default_keeps_legacy_stop_reason(self) -> None:
        from core.knowledge.bundle_builder import build_empty_bundle
        from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

        legacy = build_empty_bundle(
            query_raw="q", query_resolved="q", latency_ms=0.0,
            stop_reason="relevance_filtered", knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
        )
        self.assertEqual(legacy.stop_reason, "no_evidence")  # unchanged pre-S4 behaviour
        self.assertEqual(legacy.sufficiency, "unscored")


def _model_on_disk() -> bool:
    from core.knowledge.ranking import rerank

    return rerank._model_path() is not None


@unittest.skipUnless(_model_on_disk(), "reranker model not downloaded")
@pytest.mark.real_reranker
class RealAntPoolAcceptanceTests(unittest.TestCase):
    """S4 acceptance: the real ANT @evidence pool is insufficient → suffix, zero sources."""

    def test_ant_pool_insufficient_and_prompt_carries_suffix(self) -> None:
        from core.knowledge.bundle_builder import build_empty_bundle
        from core.knowledge.pipeline_scientific import _rank_candidates_gated
        from core.knowledge.quality_eval import fixture_pool, fixture_ranking_query, load_json
        from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_SCIENTIFIC_EVIDENCE
        from core.memory_filters import EXPLICIT_WEB_EMPTY_SUFFIX, RETRIEVAL_INSUFFICIENT_SUFFIX
        from core.prompt_blocks import build_prompt_blocks, compose_system_prompt

        set_reranker_scorer_for_tests(None)
        fx = load_json("eval/fixtures/knowledge_quality/ant_ai_candidates.json")
        query = fixture_ranking_query(fx, query_mode="production")
        pool = fixture_pool(fx, query_mode="production")
        ctx = RetrievalContext(query=query, semantic_query=query, budget=RetrievalBudget(max_results=3))
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True):
            ranking = _rank_candidates_gated(pool, ctx=ctx, max_results=3, question=query)
        self.assertTrue(ranking.scored, ranking.rerank_diag)
        self.assertIs(ranking.sufficiency.state, Sufficiency.INSUFFICIENT)
        self.assertEqual(ranking.kept, [])
        self.assertEqual(len(ranking.dropped), len(pool))
        bundle = build_empty_bundle(
            query_raw=query, query_resolved=query, latency_ms=0.0, stop_reason="relevance_filtered",
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE, sufficiency="insufficient",
            sufficiency_rationale=ranking.sufficiency.rationale,
        )
        self.assertEqual(bundle.sources, ())
        sys_p = compose_system_prompt(build_prompt_blocks(
            execution_route="NONE", explicit_remember_active=False, has_retrieval_sources=False,
            retrieval_insufficient=bundle.sufficiency == "insufficient", current_date="",
            knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
        ))
        self.assertIn(RETRIEVAL_INSUFFICIENT_SUFFIX.strip(), sys_p)
        self.assertNotIn(EXPLICIT_WEB_EMPTY_SUFFIX.strip()[:60], sys_p)


if __name__ == "__main__":
    unittest.main()
