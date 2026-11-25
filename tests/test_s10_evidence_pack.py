"""Plan S10: evidence pack / prompt close-out.

Service personas (R6), the n_ctx-aware retrieval budget (R5), the coverage
note (R5), honest Semantic Scholar review flags (V8) and adapter attribution
labels (R7).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import prompt_blocks as pb  # noqa: E402
from core.knowledge.adapters import semantic_scholar as s2  # noqa: E402
from core.knowledge.bundle_builder import (  # noqa: E402
    build_finance_knowledge_bundle,
    build_general_web_bundle,
    build_legal_knowledge_bundle,
    build_scientific_evidence_bundle,
    build_trusted_knowledge_bundle,
)
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.quality_eval import CONTRACT_FLAGS, evaluate_ant_fixture, prompt_allows_abstention  # noqa: E402
from core.knowledge.ranking.rerank import set_reranker_scorer_for_tests  # noqa: E402
from core.knowledge.ranking.sufficiency import SufficiencyThresholds  # noqa: E402
from core.knowledge.ranking.web_gate import gate_rows  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402
from core.knowledge.types import RetrievalContext  # noqa: E402
from core.knowledge.ui_adapter import bundle_to_ui_sources  # noqa: E402
from core.memory_filters import (  # noqa: E402
    RETRIEVAL_PARTIAL_NOTE,
    prefix_evidence_notes,
    retrieval_coverage_note,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt  # noqa: E402
from core.prompt_renderers import render_messages  # noqa: E402
from core.retrieval_budget import (  # noqa: E402
    DEFAULT_RETRIEVAL_CHARS,
    MIN_RETRIEVAL_CHARS,
    estimate_tokens,
    output_reserve_tokens,
    retrieval_char_budget,
)


def _blocks(service: str, **kw):
    params = dict(
        execution_route="WEB",
        explicit_remember_active=False,
        has_retrieval_sources=True,
        retrieval_source_count=3,
        web_hit_count=3,
        knowledge_service=service,
        conversation_history=[{"role": "user", "content": "q"}],
        retrieval_context="--- [1]: Example ---\nBody.",
        current_date="2026-11-24",
        engine_mode="internal",
    )
    params.update(kw)
    return build_prompt_blocks(**params)


# ---------------------------------------------------------------------------
# Personas (R6)
# ---------------------------------------------------------------------------


class ServicePersonaTests(unittest.TestCase):
    def test_each_service_gets_its_persona(self) -> None:
        cases = {
            "general_web": pb._WEB_PERSONA,
            "": pb._WEB_PERSONA,
            "scientific_evidence": pb._SCIENTIFIC_EVIDENCE_PERSONA,
            "trusted_knowledge": pb._TRUSTED_KNOWLEDGE_PERSONA,
            "wikipedia": pb._TRUSTED_KNOWLEDGE_PERSONA,
            "finance_knowledge": pb._FINANCE_KNOWLEDGE_PERSONA,
            "legal_knowledge": pb._LEGAL_KNOWLEDGE_PERSONA,
        }
        for service, persona in cases.items():
            with self.subTest(service=service):
                system = compose_system_prompt(_blocks(service))
                self.assertTrue(system.startswith(persona), service)

    def test_service_personas_keep_the_shared_answer_rules(self) -> None:
        for persona in (
            pb._TRUSTED_KNOWLEDGE_PERSONA,
            pb._FINANCE_KNOWLEDGE_PERSONA,
            pb._LEGAL_KNOWLEDGE_PERSONA,
        ):
            self.assertTrue(persona.endswith(pb._RETRIEVAL_PERSONA_RULES))
        # The general web persona is unchanged by S10: same rules, same wording.
        self.assertTrue(pb._WEB_PERSONA.endswith(pb._RETRIEVAL_PERSONA_RULES))

    def test_every_service_allows_abstention_and_keeps_citation_discipline(self) -> None:
        from core.memory_filters import CITATION_DISCIPLINE_SUFFIX

        for service in ("general_web", "trusted_knowledge", "finance_knowledge", "legal_knowledge"):
            with self.subTest(service=service):
                system = compose_system_prompt(_blocks(service))
                self.assertTrue(prompt_allows_abstention(system))
                self.assertIn(CITATION_DISCIPLINE_SUFFIX.strip()[:60], system)
                self.assertIn("Today's date is 2026-11-24", system)

    def test_disclaimers_compose_with_service_personas(self) -> None:
        from core.memory_filters import FINANCIAL_DISCLAIMER_SUFFIX, LEGAL_DISCLAIMER_SUFFIX

        fin = compose_system_prompt(_blocks("finance_knowledge", financial_disclaimer=True))
        self.assertTrue(fin.startswith(pb._FINANCE_KNOWLEDGE_PERSONA))
        self.assertIn(FINANCIAL_DISCLAIMER_SUFFIX.strip()[:50], fin)
        legal = compose_system_prompt(_blocks("legal_knowledge", legal_disclaimer=True))
        self.assertTrue(legal.startswith(pb._LEGAL_KNOWLEDGE_PERSONA))
        self.assertIn(LEGAL_DISCLAIMER_SUFFIX.strip()[:50], legal)

    def test_no_sources_turn_keeps_base_persona(self) -> None:
        system = compose_system_prompt(
            _blocks("trusted_knowledge", has_retrieval_sources=False, retrieval_source_count=0, web_hit_count=0)
        )
        self.assertNotIn("Reference sources", system)

    def test_phase1_contract_flags_stay_ok(self) -> None:
        result = evaluate_ant_fixture()
        for flag in CONTRACT_FLAGS:
            self.assertTrue(result[flag], flag)


# ---------------------------------------------------------------------------
# n_ctx-aware retrieval budget (R5)
# ---------------------------------------------------------------------------

_MATRIX = (4096, 8192, 32768)


def _history(turns: int, chars: int) -> list[dict]:
    out: list[dict] = []
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": ("word " * (chars // 5))[:chars]})
    out.append({"role": "user", "content": "How do heat pumps compare with gas boilers on running cost?"})
    return out


def _sources(n: int = 8, body_chars: int = 1500) -> list[dict]:
    text = (
        "Heat pumps move heat with a refrigerant cycle; running cost depends on the "
        "coefficient of performance and on electricity and gas prices. "
    )
    return [
        {
            "id": i,
            "filename": f"Source {i}: heat pump running costs",
            "content": (text * 40)[:body_chars],
            "type": "web",
            "venue": "Example Journal",
            "publication_date": "2024",
            "peer_reviewed": True,
        }
        for i in range(1, n + 1)
    ]


def _format(sources: list[dict], budget: int) -> str:
    from workers.llm_worker import LLMWorker

    fake = SimpleNamespace(MAX_TOTAL_RETRIEVAL_CHARS=DEFAULT_RETRIEVAL_CHARS)
    return LLMWorker._format_sources_for_llm_prompt(fake, sources, char_budget=budget)


def _prompt_tokens(history: list[dict], retrieval: str, *, skills: str, service: str) -> int:
    blocks = build_prompt_blocks(
        execution_route="WEB",
        explicit_remember_active=False,
        has_retrieval_sources=True,
        retrieval_source_count=8,
        web_hit_count=8,
        knowledge_service=service,
        conversation_history=history,
        retrieval_context=retrieval,
        current_date="2026-11-24",
        engine_mode="internal",
        evidence_has_conflicts=True,
        scientific_medical_disclaimer=True,
        financial_disclaimer=True,
        legal_disclaimer=True,
        skill_guidance=skills,
        preference_context="The user prefers concise answers with metric units. " * 4,
        reply_shape_hint="Answer in two short paragraphs.",
    )
    messages = render_messages(blocks, "system_ok")
    return sum(estimate_tokens(str(m.get("content") or "")) + 8 for m in messages)


class RetrievalBudgetTests(unittest.TestCase):
    def test_unknown_n_ctx_keeps_todays_default(self) -> None:
        for n_ctx in (None, 0, -1, "x"):
            budget = retrieval_char_budget(n_ctx=n_ctx)  # type: ignore[arg-type]
            self.assertEqual(budget.chars, DEFAULT_RETRIEVAL_CHARS)
            self.assertEqual(budget.reason, "default")

    def test_large_context_is_capped_at_todays_budget(self) -> None:
        budget = retrieval_char_budget(n_ctx=32768, history=_history(6, 400), max_output_tokens=4096)
        self.assertEqual(budget.chars, DEFAULT_RETRIEVAL_CHARS)
        self.assertEqual(budget.reason, "cap")

    def test_small_context_shrinks_and_more_history_shrinks_more(self) -> None:
        light = retrieval_char_budget(n_ctx=4096, history=_history(2, 200), max_output_tokens=4096)
        heavy = retrieval_char_budget(n_ctx=4096, history=_history(6, 600), max_output_tokens=4096)
        self.assertEqual(light.reason, "n_ctx")
        self.assertLess(light.chars, DEFAULT_RETRIEVAL_CHARS)
        self.assertLess(heavy.chars, light.chars)

    def test_floor_when_history_fills_the_window(self) -> None:
        budget = retrieval_char_budget(n_ctx=4096, history=_history(10, 1500), max_output_tokens=4096)
        self.assertEqual(budget.reason, "floor")
        self.assertEqual(budget.chars, MIN_RETRIEVAL_CHARS)

    def test_output_reserve(self) -> None:
        self.assertEqual(output_reserve_tokens(4096, 4096), 1024)
        self.assertEqual(output_reserve_tokens(32768, 4096), 4096)
        self.assertEqual(output_reserve_tokens(32768, 600), 600)
        self.assertEqual(output_reserve_tokens(1024, None), 512)

    def test_token_room_only_limits_when_the_window_needs_it(self) -> None:
        big = retrieval_char_budget(n_ctx=32768, history=_history(4, 300), max_output_tokens=4096)
        self.assertEqual(big.reason, "cap")
        # 8000 CJK characters (~8000 tokens) still fit a 32k window: no re-fit.
        self.assertGreater(big.tokens, estimate_tokens("热" * DEFAULT_RETRIEVAL_CHARS))
        small = retrieval_char_budget(n_ctx=4096, history=_history(2, 200), max_output_tokens=4096)
        self.assertEqual(small.reason, "n_ctx")
        self.assertLess(small.tokens, estimate_tokens("热" * small.chars))
        floor = retrieval_char_budget(n_ctx=4096, history=_history(10, 1500), max_output_tokens=4096)
        self.assertIsNone(floor.tokens)
        self.assertIsNone(retrieval_char_budget(n_ctx=None).tokens)

    def test_non_ascii_text_counts_more_tokens(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcdefg"), 2)
        self.assertEqual(estimate_tokens("热泵运行成本"), 6)

    def test_matrix_prompt_never_exceeds_n_ctx(self) -> None:
        """Acceptance: on 4k / 8k / 32k, prompt + answer reserve fit the window.

        Worst-case system prompt (every disclaimer, conflict suffix, a full
        skills block, preferences, reply hint) and eight 1500-char sources.
        Floor cases (history alone fills the window) are excluded: there the
        overflow comes from history, which S10 does not trim.
        """
        skills = "S" * 1200
        for n_ctx in _MATRIX:
            for turns, chars in ((0, 0), (4, 300), (6, 600), (8, 900)):
                for service in ("general_web", "scientific_evidence", "finance_knowledge"):
                    with self.subTest(n_ctx=n_ctx, turns=turns, chars=chars, service=service):
                        history = _history(turns, chars)
                        budget = retrieval_char_budget(
                            n_ctx=n_ctx, history=history, skills_chars=len(skills), max_output_tokens=4096
                        )
                        if budget.reason == "floor":
                            continue
                        body = _format(_sources(), budget.chars)
                        self.assertLessEqual(len(body), budget.chars)
                        prompt = _prompt_tokens(history, body, skills=skills, service=service)
                        reserve = output_reserve_tokens(n_ctx, 4096)
                        self.assertLessEqual(prompt + reserve, n_ctx)

    def test_old_fixed_budget_overflowed_4k(self) -> None:
        """Before S10 the 8000-char block overflowed a 4k window with modest history."""
        history = _history(6, 600)
        body = _format(_sources(), DEFAULT_RETRIEVAL_CHARS)
        prompt = _prompt_tokens(history, body, skills="S" * 1200, service="general_web")
        self.assertGreater(prompt + output_reserve_tokens(4096, 4096), 4096)


class WorkerBudgetWiringTests(unittest.TestCase):
    def _fake(self, **kw):
        from workers.llm_worker import LLMWorker

        fake = SimpleNamespace(
            engine_mode="internal",
            _native_engine=SimpleNamespace(loaded_n_ctx=lambda: 4096),
            context_window=32000,
            prompt="How do heat pumps work?",
            MAX_TOTAL_RETRIEVAL_CHARS=DEFAULT_RETRIEVAL_CHARS,
            _max_tokens_completion=lambda native: 4096,
        )
        for k, v in kw.items():
            setattr(fake, k, v)
        fake._active_context_window = lambda: LLMWorker._active_context_window(fake)
        return fake, LLMWorker

    def test_native_engine_loaded_window_wins_over_setting(self) -> None:
        fake, worker = self._fake()
        self.assertEqual(worker._active_context_window(fake), 4096)
        budget = worker._retrieval_budget_for_turn(fake, _history(4, 300))
        self.assertEqual(budget.n_ctx, 4096)
        self.assertLess(budget.chars, DEFAULT_RETRIEVAL_CHARS)

    def test_external_engine_uses_context_setting(self) -> None:
        fake, worker = self._fake(engine_mode="external")
        self.assertEqual(worker._active_context_window(fake), 32000)
        self.assertEqual(worker._retrieval_budget_for_turn(fake, _history(4, 300)).chars, DEFAULT_RETRIEVAL_CHARS)

    def test_unloaded_native_engine_falls_back_to_setting(self) -> None:
        fake, worker = self._fake(_native_engine=SimpleNamespace(loaded_n_ctx=lambda: None), context_window=8192)
        self.assertEqual(worker._active_context_window(fake), 8192)

    def test_native_engine_loaded_n_ctx(self) -> None:
        from workers.native_llama_engine import NativeLlamaEngine

        engine = NativeLlamaEngine.__new__(NativeLlamaEngine)
        engine._llama = None
        engine._n_ctx = 4096
        self.assertIsNone(engine.loaded_n_ctx())
        engine._llama = SimpleNamespace(n_ctx=lambda: 6144)
        self.assertEqual(engine.loaded_n_ctx(), 6144)


# ---------------------------------------------------------------------------
# Coverage note (R5)
# ---------------------------------------------------------------------------


class CoverageNoteTests(unittest.TestCase):
    def test_note_wording(self) -> None:
        sources = [
            {"id": 1, "directly_relevant": True},
            {"id": 2, "directly_relevant": True},
            {"id": 3, "directly_relevant": False},
            {"id": 4, "type": "memory"},  # not scored, not counted
        ]
        self.assertEqual(
            retrieval_coverage_note("partial", sources),
            "[Coverage: partial — 2 of 3 search results directly address the question.]",
        )
        self.assertEqual(
            retrieval_coverage_note("sufficient", [{"directly_relevant": True}]),
            "[Coverage: sufficient — 1 of 1 search result directly addresses the question.]",
        )

    def test_no_note_when_unscored_or_nothing_scored(self) -> None:
        self.assertEqual(retrieval_coverage_note("unscored", [{"directly_relevant": True}]), "")
        self.assertEqual(retrieval_coverage_note("insufficient", [{"directly_relevant": False}]), "")
        self.assertEqual(retrieval_coverage_note("partial", [{"id": 1, "type": "memory"}]), "")
        self.assertEqual(retrieval_coverage_note("", []), "")

    def test_note_goes_first_and_keeps_partial_note(self) -> None:
        note = retrieval_coverage_note("partial", [{"directly_relevant": False}])
        body = prefix_evidence_notes("--- [1]: X ---\nBody", partial=True, coverage_note=note)
        self.assertTrue(body.startswith(note))
        self.assertIn(RETRIEVAL_PARTIAL_NOTE, body)
        self.assertLess(body.index(RETRIEVAL_PARTIAL_NOTE), body.index("--- [1]"))
        # No sources → no notes, as before.
        self.assertEqual(prefix_evidence_notes("", partial=True, coverage_note=note), "")
        # Without a note the pack is unchanged from S7.
        self.assertEqual(
            prefix_evidence_notes("B", partial=False, coverage_note=""), "B"
        )


class DirectlyRelevantFlagTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def _gate(self):
        set_reranker_scorer_for_tests(
            lambda _q, docs: [6.0 if "basics" in d else (1.0 if "work" in d else -8.0) for d in docs]
        )
        rows = [
            {"title": "Heat pump basics", "snippet": "A heat pump moves heat.", "url": "https://example.org/a"},
            {"title": "How heat pumps work", "snippet": "Refrigerant.", "url": "https://example.org/b"},
            {"title": "Car chargers", "snippet": "EV.", "url": "https://example.org/c"},
        ]
        thresholds = SufficiencyThresholds(keep=0.0, strong=2.0, min_strong_for_sufficient=2)
        return gate_rows("what is a heat pump", rows, budget=3, thresholds=thresholds)

    def test_web_gate_marks_strong_rows_and_ui_sources_carry_it(self) -> None:
        gate = self._gate()
        self.assertEqual([r["_rerank_strong"] for r in gate.kept], [True, False])
        bundle = build_general_web_bundle(
            query_raw="q", query_resolved="q", kept_rows=gate.kept, rejected_count=1, latency_ms=1.0,
            sufficiency=gate.sufficiency.state.value,
        )
        ui = bundle_to_ui_sources(bundle)
        self.assertEqual([s["directly_relevant"] for s in ui], [True, False])
        self.assertEqual(
            retrieval_coverage_note(bundle.sufficiency, ui),
            "[Coverage: partial — 1 of 2 search results directly addresses the question.]",
        )

    def test_trusted_finance_scientific_builders_carry_flag(self) -> None:
        row = {"title": "T", "snippet": "S", "url": "https://en.wikipedia.org/wiki/T", "_rerank_score": 3.1, "_rerank_strong": True}
        trusted = build_trusted_knowledge_bundle(
            query_raw="q", query_resolved="q", kept_rows=[dict(row, _wiki_source=True)], rejected_count=0, latency_ms=1.0,
        )
        self.assertTrue(bundle_to_ui_sources(trusted)[0]["directly_relevant"])
        finance = build_finance_knowledge_bundle(
            query_raw="q", query_resolved="q", kept_rows=[dict(row, _adapter="fred", _rerank_strong=False)],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertFalse(bundle_to_ui_sources(finance)[0]["directly_relevant"])
        sci = build_scientific_evidence_bundle(
            query_raw="q", query_resolved="q", kept_rows=[dict(row, _adapter="openalex")], rejected_count=0,
            latency_ms=1.0,
        )
        self.assertTrue(bundle_to_ui_sources(sci)[0]["directly_relevant"])

    def test_unscored_rows_have_no_flag(self) -> None:
        bundle = build_general_web_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[{"title": "T", "snippet": "S", "url": "https://example.org/t"}],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertNotIn("directly_relevant", bundle_to_ui_sources(bundle)[0])

    def test_fetched_sections_carry_flag(self) -> None:
        from core.knowledge.pipeline_general_web import _gate_fetched_sections
        from core.knowledge.types import EvidenceObject

        def section(i: int, text: str) -> EvidenceObject:
            return EvidenceObject(
                id=f"ws_{i}", source_id=f"https://e.org/{i}#s", adapter="fetch_engine",
                retrieval_method="fetch", title=f"Section {i}", excerpt=text, full_text=text,
                url=f"https://e.org/{i}", document_type="web_page",
                relevance_score=0.5, authority_score=0.4, reliability_score=0.4,
                retrieved_at=0.0, fetch_status="full_extract",
            )

        set_reranker_scorer_for_tests(lambda _q, docs: [5.0 if "pump" in d else 0.5 for d in docs])
        kept, _final, _warn, _diag = _gate_fetched_sections(
            "what is a heat pump",
            [section(1, "A heat pump moves heat."), section(2, "Unrelated but positive.")],
            serp_rows=[],
            gate=SimpleNamespace(recency=None),
        )
        flags = {s.title: s.raw_metadata.get("rerank_strong") for s in kept}
        self.assertEqual(flags, {"Section 1": True, "Section 2": False})

    def test_scientific_gate_marks_strong_rows(self) -> None:
        from core.knowledge.pipeline_scientific import _rank_candidates_gated

        set_reranker_scorer_for_tests(lambda _q, docs: [5.0 if "emaglutide" in d else 0.5 for d in docs])
        rows = [
            {"title": "Semaglutide outcomes", "snippet": "Semaglutide reduced events.", "url": "https://x.org/1", "_adapter": "pubmed"},
            {"title": "Cardiology review", "snippet": "Heart outcomes.", "url": "https://x.org/2", "_adapter": "openalex"},
        ]
        ctx = RetrievalContext(query="semaglutide outcomes", semantic_query="semaglutide outcomes")
        ranking = _rank_candidates_gated(rows, ctx=ctx, max_results=2)
        flags = {r["title"]: r["_rerank_strong"] for r in ranking.kept}
        self.assertEqual(flags, {"Semaglutide outcomes": True, "Cardiology review": False})


# ---------------------------------------------------------------------------
# Semantic Scholar review status (V8)
# ---------------------------------------------------------------------------


class SemanticScholarReviewStatusTests(unittest.TestCase):
    def test_cases(self) -> None:
        cases = [
            # (paper, doi, venue) -> (peer_reviewed, preprint)
            (({"publicationTypes": ["JournalArticle"]}, "10.1/x", ""), (True, False)),
            (({}, "10.48550/arxiv.1706.03762", "NeurIPS"), (True, False)),  # named venue wins
            (({"externalIds": {"ArXiv": "2401.00001"}}, None, ""), (False, True)),
            (({}, "10.1101/2024.01.01.123", ""), (False, True)),
            (({"publicationTypes": ["JournalArticle"]}, "10.1101/x", "bioRxiv"), (False, True)),
            (({}, None, "arXiv.org"), (False, True)),
            (({"publicationVenue": {"name": "Nature", "type": "journal"}}, None, ""), (True, False)),
            (({}, None, ""), (None, False)),  # unknown: no label in the prompt
        ]
        for (paper, doi, venue), expected in cases:
            with self.subTest(paper=paper, doi=doi, venue=venue):
                self.assertEqual(s2.review_status(paper, doi=doi, venue=venue), expected)

    def test_row_flags_and_document_type(self) -> None:
        preprint = s2._row_from_paper(
            {"title": "A preprint", "abstract": "x", "externalIds": {"ArXiv": "1", "DOI": "10.48550/arXiv.1"}}
        )
        self.assertFalse(preprint["peer_reviewed"])
        self.assertTrue(preprint["preprint"])
        self.assertEqual(preprint["document_type"], "preprint")
        unknown = s2._row_from_paper({"title": "Unknown venue", "abstract": "x"})
        self.assertIsNone(unknown["peer_reviewed"])
        self.assertEqual(unknown["document_type"], "journal_abstract")

    def test_api_requests_review_fields(self) -> None:
        for field in ("publicationTypes", "journal", "publicationVenue"):
            self.assertIn(field, s2._FIELDS)


# ---------------------------------------------------------------------------
# Adapter attribution labels (R7)
# ---------------------------------------------------------------------------


def _discovery(provider: str, rows: list[dict]) -> DiscoveryResult:
    return DiscoveryResult(
        candidates=tuple(
            CandidateUrl(url=r["url"], title=r["title"], snippet=r["snippet"], source=provider) for r in rows
        ),
        raw_rows=tuple(rows),
        search_outcome=SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, candidate_count=len(rows)),
        provider_id=provider,
    )


class AttributionTests(unittest.TestCase):
    ROWS = [
        {"title": "Heat pump basics", "snippet": "A heat pump moves heat.", "url": "https://example.org/a"},
        {"title": "How heat pumps work", "snippet": "Refrigerant cycle.", "url": "https://example.org/b"},
    ]

    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def test_general_web_credits_the_discovery_provider(self) -> None:
        from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline
        from core.knowledge.retrieval_profiles import get_profile_spec

        set_reranker_scorer_for_tests(lambda _q, docs: [6.0 if "heat" in d.lower() else -8.0 for d in docs])
        spec = get_profile_spec("fast")
        ctx = RetrievalContext(query="what is a heat pump", semantic_query="what is a heat pump",
                               retrieval_profile=spec.id, budget=spec.budget)
        with patch(
            "core.knowledge.pipeline_general_web.discover_full",
            return_value=_discovery("brave_search", self.ROWS),
        ):
            bundle, _diag, _raw = run_general_web_evidence_pipeline(ctx)
        self.assertTrue(bundle.sources)
        self.assertEqual({s.adapter for s in bundle.sources}, {"brave_search"})
        self.assertEqual(bundle.adapter_calls, ("brave_search",))

    def test_duckduckgo_rows_still_say_duckduckgo(self) -> None:
        from core.knowledge.pipeline_general_web import _candidate_to_row

        row = _candidate_to_row(CandidateUrl(url="https://e.org", title="T", snippet="S", source="duckduckgo"))
        self.assertEqual(row["_discovery_provider"], "duckduckgo")
        bundle = build_general_web_bundle(query_raw="q", query_resolved="q", kept_rows=[row], rejected_count=0, latency_ms=1.0)
        self.assertEqual(bundle.sources[0].adapter, "duckduckgo")
        # Rows without a provider keep the pre-S10 label.
        legacy = build_general_web_bundle(
            query_raw="q", query_resolved="q", kept_rows=[{"title": "T", "url": "https://e.org"}],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertEqual(legacy.sources[0].adapter, "duckduckgo")

    def test_trusted_fallback_rows_credit_their_provider(self) -> None:
        bundle = build_trusted_knowledge_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[
                {"title": "W", "snippet": "s", "url": "https://en.wikipedia.org/wiki/W", "_wiki_source": True},
                {"title": "B", "snippet": "s", "url": "https://www.britannica.com/b", "_discovery_provider": "searxng"},
            ],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertEqual([s.adapter for s in bundle.sources], ["wikipedia_api", "searxng"])

    def test_untagged_scientific_row_is_not_credited_to_pubmed(self) -> None:
        bundle = build_scientific_evidence_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[{"title": "Untagged", "snippet": "x", "full_text": "x", "url": "https://x.org"}],
            rejected_count=0, latency_ms=1.0,
        )
        obj = bundle.sources[0]
        self.assertEqual(obj.adapter, "unknown")
        self.assertLess(obj.authority_score, 0.92)  # PubMed's index authority

    def test_finance_and_legal_do_not_invent_venues(self) -> None:
        finance = build_finance_knowledge_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[{"title": "Configured feed", "snippet": "x", "url": "https://x.org", "_adapter": "my_feed"}],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertIsNone(finance.sources[0].venue)
        self.assertNotEqual(finance.sources[0].document_type, "sec_filing")
        sec = build_finance_knowledge_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[{"title": "10-K", "snippet": "x", "url": "https://sec.gov/x", "_adapter": "sec_edgar"}],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertEqual(sec.sources[0].venue, "SEC EDGAR")
        self.assertEqual(sec.sources[0].document_type, "sec_filing")
        legal = build_legal_knowledge_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[{"title": "Statute", "snippet": "x", "url": "https://x.org", "_adapter": "my_laws"}],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertIsNone(legal.sources[0].venue)
        cl = build_legal_knowledge_bundle(
            query_raw="q", query_resolved="q",
            kept_rows=[{"title": "Case", "snippet": "x", "url": "https://x.org", "_adapter": "courtlistener"}],
            rejected_count=0, latency_ms=1.0,
        )
        self.assertEqual(cl.sources[0].venue, "CourtListener")
        self.assertEqual(cl.sources[0].document_type, "court_opinion")


if __name__ == "__main__":
    unittest.main()
