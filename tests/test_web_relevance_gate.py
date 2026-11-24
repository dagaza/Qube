"""Web retrieval relevance gate and pipeline helpers."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import classify_follow_up  # noqa: E402
from core.discourse_query import (  # noqa: E402
    resolve_search_target,
    resolve_web_query,
    web_query_rewrite_failed,
)
from core.discourse_state import update_discourse_state  # noqa: E402
from core.retrieval_relevance import (  # noqa: E402
    DEFAULT_WEB_MIN_TOKEN_OVERLAP,
    filter_web_results,
    passes_web_relevance_gate,
    query_snippet_token_overlap,
)


class TestQuerySnippetOverlap(unittest.TestCase):
    def test_birds_query_matches_bird_snippet(self) -> None:
        q = "Why do birds take dust baths?"
        snippet = "Birds take dust baths to remove parasites from feathers."
        overlap = query_snippet_token_overlap(q, snippet)
        self.assertGreaterEqual(overlap, DEFAULT_WEB_MIN_TOKEN_OVERLAP)
        self.assertTrue(passes_web_relevance_gate(q, "Dust bathing", snippet))

    def test_meta_query_rejects_unrelated_snippet(self) -> None:
        q = "Can you also do an online search for the answer?"
        snippet = "Learn how to configure online search settings in your browser."
        overlap = query_snippet_token_overlap(q, snippet)
        self.assertLess(overlap, DEFAULT_WEB_MIN_TOKEN_OVERLAP)
        self.assertFalse(passes_web_relevance_gate(q, "Online search", snippet))


class TestFilterWebResults(unittest.TestCase):
    def test_filters_irrelevant_hits(self) -> None:
        query = "Why do birds take dust baths?"
        items = [
            {
                "title": "Online search tips",
                "snippet": "How to search the web for answers to your questions.",
            },
            {
                "title": "Dust bathing in birds",
                "snippet": "Many bird species take dust baths to clean feathers.",
            },
        ]
        kept, diag = filter_web_results(
            query,
            items,
            use_embedding_gate=False,
        )
        self.assertEqual(diag["web_results_raw_count"], 2)
        self.assertEqual(len(kept), 1)
        self.assertIn("bird", kept[0]["snippet"].lower())

    def test_empty_when_all_irrelevant(self) -> None:
        query = "Why do birds take dust baths?"
        items = [
            {"title": "Search online", "snippet": "Use online search engines today."},
            {"title": "Web tips", "snippet": "Configure your browser search bar."},
        ]
        kept, diag = filter_web_results(query, items, use_embedding_gate=False)
        self.assertEqual(len(kept), 0)
        self.assertEqual(diag["web_results_kept_count"], 0)
        self.assertEqual(len(diag["web_relevance_dropped"]), 2)


class TestMetaWebPipeline(unittest.TestCase):
    def test_birds_follow_up_resolves_prior_question(self) -> None:
        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {"role": "assistant", "content": "Birds dust-bathe to stay clean."},
            {
                "role": "user",
                "content": (
                    "Yes that would be nice. Can you also do an online search "
                    "for the answer?"
                ),
            },
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            update_discourse_state(history, None, history[-1]["content"]),
            history,
        )
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")
        self.assertEqual(target.query, "Why do birds take dust baths?")
        self.assertFalse(
            web_query_rewrite_failed(
                history[-1]["content"],
                follow_up,
                target.query,
                explicit_web=True,
            )
        )

    def test_unresolved_meta_flags_rewrite_failed(self) -> None:
        prompt = "Can you search online for the answer?"
        follow_up = classify_follow_up(prompt, [], None)
        self.assertTrue(
            web_query_rewrite_failed(
                prompt,
                follow_up,
                prompt,
                explicit_web=True,
            )
        )

    def test_resolve_web_query_matches_search_target(self) -> None:
        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {"role": "assistant", "content": "Birds dust-bathe."},
            {"role": "user", "content": "Search online for the answer please."},
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        web_q = resolve_web_query(history[-1]["content"], follow_up, state, history)
        target = resolve_search_target(
            history[-1]["content"], follow_up, state, history
        )
        self.assertEqual(web_q, target.query)

    def test_meta_web_rewrite_skips_tool_only_prior_turn(self) -> None:
        history = [
            {"role": "user", "content": "What is the capital of Romania?"},
            {"role": "assistant", "content": "Bucharest is the capital of Romania."},
            {"role": "user", "content": "And how about its population size?"},
            {"role": "assistant", "content": "About 1.8 million in the city proper."},
            {"role": "user", "content": "@[tool:internet]"},
            {
                "role": "user",
                "content": "Can you also check online for the answer?",
            },
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            state,
            history,
        )
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")
        self.assertIn("Bucharest", target.query)
        self.assertIn("population", target.query.lower())

    def test_meta_web_rewrite_grounds_deictic_prior_with_referent(self) -> None:
        history = [
            {"role": "user", "content": "What is the capital of Romania?"},
            {"role": "assistant", "content": "Bucharest is the capital of Romania."},
            {"role": "user", "content": "Great, and how about its surface area?"},
            {"role": "assistant", "content": "About 228 km²."},
            {
                "role": "user",
                "content": "Can you check for the answer online to that last question?",
            },
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            state,
            history,
        )
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")
        self.assertEqual(target.query, "What is the surface area of Bucharest?")



# ---------------------------------------------------------------------------
# Plan S7: cross-encoder gate for web / trusted / finance (+ recency)
# ---------------------------------------------------------------------------

from datetime import date  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.ranking.rerank import set_reranker_scorer_for_tests  # noqa: E402
from core.knowledge.ranking.sufficiency import Sufficiency, relevance_probability  # noqa: E402
from core.knowledge.ranking.web_gate import (  # noqa: E402
    WARNING_RECENCY,
    WARNING_STALE,
    WARNING_UNDATED,
    WEB_THRESHOLDS,
    drop_search_ads,
    gate_rows,
    is_search_ad_url,
)
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402
from core.knowledge.types import RetrievalBudget, RetrievalContext  # noqa: E402

_S7_TODAY = date(2026, 11, 23)


def _keyword_scorer(scores: dict[str, float], default: float = -8.0):
    """Fake cross-encoder: a document scores ``scores[k]`` if it contains ``k``."""

    def score(_question: str, docs: list[str]) -> list[float]:
        out = []
        for doc in docs:
            hit = [v for k, v in scores.items() if k in doc]
            out.append(max(hit) if hit else default)
        return out

    return score


def _row(title: str, snippet: str = "", url: str | None = None) -> dict:
    return {"title": title, "snippet": snippet, "url": url or f"https://example.org/{title.lower().replace(' ', '-')}"}


class _ScorerCase(unittest.TestCase):
    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def use(self, scores: dict[str, float], default: float = -8.0) -> None:
        set_reranker_scorer_for_tests(_keyword_scorer(scores, default))


class SearchAdTests(unittest.TestCase):
    def test_ad_redirects(self) -> None:
        self.assertTrue(is_search_ad_url("https://duckduckgo.com/y.js?ad_domain=amazon.com&ad_provider=bing"))
        self.assertTrue(is_search_ad_url("https://links.duckduckgo.com/y.js?ad_domain=x"))
        self.assertFalse(is_search_ad_url("https://duckduckgo.com/about"))
        self.assertFalse(is_search_ad_url("https://example.com/y.js"))
        self.assertFalse(is_search_ad_url(""))

    def test_drop_search_ads(self) -> None:
        rows = [_row("Ad", url="https://duckduckgo.com/y.js?ad_domain=a"), _row("Real")]
        kept, n = drop_search_ads(rows)
        self.assertEqual([r["title"] for r in kept], ["Real"])
        self.assertEqual(n, 1)


class WebGateTests(_ScorerCase):
    def test_unavailable_reranker_returns_none(self) -> None:
        # conftest hides the installed model; no fake scorer → legacy path.
        self.assertIsNone(gate_rows("what is a heat pump", [_row("Heat pump")], budget=3))

    def test_drops_below_keep_and_orders_by_score(self) -> None:
        self.use({"Heat pump basics": 6.0, "How heat pumps work": 3.0, "Car charger": -4.0})
        rows = [_row("Car charger"), _row("How heat pumps work"), _row("Heat pump basics")]
        gate = gate_rows("what is a heat pump", rows, budget=3, today=_S7_TODAY)
        self.assertEqual([r["title"] for r in gate.kept], ["Heat pump basics", "How heat pumps work"])
        self.assertEqual([r["title"] for r in gate.dropped], ["Car charger"])
        self.assertIs(gate.sufficiency.state, Sufficiency.SUFFICIENT)
        self.assertEqual(gate.kept[0]["_rerank_score"], 6.0)
        self.assertEqual(gate.warnings, ())
        self.assertEqual(gate.diag["survivors"], 2)

    def test_states(self) -> None:
        cases = [
            ({"A": 3.0, "B": 2.5}, Sufficiency.SUFFICIENT),
            ({"A": 3.0, "B": 1.0}, Sufficiency.PARTIAL),
            ({"A": 0.5}, Sufficiency.PARTIAL),
            ({}, Sufficiency.INSUFFICIENT),
        ]
        for scores, state in cases:
            with self.subTest(scores=scores):
                self.use(scores)
                gate = gate_rows("question about things", [_row("A"), _row("B"), _row("C")], budget=3)
                self.assertIs(gate.sufficiency.state, state)

    def test_budget_limits_kept_and_sufficiency_uses_kept(self) -> None:
        self.use({"A": 5.0, "B": 1.0, "C": 3.0})
        gate = gate_rows("question about things", [_row("A"), _row("B"), _row("C")], budget=1)
        self.assertEqual([r["title"] for r in gate.kept], ["A"])
        self.assertEqual(len(gate.survivors), 3)
        self.assertIs(gate.sufficiency.state, Sufficiency.PARTIAL)  # one strong row kept

    def test_thresholds_match_the_scientific_gate(self) -> None:
        from core.knowledge.ranking.sufficiency import SCIENTIFIC_THRESHOLDS

        self.assertEqual(WEB_THRESHOLDS, SCIENTIFIC_THRESHOLDS)


class WebGateRecencyTests(_ScorerCase):
    Q = "What is the latest ECB deposit rate?"

    def _gate(self, rows, budget=3):
        return gate_rows(self.Q, rows, budget=budget, today=_S7_TODAY)

    def test_stale_rows_are_not_strong_and_go_last(self) -> None:
        self.use({"Old": 6.0, "New": 2.5, "Other": 2.2})
        rows = [
            _row("Old", url="https://news.example/2023/02/01/ecb"),
            _row("New", "Rate held in Oct 2026."),
            _row("Other", "Deposit facility explained."),
        ]
        gate = self._gate(rows)
        self.assertEqual([r["title"] for r in gate.kept], ["New", "Other", "Old"])
        self.assertEqual([r["_freshness"] for r in gate.kept], ["fresh", "undated", "stale"])
        self.assertIs(gate.sufficiency.state, Sufficiency.SUFFICIENT)  # New + Other are strong
        self.assertEqual(gate.warnings, (WARNING_RECENCY,))

    def test_only_stale_sources_are_partial_with_warning(self) -> None:
        self.use({"Old A": 6.0, "Old B": 5.0})
        rows = [
            _row("Old A", url="https://news.example/2022/05/01/a"),
            _row("Old B", url="https://news.example/2021/03/b"),
        ]
        gate = self._gate(rows)
        self.assertIs(gate.sufficiency.state, Sufficiency.PARTIAL)
        self.assertIn(WARNING_STALE, gate.warnings)
        self.assertIn("current information", gate.sufficiency.rationale)

    def test_stale_and_undated_without_fresh_is_partial(self) -> None:
        self.use({"Old": 6.0, "Undated": 5.0})
        rows = [_row("Old", url="https://news.example/2022/05/01/a"), _row("Undated", "ECB rates")]
        gate = self._gate(rows)
        self.assertIs(gate.sufficiency.state, Sufficiency.PARTIAL)
        self.assertIn(WARNING_STALE, gate.warnings)

    def test_undated_only_keeps_state_and_is_marked(self) -> None:
        self.use({"A": 6.0, "B": 5.0})
        gate = self._gate([_row("A", "ECB rates"), _row("B", "deposit facility")])
        self.assertIs(gate.sufficiency.state, Sufficiency.SUFFICIENT)
        self.assertEqual(gate.warnings, (WARNING_RECENCY, WARNING_UNDATED))

    def test_no_recency_need_no_freshness(self) -> None:
        self.use({"A": 6.0})
        gate = gate_rows("what is a deposit facility", [_row("A", url="https://x.org/2020/01/a")], budget=3)
        self.assertNotIn("_freshness", gate.kept[0])
        self.assertEqual(gate.warnings, ())


def _web_ctx(profile: str = "fast", query: str = "what is a heat pump") -> RetrievalContext:
    from core.knowledge.retrieval_profiles import get_profile_spec

    spec = get_profile_spec(profile)
    return RetrievalContext(query=query, semantic_query=query, retrieval_profile=spec.id, budget=spec.budget)


def _discovery(rows: list[dict]) -> DiscoveryResult:
    return DiscoveryResult(
        candidates=tuple(
            CandidateUrl(url=r["url"], title=r["title"], snippet=r["snippet"], source="duckduckgo") for r in rows
        ),
        raw_rows=tuple(rows),
        search_outcome=SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, candidate_count=len(rows)),
    )


class GeneralWebPipelineGateTests(_ScorerCase):
    ROWS = [
        _row("Car charger costs", "EV charger installation"),
        _row("Heat pump basics", "A heat pump moves heat."),
        _row("Ad", "Buy now", url="https://duckduckgo.com/y.js?ad_domain=shop.example"),
        _row("How heat pumps work", "Refrigerant cycle."),
    ]

    def _run(self, rows=None, profile="fast"):
        from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline

        with patch("core.knowledge.pipeline_general_web.discover_full", return_value=_discovery(rows or self.ROWS)):
            return run_general_web_evidence_pipeline(_web_ctx(profile))

    def test_gate_orders_drops_and_sets_sufficiency(self) -> None:
        self.use({"Heat pump basics": 6.0, "How heat pumps work": 3.0, "Car charger": -4.0, "Ad": 9.0})
        bundle, diag, _raw = self._run()
        self.assertEqual([s.title for s in bundle.sources], ["Heat pump basics", "How heat pumps work"])
        self.assertEqual(bundle.sufficiency, "sufficient")
        self.assertAlmostEqual(bundle.sources[0].relevance_score, relevance_probability(6.0), places=4)
        self.assertEqual(diag["web_relevance_gate_mode"], "rerank")
        self.assertEqual(diag["web_ads_dropped"], 1)
        self.assertEqual([d["title"] for d in diag["rerank_dropped"]], ["Car charger costs"])
        self.assertNotIn("Ad", [s.title for s in bundle.sources])

    def test_one_strong_source_is_partial_with_capped_confidence(self) -> None:
        self.use({"Heat pump basics": 6.0, "How heat pumps work": 0.5})
        bundle, _diag, _raw = self._run()
        self.assertEqual(bundle.sufficiency, "partial")
        self.assertEqual(bundle.coverage, "poor")
        self.assertLessEqual(bundle.confidence, 0.55)

    def test_everything_dropped_is_insufficient(self) -> None:
        self.use({})
        bundle, _diag, _raw = self._run()
        self.assertEqual(bundle.sources, ())
        self.assertEqual(bundle.sufficiency, "insufficient")
        self.assertEqual(bundle.stop_reason, "no_evidence")
        self.assertEqual(bundle.coverage, "none")

    def test_without_reranker_the_legacy_gate_runs(self) -> None:
        bundle, diag, _raw = self._run()
        self.assertEqual(bundle.sufficiency, "unscored")
        self.assertEqual(diag["web_relevance_gate_mode"], "strict")
        self.assertEqual(diag["rerank"]["skipped_reason"], "model_missing")
        self.assertNotIn("Ad", [s.title for s in bundle.sources])  # ads go on both paths

    def test_recency_turn_marks_stale_snippets(self) -> None:
        from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline

        self.use({"ECB": 6.0})
        rows = [
            _row("ECB holds rates", "Deposit rate unchanged.", url="https://news.example/2021/06/10/ecb"),
            _row("ECB raises rates", "Deposit rate up.", url="https://news.example/2022/09/08/ecb"),
        ]
        ctx = _web_ctx(query="What is the current ECB deposit rate?")
        with patch("core.knowledge.pipeline_general_web.discover_full", return_value=_discovery(rows)):
            bundle, diag, _raw = run_general_web_evidence_pipeline(ctx)
        self.assertEqual(bundle.sufficiency, "partial")
        self.assertIn(WARNING_STALE, bundle.warnings)
        self.assertIn(WARNING_RECENCY, bundle.warnings)
        # URL dates become publication dates; equal scores keep search order.
        self.assertEqual([s.publication_date for s in bundle.sources], ["2021-06-10", "2022-09-08"])
        self.assertEqual({s.raw_metadata["freshness"] for s in bundle.sources}, {"stale"})
        self.assertEqual(diag["rerank"]["recency_window_days"], 365)


_ARTICLE_HTML = """<html><head><title>Heat pumps</title></head><body><article>
<h1>Heat pumps explained</h1><p>A heat pump moves heat from outside air into a home using a
refrigerant cycle. It works in cold weather and can also cool the home in summer. Installers size
the unit to the house.</p><p>Heat pumps use electricity to move heat rather than to make it, so
they deliver more heat energy than the electricity they use.</p></article></body></html>"""


class FetchedSectionGateTests(_ScorerCase):
    def _run(self, scores):
        from core.knowledge.fetch.engine import fetch_html_string
        from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline

        self.use(scores)
        rows = [_row("Heat pump basics", "A heat pump moves heat.", url="https://example.org/heat-pumps")]
        with patch("core.knowledge.pipeline_general_web.discover_full", return_value=_discovery(rows)), patch(
            "core.knowledge.pipeline_general_web.fetch_url",
            return_value=fetch_html_string(_ARTICLE_HTML, url="https://example.org/heat-pumps"),
        ):
            return run_general_web_evidence_pipeline(_web_ctx("balanced"))

    def test_relevant_page_sections_are_scored(self) -> None:
        bundle, diag, _raw = self._run({"Heat pump": 5.0, "heat pump": 5.0})
        self.assertTrue(bundle.sources)
        self.assertEqual(bundle.sources[0].fetch_status, "full_extract")
        self.assertIn(bundle.sufficiency, ("sufficient", "partial"))
        self.assertIn("rerank_sections", diag)
        self.assertIn("rerank_score", bundle.sources[0].raw_metadata)

    def test_irrelevant_page_falls_back_to_gated_snippets(self) -> None:
        # The snippet passes; no fetched section does.
        bundle, diag, _raw = self._run({"A heat pump moves heat.": 5.0})
        self.assertEqual([s.fetch_status for s in bundle.sources], ["snippet_only"])
        self.assertIn("fetched_page_irrelevant", bundle.warnings)
        self.assertEqual(diag["rerank_sections"]["sections_kept"], 0)


def _wiki(title: str, text: str) -> dict:
    return {
        "title": title,
        "snippet": text,
        "full_text": text,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "_wiki_source": True,
    }


class TrustedPipelineGateTests(_ScorerCase):
    def _run(self, wiki_rows, ddg_rows, budget=3):
        from core.knowledge.pipeline_trusted import TrustedEvidencePipeline

        ctx = RetrievalContext(
            query="How many MMR doses do children need?",
            semantic_query="How many MMR doses do children need?",
            budget=RetrievalBudget(max_results=budget),
        )
        with patch("core.knowledge.pipeline_trusted.search_wikipedia", return_value=wiki_rows), patch(
            "core.knowledge.pipeline_trusted.discover_full_with_fallback", return_value=_discovery(ddg_rows)
        ):
            return TrustedEvidencePipeline().run(ctx)

    def test_irrelevant_wikipedia_row_is_dropped(self) -> None:
        self.use({"MMR vaccine": 6.0, "CDC": 5.0, "Polio": -6.0})
        bundle, diag, _raw = self._run(
            [_wiki("Polio vaccine", "Polio vaccines."), _wiki("MMR vaccine", "Two doses of MMR vaccine.")],
            [_row("CDC MMR schedule", "Two doses.", url="https://www.cdc.gov/mmr"), _row("Blog", "x", url="https://blog.example/mmr")],
        )
        titles = [s.title for s in bundle.sources]
        self.assertEqual(titles, ["MMR vaccine", "CDC MMR schedule"])
        self.assertEqual(bundle.sufficiency, "sufficient")
        self.assertIn("Polio vaccine", [d["title"] for d in diag["rerank_dropped"]])

    def test_no_search_when_wikipedia_fills_the_budget(self) -> None:
        from core.knowledge.pipeline_trusted import TrustedEvidencePipeline

        self.use({"MMR": 6.0})
        ctx = RetrievalContext(query="MMR doses", semantic_query="MMR doses", budget=RetrievalBudget(max_results=2))
        with patch(
            "core.knowledge.pipeline_trusted.search_wikipedia",
            return_value=[_wiki("MMR vaccine", "MMR doses."), _wiki("MMR schedule", "MMR doses.")],
        ), patch("core.knowledge.pipeline_trusted.discover_full_with_fallback") as discover:
            bundle, _diag, _raw = TrustedEvidencePipeline().run(ctx)
        discover.assert_not_called()
        self.assertEqual(len(bundle.sources), 2)

    def test_all_irrelevant_is_insufficient(self) -> None:
        self.use({})
        bundle, _diag, _raw = self._run([_wiki("Polio vaccine", "Polio.")], [_row("CDC", "x", url="https://www.cdc.gov/x")])
        self.assertEqual(bundle.sources, ())
        self.assertEqual(bundle.sufficiency, "insufficient")

    def test_without_reranker_wikipedia_is_kept_as_before(self) -> None:
        bundle, _diag, _raw = self._run([_wiki("Polio vaccine", "Polio vaccines.")], [])
        self.assertEqual([s.title for s in bundle.sources], ["Polio vaccine"])
        self.assertEqual(bundle.sufficiency, "unscored")


class FinancePipelineGateTests(_ScorerCase):
    def _run(self, rows, query="Apple latest 10-K annual report"):
        from core.knowledge.pipeline_finance import FinanceEvidencePipeline

        ctx = RetrievalContext(query=query, semantic_query=query, budget=RetrievalBudget(max_results=3))
        with patch(
            "core.knowledge.pipeline_finance._fetch_finance_rows", return_value=(["sec_edgar"], [dict(r) for r in rows])
        ), patch("core.knowledge.pipeline_finance.resolve_service_adapters", return_value=("sec_edgar",)):
            return FinanceEvidencePipeline().run(ctx)

    def test_irrelevant_filing_dropped_even_within_budget(self) -> None:
        self.use({"Apple": 5.0})
        rows = [
            {"title": "10-K — Apple Inc.", "snippet": "Annual report", "url": "https://www.sec.gov/a", "publication_date": "2026-11-03", "_adapter": "sec_edgar"},
            {"title": "10-Q — SentinelOne, Inc.", "snippet": "Quarterly", "url": "https://www.sec.gov/b", "publication_date": "2026-09-01", "_adapter": "sec_edgar"},
        ]
        bundle, diag, _raw = self._run(rows)
        self.assertEqual([s.title for s in bundle.sources], ["10-K — Apple Inc."])
        self.assertEqual(bundle.sufficiency, "partial")
        self.assertIn("not_financial_advice", bundle.warnings)
        self.assertIn(WARNING_RECENCY, bundle.warnings)
        self.assertEqual(bundle.sources[0].raw_metadata.get("rerank_score"), 5.0)

    def test_stale_filing_on_latest_question(self) -> None:
        self.use({"Apple": 5.0})
        rows = [
            {"title": "10-K — Apple Inc.", "snippet": "2019", "url": "https://www.sec.gov/a", "publication_date": "2019-11-01", "_adapter": "sec_edgar"},
            {"title": "10-K — Apple Inc. (2020)", "snippet": "", "url": "https://www.sec.gov/c", "publication_date": "2020-10-30", "_adapter": "sec_edgar"},
        ]
        bundle, _diag, _raw = self._run(rows)
        self.assertEqual(bundle.sufficiency, "partial")
        self.assertIn(WARNING_STALE, bundle.warnings)


class EvidencePackRecencyTests(unittest.TestCase):
    def _prompt(self, ui_sources):
        from workers.llm_worker import LLMWorker

        fake_self = SimpleNamespace(MAX_TOTAL_RETRIEVAL_CHARS=LLMWorker.MAX_TOTAL_RETRIEVAL_CHARS)
        return LLMWorker._format_sources_for_llm_prompt(fake_self, ui_sources)

    def test_ui_rows_carry_freshness_and_prompt_shows_dates(self) -> None:
        from core.knowledge.bundle_builder import build_general_web_bundle
        from core.knowledge.ui_adapter import bundle_to_ui_sources

        rows = [
            {**_row("Fresh", "CPI rose in Oct 2026."), "_rerank_score": 5.0, "_freshness": "fresh",
             "_source_date": "2026-10", "_source_date_basis": "mentioned"},
            {**_row("Undated", "CPI explained."), "_rerank_score": 4.0, "_freshness": "undated",
             "_source_date": None, "_source_date_basis": None},
            {**_row("Old", "CPI.", url="https://news.example/2021/05/02/cpi"), "_rerank_score": 3.0,
             "_freshness": "stale", "_source_date": "2021-05-02", "_source_date_basis": "url"},
        ]
        bundle = build_general_web_bundle(
            query_raw="q", query_resolved="q", kept_rows=rows, rejected_count=0, latency_ms=1.0,
            sufficiency="sufficient", sufficiency_rationale="ok", extra_warnings=(WARNING_RECENCY,),
        )
        ui = bundle_to_ui_sources(bundle)
        self.assertEqual([u["freshness"] for u in ui], ["fresh", "undated", "stale"])
        self.assertNotIn("publication_date", ui[0])  # a mention is not a publication date
        self.assertEqual(ui[2]["publication_date"], "2021-05-02")
        prompt = self._prompt(ui)
        self.assertIn("mentions 2026-10", prompt)
        self.assertIn("date unknown", prompt)
        self.assertIn("2021-05-02 · may be out of date", prompt)

    def test_non_recency_rows_unchanged_in_prompt(self) -> None:
        from core.knowledge.bundle_builder import build_general_web_bundle
        from core.knowledge.ui_adapter import bundle_to_ui_sources

        bundle = build_general_web_bundle(
            query_raw="q", query_resolved="q", kept_rows=[_row("Plain", "Text.")], rejected_count=0, latency_ms=1.0,
        )
        ui = bundle_to_ui_sources(bundle)
        self.assertNotIn("freshness", ui[0])
        prompt = self._prompt(ui)
        self.assertNotIn("date unknown", prompt)
        self.assertNotIn("mentions", prompt)



class EvidenceNoteTests(unittest.TestCase):
    def test_note_order_and_choice(self) -> None:
        from core.memory_filters import (
            RETRIEVAL_PARTIAL_NOTE,
            RETRIEVAL_RECENCY_NOTE,
            RETRIEVAL_STALE_NOTE,
            prefix_evidence_notes,
        )

        body = "--- [1]: Source ---\ntext"
        self.assertEqual(prefix_evidence_notes(body, partial=False), body)
        self.assertEqual(prefix_evidence_notes(body, partial=True), f"{RETRIEVAL_PARTIAL_NOTE}\n\n{body}")
        self.assertEqual(
            prefix_evidence_notes(body, partial=False, bundle_warnings=("recency_requested", "undated_sources")),
            f"{RETRIEVAL_RECENCY_NOTE}\n\n{body}",
        )
        self.assertEqual(
            prefix_evidence_notes(body, partial=True, bundle_warnings=("recency_requested", "stale_sources")),
            f"{RETRIEVAL_STALE_NOTE}\n\n{RETRIEVAL_PARTIAL_NOTE}\n\n{body}",
        )
        # stale_sources without a recency request (never emitted) adds nothing.
        self.assertEqual(prefix_evidence_notes(body, partial=False, bundle_warnings=("stale_sources",)), body)
        self.assertEqual(prefix_evidence_notes("", partial=True, bundle_warnings=("recency_requested",)), "")


if __name__ == "__main__":
    unittest.main()
