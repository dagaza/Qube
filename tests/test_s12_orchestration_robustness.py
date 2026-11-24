"""Plan S12 (R10, R11): orchestration and robustness fixes from the §3.7 audit."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.backoff import (  # noqa: E402
    DDG_RATE_LIMIT_BACKOFF_DEFAULT_SECONDS,
    DDG_RATE_LIMIT_BACKOFF_MAX_SECONDS,
    DDG_RATE_LIMIT_BACKOFF_MIN_SECONDS,
    get_provider_backoff,
    mark_provider_backoff,
    rate_limit_backoff_seconds,
    reset_discovery_backoff,
)
from core.knowledge.discovery.types import DiscoveryResult  # noqa: E402
from core.knowledge.search_outcome import (  # noqa: E402
    BLOCK_CAUSE_PROVIDER_CHALLENGE,
    BLOCK_CAUSE_PROVIDER_RATE_LIMIT,
    BLOCK_CAUSE_QUBE_LIMIT,
    SearchOutcome,
    SearchOutcomeKind,
    build_search_outcome_from_ddg,
    degraded_discovery_note,
    format_search_outcome_summary_line,
)


def _ddg_response(status: int, *, retry_after: str | None = None):
    from core.knowledge.discovery.ddg_http import DdgHttpResponse

    return DdgHttpResponse(status_code=status, text="", transport="test", retry_after=retry_after)


class TestDdgHttpBlocks(unittest.TestCase):
    """DuckDuckGo 429 / 403 used to become a generic network error (status lost)."""

    def _run(self, status: int, retry_after: str | None = None):
        from mcp.internet_tool import execute_internet_search

        with patch("core.knowledge.discovery.pacing.wait_for_ddg_pace_slot", return_value=(True, 0)), patch(
            "core.knowledge.discovery.session_budget.record_ddg_live_request"
        ), patch(
            "core.knowledge.discovery.ddg_http.ddg_html_search",
            return_value=_ddg_response(status, retry_after=retry_after),
        ), patch("core.knowledge.discovery.ddg_http.reset_ddg_browser_session") as reset:
            return execute_internet_search("solar panels", max_results=3), reset

    def test_429_is_rate_limited_with_retry_after(self) -> None:
        from core.knowledge.adapters.duckduckgo import failure_sentinel_reason

        response, reset = self._run(429, retry_after="120")
        self.assertEqual(response.inspection["response_kind"], "rate_limited")
        self.assertEqual(response.inspection["http_status"], 429)
        self.assertEqual(response.inspection["retry_after"], "120")
        self.assertEqual(failure_sentinel_reason(response.rows), "ddg_rate_limited")
        reset.assert_not_called()
        outcome = build_search_outcome_from_ddg(response.rows, response.inspection)
        self.assertEqual(outcome.kind, SearchOutcomeKind.BOT_CHALLENGE)
        self.assertEqual(outcome.http_status, 429)
        self.assertEqual(outcome.failure_sentinel_reason, "ddg_rate_limited")
        self.assertEqual(outcome.block_cause, BLOCK_CAUSE_PROVIDER_RATE_LIMIT)
        self.assertAlmostEqual(outcome.retry_after_s or 0, 120.0, places=1)

    def test_403_is_forbidden_and_resets_session(self) -> None:
        from core.knowledge.adapters.duckduckgo import failure_sentinel_reason

        response, reset = self._run(403)
        self.assertEqual(response.inspection["response_kind"], "forbidden")
        self.assertEqual(failure_sentinel_reason(response.rows), "ddg_forbidden")
        reset.assert_called_once()
        outcome = build_search_outcome_from_ddg(response.rows, response.inspection)
        self.assertEqual(outcome.kind, SearchOutcomeKind.BOT_CHALLENGE)
        self.assertEqual(outcome.block_cause, BLOCK_CAUSE_PROVIDER_CHALLENGE)

    def test_other_errors_still_network_error(self) -> None:
        response, _ = self._run(500)
        outcome = build_search_outcome_from_ddg(response.rows, response.inspection)
        self.assertEqual(outcome.kind, SearchOutcomeKind.NETWORK_ERROR)


class TestBlockCause(unittest.TestCase):
    def test_causes(self) -> None:
        cases = {
            "ddg_pacing_timeout": BLOCK_CAUSE_QUBE_LIMIT,
            "ddg_rate_limited": BLOCK_CAUSE_PROVIDER_RATE_LIMIT,
            "ddg_bot_challenge": BLOCK_CAUSE_PROVIDER_CHALLENGE,
            "ddg_forbidden": BLOCK_CAUSE_PROVIDER_CHALLENGE,
            "ddg_backoff": BLOCK_CAUSE_PROVIDER_CHALLENGE,
        }
        for reason, cause in cases.items():
            with self.subTest(reason=reason):
                outcome = SearchOutcome(
                    kind=SearchOutcomeKind.BOT_CHALLENGE,
                    provider="duckduckgo",
                    failure_sentinel_reason=reason,
                )
                self.assertEqual(outcome.block_cause, cause)
                self.assertEqual(outcome.to_dict().get("block_cause"), cause)

    def test_pacing_timeout_inspection_is_qube_limit(self) -> None:
        from mcp.internet_tool import DDG_PACING_TIMEOUT_SNIPPET

        outcome = build_search_outcome_from_ddg(
            [{"title": "", "snippet": DDG_PACING_TIMEOUT_SNIPPET}],
            {"response_kind": "pacing_timeout", "bot_challenge_signals": ["pacing_timeout"]},
        )
        self.assertEqual(outcome.block_cause, BLOCK_CAUSE_QUBE_LIMIT)
        line = format_search_outcome_summary_line(outcome) or ""
        self.assertIn("cause=Qube's own search limit", line)

    def test_success_has_no_cause(self) -> None:
        outcome = SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, provider="duckduckgo", parsed_rows=3)
        self.assertIsNone(outcome.block_cause)
        self.assertNotIn("block_cause", outcome.to_dict())

    def test_retry_after_round_trips(self) -> None:
        outcome = SearchOutcome(
            kind=SearchOutcomeKind.BOT_CHALLENGE,
            provider="duckduckgo",
            failure_sentinel_reason="ddg_rate_limited",
            retry_after_s=90.0,
        )
        self.assertEqual(SearchOutcome.from_dict(outcome.to_dict()).retry_after_s, 90.0)


class TestPrimaryOutcomeRecording(unittest.TestCase):
    """Qube's own pacing must not pause DuckDuckGo or tighten pacing (§3.7)."""

    def setUp(self) -> None:
        reset_discovery_backoff()

    def tearDown(self) -> None:
        reset_discovery_backoff()

    def _record(self, reason: str, retry_after_s: float | None = None):
        from core.knowledge.discovery.registry import _record_primary_outcome

        result = DiscoveryResult(
            candidates=(),
            raw_rows=(),
            search_outcome=SearchOutcome(
                kind=SearchOutcomeKind.BOT_CHALLENGE,
                provider="duckduckgo",
                failure_sentinel_reason=reason,
                retry_after_s=retry_after_s,
            ),
            provider_id="duckduckgo",
        )
        with patch("core.knowledge.discovery.health.record_ddg_bot_challenge") as challenge:
            _record_primary_outcome("duckduckgo", result)
        return challenge

    def test_pacing_timeout_marks_nothing(self) -> None:
        challenge = self._record("ddg_pacing_timeout")
        challenge.assert_not_called()
        self.assertIsNone(get_provider_backoff("duckduckgo"))

    def test_rate_limit_marks_bounded_backoff_without_challenge(self) -> None:
        challenge = self._record("ddg_rate_limited", retry_after_s=120.0)
        challenge.assert_not_called()
        entry = get_provider_backoff("duckduckgo")
        assert entry is not None
        self.assertEqual(entry.reason, "rate_limited")
        self.assertLessEqual(entry.remaining_seconds, 120)
        self.assertGreater(entry.remaining_seconds, 100)

    def test_bot_challenge_unchanged(self) -> None:
        challenge = self._record("ddg_bot_challenge")
        challenge.assert_called_once()
        entry = get_provider_backoff("duckduckgo")
        assert entry is not None
        self.assertEqual(entry.reason, "bot_challenge")

    def test_rate_limit_backoff_bounds(self) -> None:
        self.assertEqual(rate_limit_backoff_seconds(None), DDG_RATE_LIMIT_BACKOFF_DEFAULT_SECONDS)
        self.assertEqual(rate_limit_backoff_seconds(5), DDG_RATE_LIMIT_BACKOFF_MIN_SECONDS)
        self.assertEqual(rate_limit_backoff_seconds(10**6), DDG_RATE_LIMIT_BACKOFF_MAX_SECONDS)
        self.assertEqual(rate_limit_backoff_seconds(300), 300)

    def test_synthetic_backoff_names_rate_limit(self) -> None:
        from core.knowledge.discovery.registry import _synthetic_backoff_result

        mark_provider_backoff("duckduckgo", reason="rate_limited", ttl_seconds=120)
        outcome = _synthetic_backoff_result("duckduckgo").search_outcome
        assert outcome is not None
        self.assertEqual(outcome.failure_sentinel_reason, "ddg_rate_limited")
        self.assertIn("rate limit", outcome.recovery_hint or "")
        reset_discovery_backoff()
        mark_provider_backoff("duckduckgo", reason="bot_challenge", ttl_seconds=120)
        outcome = _synthetic_backoff_result("duckduckgo").search_outcome
        assert outcome is not None
        self.assertEqual(outcome.failure_sentinel_reason, "ddg_backoff")

    def test_rate_limit_toast_does_not_ask_for_verification(self) -> None:
        from core.notification_types import ddg_backoff_event

        event = ddg_backoff_event(remaining_seconds=300, reason="rate_limited")
        self.assertIn("HTTP 429", event.body)
        self.assertNotIn("verify", event.body.lower())
        self.assertIsNone(event.action_id)
        self.assertIn("verify", ddg_backoff_event(remaining_seconds=300).body.lower())


class TestDegradedDiscoveryNote(unittest.TestCase):
    def test_fallback_note_names_cause(self) -> None:
        outcome = SearchOutcome(
            kind=SearchOutcomeKind.SERP_SUCCESS,
            provider="wikipedia",
            parsed_rows=2,
            fallback_from="duckduckgo",
            fallback_reason="bot_challenge",
            primary_outcome={"block_cause": BLOCK_CAUSE_QUBE_LIMIT},
        )
        note = degraded_discovery_note(outcome)
        self.assertIn("Wikipedia", note)
        self.assertIn("DuckDuckGo", note)
        self.assertIn("Qube's own search limit", note)

    def test_no_note_when_primary_ran(self) -> None:
        outcome = SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, provider="duckduckgo", parsed_rows=3)
        self.assertEqual(degraded_discovery_note(outcome), "")
        self.assertEqual(degraded_discovery_note(None), "")

    def test_blocked_without_fallback(self) -> None:
        outcome = SearchOutcome(
            kind=SearchOutcomeKind.BOT_CHALLENGE,
            provider="duckduckgo",
            failure_sentinel_reason="ddg_rate_limited",
        )
        note = degraded_discovery_note(outcome)
        self.assertIn("did not run", note)
        self.assertIn("provider rate limit", note)


class TestFailedSources(unittest.TestCase):
    def test_hosts_that_never_answered(self) -> None:
        from core.knowledge.http_metrics import HttpMetricsCollector, failed_sources, failed_sources_note

        c = HttpMetricsCollector()
        c.record(host="efts.sec.gov", status_code=503, latency_ms=10)
        c.record(host="efts.sec.gov", status_code=503, latency_ms=10, is_retry=True)
        c.record(host="api.crossref.org", status_code=0, latency_ms=10)
        c.record(host="api.openalex.org", status_code=429, latency_ms=10)
        c.record(host="api.openalex.org", status_code=200, latency_ms=10)
        c.record(host="eutils.ncbi.nlm.nih.gov", status_code=404, latency_ms=10)
        c.record_retry_reason("export.arxiv.org:negative_cache_rate_limited")
        summary = c.snapshot()
        rows = {r["host"]: r["reason"] for r in failed_sources(summary)}
        self.assertEqual(
            rows,
            {"efts.sec.gov": "server_error", "api.crossref.org": "network_error", "export.arxiv.org": "skipped"},
        )
        _, note = failed_sources_note(summary)
        self.assertIn("efts.sec.gov (server error)", note)
        self.assertEqual(failed_sources_note({}), ([], ""))

    def test_merge_keeps_failures(self) -> None:
        from core.knowledge.http_metrics import HttpMetricsCollector, merge_http_summaries

        c = HttpMetricsCollector()
        c.record(host="a.org", status_code=503, latency_ms=1)
        merged = merge_http_summaries([c.snapshot(), c.snapshot()])
        self.assertEqual(merged["by_host"]["a.org"]["failures"], {"server_error": 2})
        self.assertEqual(merged["by_host"]["a.org"]["ok"], 0)


class TestFinanceLegalAdapterFailures(unittest.TestCase):
    """``except Exception: result = []`` made a failed adapter look like no results."""

    def _run(self, module_name: str, pipeline_cls: str, service: str):
        import importlib

        from core.knowledge.types import RetrievalContext

        module = importlib.import_module(f"core.knowledge.{module_name}")

        def boom(*_a, **_k):
            raise RuntimeError("upstream exploded")

        with patch.object(module, "resolve_service_adapters", return_value=("boom_adapter",)), patch.object(
            module, "get_search_function", return_value=boom
        ), patch.object(module, "get_knowledge_source_preferences", return_value={}):
            ctx = RetrievalContext(query="Apple 10-K revenue", semantic_query="Apple 10-K revenue", knowledge_service=service)
            return getattr(module, pipeline_cls)().run(ctx)

    def test_finance(self) -> None:
        bundle, diag, _ = self._run("pipeline_finance", "FinanceEvidencePipeline", "finance_knowledge")
        self.assertEqual(len(bundle.sources), 0)
        self.assertIn("Sources that raised an error: boom_adapter.", bundle.coverage_rationale)
        self.assertEqual(diag["adapter_failures"][0]["adapter"], "boom_adapter")
        self.assertIn("RuntimeError", diag["adapter_failures"][0]["error"])
        self.assertIn("http_summary", diag)

    def test_legal(self) -> None:
        bundle, diag, _ = self._run("pipeline_legal", "LegalEvidencePipeline", "legal_knowledge")
        self.assertIn("Sources that raised an error: boom_adapter.", bundle.coverage_rationale)
        self.assertEqual(diag["adapter_failures"][0]["adapter"], "boom_adapter")


class TestMergeRejectPatterns(unittest.TestCase):
    def test_question_topic_is_never_rejected(self) -> None:
        from core.knowledge.deep_research_merge import extract_merge_reject_title_patterns

        self.assertNotIn("chagas", extract_merge_reject_title_patterns("Chagas disease treatment with benznidazole"))
        self.assertIn("chagas", extract_merge_reject_title_patterns("heart failure drug therapy"))


def _source(title: str, *, sub_score: float | None = None, authority: float = 0.5):
    from core.knowledge.types import EvidenceObject

    meta = {"rerank_score": sub_score} if sub_score is not None else {}
    return EvidenceObject(
        id=title,
        source_id=title,
        adapter="pubmed",
        retrieval_method="api",
        title=title,
        excerpt=title,
        full_text=None,
        url=None,
        document_type="article",
        authority_score=authority,
        raw_metadata=meta,
    )


class TestMergedRerankGate(unittest.TestCase):
    def setUp(self) -> None:
        from core.knowledge.ranking import rerank

        self._rerank = rerank
        # Score = 5 when the text mentions "beta blocker", else -5.
        rerank.set_reranker_scorer_for_tests(
            lambda q, texts: [5.0 if "beta blocker" in t.lower() else -5.0 for t in texts]
        )
        self._enabled = patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True)
        self._enabled.start()

    def tearDown(self) -> None:
        self._enabled.stop()
        self._rerank.set_reranker_scorer_for_tests(None)

    def test_keeps_relevant_and_sub_question_matches_up_to_cap(self) -> None:
        from core.knowledge.deep_research import rerank_merged_sources

        sources = [
            _source("Beta blocker trial A"),
            _source("Unrelated cooking paper"),
            _source("Sub-question match", sub_score=1.5),
            *(_source(f"Beta blocker cohort {i}") for i in range(12)),
        ]
        result = rerank_merged_sources("beta blockers in heart failure", sources, top_k=10)
        assert result is not None
        kept, dropped, diag = result
        titles = [s.title for s in kept]
        self.assertEqual(len(kept), 10)
        self.assertNotIn("Unrelated cooking paper", titles)
        self.assertEqual(dropped, len(sources) - 10)
        self.assertEqual(diag["merged_gate"], "rerank")
        self.assertEqual(diag["merged_rerank_below_keep"], 1)
        full = rerank_merged_sources("beta blockers in heart failure", sources, top_k=24)
        assert full is not None
        self.assertIn("Sub-question match", [s.title for s in full[0]])

    def test_gate_uses_profile_cap_and_falls_back_without_reranker(self) -> None:
        from core.knowledge.deep_research import apply_merged_relevance_gate
        from core.knowledge.types import EvidenceBundle

        bundle = MagicMock(spec=EvidenceBundle)
        bundle.sources = tuple(_source(f"Beta blocker study {i}") for i in range(15))
        with patch.object(self._rerank, "unavailable_reason", return_value="disabled"):
            from core.knowledge import deep_research

            with patch.object(deep_research, "filter_merged_sources_for_query", return_value=([], 0, {})) as v2:
                apply_merged_relevance_gate(query="beta blockers", bundle=bundle, top_k=24)
            self.assertEqual(v2.call_args.kwargs["top_k"], 24)


class TestDeepResearchOrchestration(unittest.TestCase):
    def _bundle(self, sufficiency: str, titles: tuple[str, ...]):
        from core.knowledge.types import EvidenceBundle

        b = MagicMock(spec=EvidenceBundle)
        b.sufficiency = sufficiency
        b.sources = tuple(_source(t) for t in titles)
        return b

    def test_budget_explicit_and_trusted_fallback(self) -> None:
        from core.knowledge import deep_research
        from core.knowledge.deep_research_profiles import get_profile_spec
        from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE, SERVICE_TRUSTED_KNOWLEDGE

        calls: list[dict] = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            out = MagicMock()
            if kwargs["knowledge_service"] == SERVICE_TRUSTED_KNOWLEDGE:
                out.bundle = self._bundle("partial", ("Trusted page",))
            elif "history" in kwargs["query"]:
                out.bundle = self._bundle("insufficient", ())
            else:
                out.bundle = self._bundle("sufficient", ("Scholarly paper",))
            return out

        with patch.object(deep_research, "run_v2_web_retrieval", side_effect=fake_run), patch.object(
            deep_research, "decompose_query", return_value=("drug effect", "drug history")
        ), patch.object(deep_research, "merge_evidence_bundles", return_value=None) as merge:
            result = deep_research.run_deep_research("drug effect and history", profile=get_profile_spec(None))

        spec = get_profile_spec(None)
        self.assertTrue(all(c["budget_is_explicit"] for c in calls))
        self.assertTrue(all(c["budget"] == spec.budget for c in calls))
        services = [c["knowledge_service"] for c in calls]
        self.assertEqual(services.count(SERVICE_TRUSTED_KNOWLEDGE), 1)
        self.assertEqual(services.count(SERVICE_SCIENTIFIC_EVIDENCE), 2)
        bundles = merge.call_args.kwargs["bundles"]
        self.assertEqual(len(bundles), 3)  # both scholarly results kept + the trusted one
        fallbacks = result.diagnostics["sub_query_fallbacks"]
        self.assertEqual([(f["index"], f["sources"]) for f in fallbacks], [(2, 1)])

    def test_report_labels_fallback(self) -> None:
        from core.knowledge.deep_research import build_bibliography_report

        report = build_bibliography_report(
            query="q",
            bundle=None,
            sub_queries=("a", "b"),
            sub_query_fallbacks=({"index": 2, "sources": 3}, {"index": 1, "sources": 0}),
        )
        self.assertIn("Sub-query 2: no relevant scholarly source; 3 result(s) from trusted sources", report)
        self.assertIn("Sub-query 1: no relevant scholarly or trusted source found.", report)


class TestExplicitBudget(unittest.TestCase):
    def test_profile_clamp_skipped_only_when_explicit(self) -> None:
        from core.knowledge import web_retrieval
        from core.knowledge.types import RetrievalBudget

        seen: list[RetrievalBudget] = []

        class FakeService:
            id = "scientific_evidence"

            def default_budget(self):
                return RetrievalBudget()

            def retrieve(self, ctx):
                seen.append(ctx.budget)
                raise RuntimeError("stop")

        big = RetrievalBudget(max_results=8, max_adapter_calls=3, max_latency_ms=45000)
        with patch.object(web_retrieval, "get_knowledge_service", return_value=FakeService()):
            for explicit in (False, True):
                try:
                    web_retrieval.run_v2_web_retrieval(
                        query="q", semantic_query="q", knowledge_service="scientific_evidence", budget=big,
                        retrieval_profile="balanced", budget_is_explicit=explicit,
                    )
                except RuntimeError:
                    pass
        self.assertEqual((seen[0].max_results, seen[0].max_latency_ms), (3, 8000))
        self.assertEqual((seen[1].max_results, seen[1].max_latency_ms), (8, 45000))


class TestTrustedSiteScope(unittest.TestCase):
    def test_trusted_discovery_is_scoped(self) -> None:
        from core.knowledge import pipeline_trusted
        from core.knowledge.discovery.duckduckgo import format_site_bias_query
        from core.knowledge.types import RetrievalContext

        self.assertEqual(
            format_site_bias_query("MMR doses", pipeline_trusted.TRUSTED_DISCOVERY_SITE_SCOPE),
            ("site:wikipedia.org OR site:gov OR site:int OR site:edu MMR doses", None),
        )
        empty = DiscoveryResult(candidates=(), raw_rows=(), search_outcome=None, provider_id="duckduckgo")
        with patch.object(pipeline_trusted, "search_wikipedia", return_value=[]), patch.object(
            pipeline_trusted, "discover_full_with_fallback", return_value=empty
        ) as discover:
            ctx = RetrievalContext(query="MMR doses", semantic_query="MMR doses", knowledge_service="trusted_knowledge")
            pipeline_trusted.TrustedEvidencePipeline().run(ctx)
        self.assertEqual(discover.call_args.kwargs["site_bias"], ("wikipedia.org", "gov", "int", "edu"))


if __name__ == "__main__":
    unittest.main()
