"""Plan S8: adapter deadlines, multi-query union, cache key, Crossref, STS,
web pool widening and the widened trusted allowlist."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapter_collection import (  # noqa: E402
    TIMEOUT_ADAPTER,
    TIMEOUT_DEADLINE,
    VARIANT_PRIMARY,
    VARIANT_RAW,
    AdapterTask,
    collect_adapter_rows,
    rrf_merge,
)
from core.knowledge.evidence_cache import make_cache_key  # noqa: E402
from core.knowledge.ranking.authority import (  # noqa: E402
    is_allowlisted_url,
    is_intergovernmental_url,
    is_legacy_allowlisted_url,
    tier_label_for_url,
)
from core.knowledge.ranking.rerank import set_reranker_scorer_for_tests  # noqa: E402
from core.knowledge.adapter_collection import VARIANT_SPELLING  # noqa: E402
from core.knowledge.spelling import (  # noqa: E402
    conversation_terms,
    correct_word,
    spelling_available,
    spelling_variant,
)
from core.knowledge.retrieval_profiles import get_profile_spec  # noqa: E402
from core.knowledge.types import RetrievalBudget, RetrievalContext  # noqa: E402


def _row(title: str, adapter: str, *, doi: str | None = None) -> dict:
    return {
        "title": title,
        "snippet": title,
        "full_text": title,
        "url": f"https://example.test/{adapter}/{abs(hash(title)) % 10_000}",
        "_adapter": adapter,
        "doi": doi,
        "document_type": "journal_abstract",
    }


def _fns(table: dict):
    return lambda aid: table.get(aid)


class CollectAdapterRowsTests(unittest.TestCase):
    def test_slow_adapter_times_out_without_blocking(self) -> None:
        release = threading.Event()

        def slow(q, max_results=3):
            release.wait(2.0)
            return [_row("late", "slow")]

        def fast(q, max_results=3):
            return [_row(f"fast {i}", "fast") for i in range(max_results)]

        t0 = time.monotonic()
        out = collect_adapter_rows(
            [AdapterTask("slow", VARIANT_PRIMARY, "q"), AdapterTask("fast", VARIANT_PRIMARY, "q")],
            per_adapter=2,
            adapter_timeout_s=0.2,
            deadline_s=5.0,
            search_fn_for=_fns({"slow": slow, "fast": fast}),
        )
        elapsed = time.monotonic() - t0
        release.set()
        self.assertLess(elapsed, 1.0)
        self.assertEqual(out.adapter_calls, ["fast"])
        self.assertEqual([len(lst) for lst in out.ranked_lists], [2])
        self.assertEqual(len(out.timeouts), 1)
        self.assertEqual(out.timeouts[0]["adapter"], "slow")
        self.assertEqual(out.timeouts[0]["reason"], TIMEOUT_ADAPTER)
        self.assertTrue(out.timeouts[0]["started"])
        self.assertEqual(out.timed_out_adapters(), ["slow"])

    def test_overall_deadline_covers_tasks_that_never_started(self) -> None:
        release = threading.Event()

        def slow(q, max_results=3):
            release.wait(2.0)
            return [_row(q, "slow")]

        out = collect_adapter_rows(
            [AdapterTask("slow", VARIANT_PRIMARY, "a"), AdapterTask("slow", VARIANT_RAW, "b")],
            per_adapter=3,
            adapter_timeout_s=None,
            deadline_s=0.2,
            search_fn_for=_fns({"slow": slow}),
            max_workers=1,
        )
        release.set()
        self.assertEqual(out.ranked_lists, [])
        self.assertEqual({t["reason"] for t in out.timeouts}, {TIMEOUT_DEADLINE})
        self.assertEqual(sorted(t["started"] for t in out.timeouts), [False, True])

    def test_exceptions_are_recorded_not_swallowed(self) -> None:
        def boom(q, max_results=3):
            raise RuntimeError("HTTP 503")

        out = collect_adapter_rows(
            [AdapterTask("bad", VARIANT_PRIMARY, "q")],
            per_adapter=3,
            adapter_timeout_s=1.0,
            deadline_s=1.0,
            search_fn_for=_fns({"bad": boom}),
        )
        self.assertEqual(out.adapter_calls, [])
        self.assertEqual(out.errors[0]["adapter"], "bad")
        self.assertIn("HTTP 503", out.errors[0]["error"])
        self.assertEqual(out.timeouts, [])

    def test_unknown_adapters_and_empty_queries_are_skipped(self) -> None:
        out = collect_adapter_rows(
            [AdapterTask("missing", VARIANT_PRIMARY, "q"), AdapterTask("ok", VARIANT_PRIMARY, "  ")],
            per_adapter=3,
            adapter_timeout_s=1.0,
            deadline_s=1.0,
            search_fn_for=_fns({"ok": lambda q, max_results=3: [_row("x", "ok")]}),
        )
        self.assertEqual(out.ranked_lists, [])
        self.assertEqual(out.timings, [])

    def test_rows_carry_their_variant(self) -> None:
        out = collect_adapter_rows(
            [AdapterTask("a", VARIANT_RAW, "q")],
            per_adapter=1,
            adapter_timeout_s=1.0,
            deadline_s=1.0,
            search_fn_for=_fns({"a": lambda q, max_results=3: [_row("t", "a")]}),
        )
        self.assertEqual(out.ranked_lists[0][0]["_query_variant"], VARIANT_RAW)


class RrfMergeTests(unittest.TestCase):
    def test_fuses_ranks_and_dedupes_by_doi_or_title(self) -> None:
        a = [_row("Paper A", "openalex", doi="10.1/a"), _row("Paper B", "openalex")]
        b = [_row("paper   b", "crossref"), _row("Paper C", "crossref"), _row("Other title", "crossref", doi="10.1/A")]
        for r in b:
            r["_query_variant"] = VARIANT_RAW
        merged = rrf_merge([a, b])
        titles = [r["title"] for r in merged]
        # B is found twice (title match), A twice (DOI match, case-insensitive).
        self.assertEqual(titles[:2], ["Paper B", "Paper A"])
        self.assertEqual(titles[2:], ["Paper C"])
        by_title = {r["title"]: r for r in merged}
        self.assertEqual(by_title["Paper B"]["_query_variants"], [VARIANT_PRIMARY, VARIANT_RAW])
        self.assertEqual(by_title["Paper B"]["_adapter"], "openalex")  # first-seen copy kept
        self.assertGreater(by_title["Paper B"]["_rrf_score"], by_title["Paper C"]["_rrf_score"])

    def test_limit(self) -> None:
        rows = [_row(f"T{i}", "x") for i in range(10)]
        self.assertEqual(len(rrf_merge([rows], limit=4)), 4)


class CacheKeyTests(unittest.TestCase):
    def test_previous_keys_are_unchanged(self) -> None:
        raw = "scientific_evidence|statins|openalex,pubmed"
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        key = make_cache_key(
            knowledge_service="scientific_evidence", query="Statins ", adapter_filter=("pubmed", "openalex")
        )
        self.assertEqual(key, expected)

    def test_pool_size_and_variants_change_the_key(self) -> None:
        base = dict(knowledge_service="scientific_evidence", query="statins", adapter_filter=("pubmed",))
        k3 = make_cache_key(**base, pool_size=3)
        self.assertNotEqual(k3, make_cache_key(**base, pool_size=8))
        self.assertNotEqual(k3, make_cache_key(**base, pool_size=3, query_variants=("statin myopathy",)))
        # Variant order and case do not matter.
        self.assertEqual(
            make_cache_key(**base, pool_size=3, query_variants=("B", "a")),
            make_cache_key(**base, pool_size=3, query_variants=("A", "b")),
        )


class ProfileDeadlineTests(unittest.TestCase):
    def test_deadlines_scale_with_profile(self) -> None:
        fast, bal, thorough = (get_profile_spec(p) for p in ("fast", "balanced", "thorough"))
        self.assertLess(fast.collection_deadline_s, bal.collection_deadline_s)
        self.assertLess(bal.collection_deadline_s, thorough.collection_deadline_s)
        for spec in (fast, bal, thorough):
            self.assertLessEqual(spec.adapter_timeout_s, spec.collection_deadline_s)
            self.assertLess(spec.collection_deadline_s * 1000, spec.budget.max_latency_ms)


def _scorer(question, texts):
    return [5.0 if "relevant" in t.lower() else -5.0 for t in texts]


class ScientificPipelineS8Tests(unittest.TestCase):
    def setUp(self) -> None:
        set_reranker_scorer_for_tests(_scorer)
        self.calls: list[tuple[str, str]] = []
        # The raw-wording variant needs an OpenAlex key (D8); on by default here.
        self._key = patch("core.knowledge.pipeline_scientific._openalex_key_configured", return_value=True)
        self._key.start()

    def tearDown(self) -> None:
        self._key.stop()
        set_reranker_scorer_for_tests(None)

    def _search(self, aid: str, *, delay: float = 0.0, release: threading.Event | None = None):
        def fn(q: str, max_results: int = 3) -> list[dict]:
            self.calls.append((aid, q))
            if release is not None:
                release.wait(delay)
            return [_row(f"{aid} relevant {q} {i}", aid) for i in range(max_results)]

        return fn

    def _run(self, *, adapters, table, query, query_raw=None, profile=None, conversation_terms=frozenset()):
        from core.knowledge.pipeline_scientific import ScientificEvidencePipeline

        spec = profile or get_profile_spec("balanced")
        with patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters", return_value=adapters
        ), patch("core.knowledge.adapters.registry.SEARCH_FUNCTIONS", table), patch(
            "core.knowledge.pipeline_scientific.get_profile_spec", return_value=spec
        ), patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None), patch(
            "core.knowledge.pipeline_scientific.set_cached_rows"
        ) as set_cache:
            bundle, diag, raw = ScientificEvidencePipeline().run(
                RetrievalContext(
                    query=query,
                    semantic_query=query,
                    budget=RetrievalBudget(max_results=3),
                    query_raw=query_raw,
                    conversation_terms=conversation_terms,
                )
            )
        return bundle, diag, set_cache

    def test_raw_wording_is_searched_on_variant_adapters_only(self) -> None:
        table = {aid: self._search(aid) for aid in ("openalex", "crossref", "arxiv")}
        resolved = "How does actor-network theory analyze artificial intelligence?"
        _bundle, diag, _ = self._run(
            adapters=("openalex", "crossref", "arxiv"),
            table=table,
            query=resolved,
            query_raw="How does ANT analyze AI?",
        )
        raw_calls = [aid for aid, q in self.calls if "ANT" in q]
        # Never arXiv (3.5 s interval) or Crossref (concurrency 429s, S8 live run).
        self.assertEqual(raw_calls, ["openalex"])
        variants = [v["id"] for v in diag["scientific_pool"]["variants"]]
        self.assertEqual(variants, [VARIANT_PRIMARY, VARIANT_RAW])
        self.assertEqual(diag["scientific_pool"]["per_adapter"], 3)  # depth unchanged (S8 measured)

    def test_no_raw_variant_without_rewrite_or_reranker(self) -> None:
        table = {"openalex": self._search("openalex")}
        q = "statins muscle pain in older adults"
        self._run(adapters=("openalex",), table=table, query=q, query_raw=None)
        self._run(adapters=("openalex",), table=table, query=q, query_raw=q.upper())  # same wording
        self.assertEqual(len(self.calls), 2)
        set_reranker_scorer_for_tests(None)  # legacy path: no union (conftest hides the model)
        self.calls.clear()
        self._run(adapters=("openalex",), table=table, query=q, query_raw="what about their side effects")
        self.assertEqual(len(self.calls), 1)

    def test_no_raw_variant_without_an_openalex_key(self) -> None:
        table = {"openalex": self._search("openalex")}
        with patch("core.knowledge.pipeline_scientific._openalex_key_configured", return_value=False):
            _b, diag, _ = self._run(
                adapters=("openalex",),
                table=table,
                query="How does actor-network theory analyze artificial intelligence?",
                query_raw="How does ANT analyze AI?",
            )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([v["id"] for v in diag["scientific_pool"]["variants"]], [VARIANT_PRIMARY])

    @unittest.skipUnless(spelling_available(), "pyspellchecker not installed")
    def test_misspelt_question_adds_a_correctly_spelt_search(self) -> None:
        table = {aid: self._search(aid) for aid in ("openalex", "crossref")}
        q = "How does actor-network theory analyze Artificial Inteligence?"
        _b, diag, _ = self._run(adapters=("openalex", "crossref"), table=table, query=q)
        # The user's wording is always searched; the corrected wording only on OpenAlex.
        searched = q.rstrip("?")  # the adapter query sanitizer drops the "?"
        self.assertIn(("openalex", searched), self.calls)
        self.assertIn(("crossref", searched), self.calls)
        corrected = [(aid, qq) for aid, qq in self.calls if "Intelligence" in qq]
        self.assertEqual(corrected, [("openalex", "How does actor-network theory analyze Artificial Intelligence")])
        pool = diag["scientific_pool"]
        self.assertEqual([v["id"] for v in pool["variants"]], [VARIANT_PRIMARY, VARIANT_SPELLING])
        self.assertEqual(pool["spelling_corrections"][0]["original"], "Inteligence")
        self.assertEqual(pool["spelling_corrections"][0]["corrected"], "Intelligence")

    @unittest.skipUnless(spelling_available(), "pyspellchecker not installed")
    def test_statins_is_never_turned_into_stations(self) -> None:
        table = {"openalex": self._search("openalex")}
        for q in ("statins muscle pain in older adults", "Do statin drugs cause muscle pain?"):
            with self.subTest(q=q):
                self.calls.clear()
                _b, diag, _ = self._run(adapters=("openalex",), table=table, query=q)
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(diag["scientific_pool"]["spelling_corrections"], [])

    @unittest.skipUnless(spelling_available(), "pyspellchecker not installed")
    def test_assistant_words_are_not_corrected(self) -> None:
        table = {"openalex": self._search("openalex")}
        self._run(
            adapters=("openalex",),
            table=table,
            query="what does inteligence mean here",
            conversation_terms=frozenset({"inteligence"}),
        )
        self.assertEqual(len(self.calls), 1)

    @unittest.skipUnless(spelling_available(), "pyspellchecker not installed")
    def test_no_spelling_variant_on_the_legacy_path(self) -> None:
        set_reranker_scorer_for_tests(None)  # conftest hides the model
        table = {"openalex": self._search("openalex")}
        self._run(adapters=("openalex",), table=table, query="Artificial Inteligence and labour")
        self.assertEqual(len(self.calls), 1)

    def test_content_free_raw_wording_is_not_searched(self) -> None:
        table = {"openalex": self._search("openalex")}
        self._run(
            adapters=("openalex",), table=table, query="statins side effects", query_raw="what about it?"
        )
        self.assertEqual(len(self.calls), 1)

    def test_adapter_timeout_reaches_trace_rationale_and_warnings(self) -> None:
        release = threading.Event()
        table = {
            "openalex": self._search("openalex"),
            "europe_pmc": self._search("europe_pmc", delay=3.0, release=release),
        }
        spec = dataclasses.replace(get_profile_spec("balanced"), adapter_timeout_s=0.2, collection_deadline_s=1.0)
        t0 = time.monotonic()
        bundle, diag, set_cache = self._run(
            adapters=("openalex", "europe_pmc"), table=table, query="statins side effects", profile=spec
        )
        release.set()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual([t["adapter"] for t in diag["adapter_timeouts"]], ["europe_pmc"])
        self.assertIn("adapter_timeout", bundle.warnings)
        self.assertIn("did not answer in time: europe_pmc", bundle.coverage_rationale)
        self.assertTrue(bundle.sources)
        set_cache.assert_not_called()  # a pool missing late adapters is not cached

    def test_complete_pool_is_cached(self) -> None:
        table = {"openalex": self._search("openalex")}
        _b, diag, set_cache = self._run(adapters=("openalex",), table=table, query="statins side effects")
        set_cache.assert_called_once()
        self.assertNotIn("adapter_timeouts", diag)

    def test_service_passes_profile_and_raw_query(self) -> None:
        from core.knowledge.services.scientific_evidence import ScientificEvidenceService

        seen = {}

        def fake_run(self_, ctx):
            seen["profile"], seen["raw"] = ctx.retrieval_profile, ctx.query_raw
            raise StopIteration

        with patch("core.knowledge.pipeline_scientific.ScientificEvidencePipeline.run", fake_run):
            with self.assertRaises(StopIteration):
                ScientificEvidenceService().retrieve(
                    RetrievalContext(query="q", semantic_query="q", retrieval_profile="thorough", query_raw="raw q")
                )
        self.assertEqual(seen, {"profile": "thorough", "raw": "raw q"})


@unittest.skipUnless(spelling_available(), "pyspellchecker not installed")
class SpellingVariantTests(unittest.TestCase):
    def test_corrects_confident_misspellings_only(self) -> None:
        cases = {
            "Yes, please tell me how ANT analyzes Artificial Inteligence for example.":
                "Yes, please tell me how ANT analyzes Artificial Intelligence for example.",
            "What is the evidence for nueral netwrok pruning?": "What is the evidence for neural network pruning?",
            "polution and asthma in children": "pollution and asthma in children",  # not "solution"
            "microbiome and depresion": "microbiome and depression",
            "psycology of habit formation": "psychology of habit formation",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                variant = spelling_variant(text)
                self.assertIsNotNone(variant)
                self.assertEqual(variant.text, expected)

    def test_leaves_domain_terms_names_and_acronyms_alone(self) -> None:
        for text in (
            "statins side effects in older adults",  # stations / stains / stating are all one edit away
            "Do statin drugs cause muscle pain?",  # drug stem
            "atorvastatin dosing",
            "covid vaccine efficacy",
            "mRNA vaccine durability",
            "How does the OAuth 2.0 authorization code flow work",
            "CRISPR off-target effects",
            "effects of semaglutide on weight",
            "Latour laboratory life",
            "Zorblax 9000 smart toaster firmware",
            "zelvoraxine migraine trials",
            "diabetis prevention trials",  # diabetes / diabetic: ambiguous
            "What is the effect of social media use on adolescent depression?",
            "",
        ):
            with self.subTest(text=text):
                self.assertIsNone(spelling_variant(text))

    def test_correction_keeps_capitalisation_and_reports_evidence(self) -> None:
        fix = correct_word("Inteligence")
        self.assertEqual(fix.corrected, "Intelligence")
        self.assertGreaterEqual(fix.candidate_count, 200)
        self.assertIsNone(correct_word("intelligence"))  # known word

    def test_protected_terms(self) -> None:
        self.assertIsNone(spelling_variant("Artificial Inteligence", protected_terms={"Inteligence"}))

    def test_conversation_terms_come_from_assistant_messages(self) -> None:
        history = [
            {"role": "user", "content": "Tell me about Inteligence"},
            {"role": "assistant", "content": "Actor-network theory (ANT), after Latour and Callon."},
        ]
        terms = conversation_terms(history)
        self.assertIn("latour", terms)
        self.assertIn("callon", terms)
        self.assertNotIn("inteligence", terms)  # the user's own misspelling is not protected


class CrossrefAndRoutingTests(unittest.TestCase):
    def test_crossref_review_status_follows_type(self) -> None:
        from core.knowledge.adapters.crossref import _row_from_item

        base = {"title": ["T"], "abstract": "A", "DOI": "10.1/x"}
        article = _row_from_item({**base, "type": "journal-article"})
        preprint = _row_from_item({**base, "type": "posted-content"})
        chapter = _row_from_item({**base, "type": "book-chapter"})
        self.assertTrue(article["peer_reviewed"])
        self.assertFalse(article["preprint"])
        self.assertTrue(preprint["preprint"])
        self.assertFalse(preprint["peer_reviewed"])
        self.assertEqual(preprint["document_type"], "preprint")
        self.assertFalse(chapter["peer_reviewed"])

    def test_crossref_joins_general_and_social_science(self) -> None:
        from core.knowledge.source_preferences import resolve_service_adapters
        from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

        for q in ("How do ants build colonies?", "income inequality and social mobility"):
            with self.subTest(q=q):
                ids = resolve_service_adapters(SERVICE_SCIENTIFIC_EVIDENCE, query=q, stored_preferences={})
                self.assertIn("crossref", ids)
                self.assertNotIn("semantic_scholar", ids)  # keyless users get nothing from it (V8)

    def test_sts_vocabulary_routes_to_social_science(self) -> None:
        from core.knowledge.scientific_discipline import detect_scientific_discipline

        cases = {
            "How does actor-network theory analyze artificial intelligence?": "sociology",
            "actor-network theory and machine learning algorithms": "sociology",
            "What is the social construction of technology?": "sociology",
            "Latour laboratory life": "sociology",
            "transformer attention mechanism": "computer_science",
            "How do ants build colonies?": "general_science",
        }
        for q, discipline in cases.items():
            with self.subTest(q=q):
                self.assertEqual(detect_scientific_discipline(q).discipline, discipline)


class TrustedAllowlistTests(unittest.TestCase):
    def test_widened_allowlist(self) -> None:
        allowed = [
            "https://www.who.int/publications/i/item/9789240034228",
            "https://iris.who.int/handle/1",
            "https://www.ecb.europa.eu/stats",
            "https://www.oecd.org/en/topics.html",
            "https://www.un.org/en/",
            "https://www.unicef.org/reports",
            "https://www.worldbank.org/en/topic",
            "https://www.abs.gov.au/statistics",
            "https://www.canada.ca/en/health-canada.html",
            "https://www.service-public.gouv.fr/",
            "https://www.mhlw.go.jp/",
            "https://www.nhs.uk/conditions/",
            "https://www.insee.fr/fr/statistiques",
            "https://www.rki.de/DE/Home/homepage_node.html",
            "https://www.u-tokyo.ac.jp/en/",
            "https://www.unimelb.edu.au/",
            # pre-S8 list still allowed
            "https://en.wikipedia.org/wiki/Air_pollution",
            "https://www.cdc.gov/vaccines/",
            "https://www.ons.gov.uk/",
            "https://mit.edu/",
            "https://www.ox.ac.uk/",
        ]
        for url in allowed:
            with self.subTest(url=url):
                self.assertTrue(is_allowlisted_url(url))
        refused = [
            "https://example.com/who",
            "https://gov.example.com/",
            "https://notgov.co/",
            "https://fakeinsee.fr/",
            "https://insee.fr.example.com/",
            "https://who-int.example.com/",
            "https://www.oecd-ilibrary.example/",
            "",
        ]
        for url in refused:
            with self.subTest(url=url):
                self.assertFalse(is_allowlisted_url(url))

    def test_legacy_list_is_the_pre_s8_list(self) -> None:
        self.assertFalse(is_legacy_allowlisted_url("https://www.who.int/x"))
        self.assertFalse(is_allowlisted_url("https://www.who.int/x", widened=False))
        for url in ("https://en.wikipedia.org/wiki/X", "https://www.cdc.gov/", "https://www.ons.gov.uk/",
                    "https://mit.edu/", "https://www.ox.ac.uk/"):
            with self.subTest(url=url):
                self.assertTrue(is_legacy_allowlisted_url(url))

    def test_tier_labels(self) -> None:
        self.assertTrue(is_intergovernmental_url("https://www.who.int/"))
        self.assertEqual(tier_label_for_url("https://www.who.int/"), "intergovernmental")
        self.assertEqual(tier_label_for_url("https://www.abs.gov.au/"), "government")
        self.assertEqual(tier_label_for_url("https://www.u-tokyo.ac.jp/"), "academic")


class WebPoolWideningTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def _ctx(self):
        return RetrievalContext(
            query="what is a heat pump",
            semantic_query="what is a heat pump",
            knowledge_service="general_web",
            budget=RetrievalBudget(max_results=3),
            retrieval_profile="fast",  # snippets only, no fetch
        )

    def _rows(self, n):
        return [
            {"title": f"Heat pump guide {i}" if i % 2 == 0 else f"Unrelated {i}", "snippet": "s",
             "url": f"https://site{i}.example/page"}
            for i in range(n)
        ]

    def _run(self, rows):
        from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult
        from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind
        from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline

        seen = {}

        def fake_discover(query, *, max_results, **kw):
            seen["max_results"] = max_results
            picked = rows[:max_results]
            return DiscoveryResult(
                candidates=tuple(
                    CandidateUrl(url=r["url"], title=r["title"], snippet=r["snippet"], source="duckduckgo")
                    for r in picked
                ),
                raw_rows=tuple(picked),
                search_outcome=SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, candidate_count=len(picked)),
            )

        with patch("core.knowledge.pipeline_general_web.discover_full", side_effect=fake_discover):
            bundle, diag, _raw = run_general_web_evidence_pipeline(self._ctx())
        return bundle, diag, seen

    def test_gated_turn_reads_eight_rows_and_presents_budget(self) -> None:
        set_reranker_scorer_for_tests(lambda q, texts: [6.0 if "heat pump" in t.lower() else -6.0 for t in texts])
        bundle, diag, seen = self._run(self._rows(10))
        self.assertEqual(seen["max_results"], 8)
        self.assertEqual(diag["web_results_raw_count"], 8)
        self.assertEqual(len(bundle.sources), 3)  # budget, not every survivor
        self.assertEqual(diag["rerank"]["survivors"], 4)
        self.assertEqual(bundle.sufficiency, "sufficient")

    def test_serp_order_breaks_near_ties_but_not_clear_gaps(self) -> None:
        from core.knowledge.ranking.web_gate import SERP_RANK_PRIOR, WEB_THRESHOLDS, gate_rows

        scores = {"a": 4.0, "b": 4.5, "c": 9.0, "d": 1.0}  # SERP order a, b, c, d
        set_reranker_scorer_for_tests(lambda q, texts: [scores[t.split()[0]] for t in texts])
        rows = [{"title": f"{k} heat pump", "snippet": "", "url": f"https://{k}.example/"} for k in scores]
        plain = gate_rows("what is a heat pump", [dict(r) for r in rows], budget=3, thresholds=WEB_THRESHOLDS)
        prior = gate_rows(
            "what is a heat pump", [dict(r) for r in rows], budget=3, thresholds=WEB_THRESHOLDS,
            rank_prior=SERP_RANK_PRIOR,
        )
        self.assertEqual([r["title"][0] for r in plain.kept], ["c", "b", "a"])
        # a (rank 0) beats b (rank 1, +0.5 only); c's +4.5 lead survives two ranks.
        self.assertEqual([r["title"][0] for r in prior.kept], ["c", "a", "b"])
        # Thresholds and state use the raw score: d (1.0 < keep 3) is dropped either way.
        self.assertEqual([r["title"][0] for r in prior.dropped], ["d"])
        self.assertEqual(prior.sufficiency.state, plain.sufficiency.state)
        self.assertEqual(prior.sufficiency.strong_count, plain.sufficiency.strong_count)

    def test_pipeline_orders_kept_rows_with_the_serp_prior(self) -> None:
        scores = {"a": 4.0, "b": 4.5, "c": 9.0, "d": 1.0}
        set_reranker_scorer_for_tests(lambda q, texts: [scores[t.split()[0]] for t in texts])
        rows = [{"title": f"{k} heat pump", "snippet": "s", "url": f"https://{k}.example/"} for k in scores]
        bundle, _diag, _seen = self._run(rows)
        self.assertEqual([src.title[0] for src in bundle.sources], ["c", "a", "b"])

    def test_legacy_turn_keeps_the_old_pool(self) -> None:
        _bundle, diag, seen = self._run(self._rows(10))  # conftest hides the model
        self.assertEqual(seen["max_results"], 3)
        self.assertEqual(diag["rerank"]["skipped_reason"], "model_missing")


if __name__ == "__main__":
    unittest.main()
