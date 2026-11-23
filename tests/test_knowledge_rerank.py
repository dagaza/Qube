"""Unit tests for the cross-encoder rerank module (plan S3).

The real model is exercised only in ``RerankModelIntegrationTests`` (skipped
when the reranker is not downloaded); everything else injects a fake scorer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.ranking import rerank  # noqa: E402
from core.knowledge.ranking.rerank import (  # noqa: E402
    SKIP_DISABLED,
    SKIP_EMPTY,
    SKIP_MODEL_MISSING,
    SKIP_NON_ENGLISH,
    is_probably_english,
    rerank_row_text,
    rerank_rows,
    rerank_texts,
    reranker_threads,
    set_reranker_scorer_for_tests,
    unavailable_reason,
)


def _len_scorer(question: str, docs: list[str]) -> list[float]:
    return [float(len(d)) for d in docs]


class FakeScorerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_reranker_scorer_for_tests(_len_scorer)

    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)


class RowTextTests(unittest.TestCase):
    def test_title_plus_body_truncated(self) -> None:
        row = {"title": "Heat  pumps", "full_text": "x" * 2000, "snippet": "ignored"}
        text = rerank_row_text(row, text_chars=100)
        self.assertTrue(text.startswith("Heat pumps. xxx"))
        self.assertEqual(len(text), 100)

    def test_falls_back_to_snippet_then_title(self) -> None:
        self.assertEqual(rerank_row_text({"title": "T", "snippet": "s"}), "T. s")
        self.assertEqual(rerank_row_text({"title": "Only title"}), "Only title")
        self.assertEqual(rerank_row_text({"snippet": "only body"}), "only body")


class ScoringContractTests(FakeScorerTestCase):
    def test_scores_preserve_input_order(self) -> None:
        res = rerank_texts("what is a heat pump", ["aaa", "a", "aa"])
        self.assertIsNotNone(res)
        self.assertEqual(res.scores, (3.0, 1.0, 2.0))
        self.assertEqual(res.model_id, rerank.RERANKER_MODEL_ID)
        self.assertGreaterEqual(res.latency_ms, 0.0)

    def test_rows_capped_at_max_rows(self) -> None:
        rows = [{"title": f"t{i}"} for i in range(10)]
        res = rerank_rows("what is a heat pump", rows, max_rows=4)
        self.assertEqual(len(res.scores), 4)

    def test_rows_text_chars_applied(self) -> None:
        res = rerank_rows("q about pumps", [{"title": "T", "full_text": "y" * 5000}], text_chars=50)
        self.assertEqual(res.scores, (50.0,))
        self.assertEqual(res.text_chars, 50)

    def test_empty_inputs_return_none(self) -> None:
        self.assertIsNone(rerank_texts("what is it", []))
        self.assertIsNone(rerank_texts("   ", ["doc"]))
        self.assertEqual(unavailable_reason("  "), SKIP_EMPTY)

    def test_non_dict_rows_return_none(self) -> None:
        self.assertIsNone(rerank_rows("how does it work", [{"title": "a"}, "oops"]))  # type: ignore[list-item]

    def test_scorer_exception_is_swallowed(self) -> None:
        def boom(_q: str, _d: list[str]) -> list[float]:
            raise RuntimeError("onnx exploded")

        set_reranker_scorer_for_tests(boom)
        self.assertIsNone(rerank_texts("what is a heat pump", ["doc"]))

    def test_wrong_score_count_rejected(self) -> None:
        set_reranker_scorer_for_tests(lambda _q, _d: [1.0])
        self.assertIsNone(rerank_texts("what is a heat pump", ["a", "b"]))

    def test_non_english_query_skipped(self) -> None:
        self.assertIsNone(rerank_texts("cum funcționează o pompă de căldură", ["doc"]))
        self.assertEqual(unavailable_reason("wie funktioniert eine Wärmepumpe"), SKIP_NON_ENGLISH)


class AvailabilityTests(unittest.TestCase):
    def test_disabled_setting(self) -> None:
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False):
            self.assertEqual(unavailable_reason(), SKIP_DISABLED)
            self.assertIsNone(rerank_texts("what is a heat pump", ["doc"]))

    def test_model_missing(self) -> None:
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True), patch(
            "core.bootstrap_search_download.resolve_reranker_path", return_value=None
        ):
            self.assertEqual(unavailable_reason(), SKIP_MODEL_MISSING)
            self.assertFalse(rerank.reranker_available())
            self.assertFalse(rerank.warm_up_reranker())
            self.assertIsNone(rerank_texts("what is a heat pump", ["doc"]))

    def test_load_failure_is_swallowed_and_backed_off(self) -> None:
        rerank.unload_reranker()
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True), patch(
            "core.bootstrap_search_download.resolve_reranker_path", return_value="/nonexistent"
        ), patch(
            "fastembed.rerank.cross_encoder.TextCrossEncoder", side_effect=RuntimeError("bad onnx")
        ):
            self.assertIsNone(rerank_texts("what is a heat pump", ["doc"]))
            self.assertEqual(unavailable_reason(), rerank.SKIP_LOAD_FAILED)
        rerank.unload_reranker()

    def test_thread_cap(self) -> None:
        self.assertGreaterEqual(reranker_threads(), 1)
        self.assertLessEqual(reranker_threads(), 4)


class LanguageHeuristicTests(unittest.TestCase):
    def test_english_queries(self) -> None:
        for q in (
            "How does actor-network theory analyze artificial intelligence?",
            "Bled Castle opening hours",
            "who designed the sydney opera house",
            "Jørn Utzon",
            "GDPR right to erasure exceptions",
            "why do plants die in winter",
            "café near the Louvre",
        ):
            with self.subTest(q=q):
                self.assertTrue(is_probably_english(q))

    def test_non_english_queries(self) -> None:
        for q in (
            "cum funcționează o pompă de căldură",
            "wie funktioniert eine Wärmepumpe",
            "comment fonctionne une pompe à chaleur",
            "cómo funciona una bomba de calor",
            "как работает тепловой насос",
            "热泵是如何工作的",
        ):
            with self.subTest(q=q):
                self.assertFalse(is_probably_english(q))


class ContentGuardTests(FakeScorerTestCase):
    def test_content_free_turns_are_skipped(self) -> None:
        for q in ("Yes, please.", "tell me more", "ok, go ahead", "Yes please, another example"):
            with self.subTest(q=q):
                self.assertFalse(rerank.has_retrieval_content(q))
                self.assertEqual(unavailable_reason(q), rerank.SKIP_NO_CONTENT)
                self.assertIsNone(rerank_texts(q, ["doc"]))

    def test_questions_with_content_are_scored(self) -> None:
        for q in ("Yes, please tell me how ANT analyzes AI", "and what about 3.14?", "heat pump cost"):
            with self.subTest(q=q):
                self.assertTrue(rerank.has_retrieval_content(q))
                self.assertIsNotNone(rerank_texts(q, ["doc"]))


class DownloadSpecTests(unittest.TestCase):
    def test_reranker_spec_and_paths(self) -> None:
        from core.bootstrap_search_download import (
            RERANKER_DOWNLOAD_SPEC,
            RERANKER_MODEL_ID,
            reranker_complete,
            reranker_dir,
        )

        self.assertEqual(RERANKER_MODEL_ID, rerank.RERANKER_MODEL_ID)
        self.assertIn("onnx/model.onnx", RERANKER_DOWNLOAD_SPEC.files)
        self.assertEqual(reranker_dir().name, "reranker")
        with patch("core.bootstrap_search_download.reranker_dir", return_value=Path("/nonexistent/rr")):
            self.assertFalse(reranker_complete())

    def test_download_streams_each_file(self) -> None:
        import tempfile

        from core.bootstrap_search_download import RERANKER_DOWNLOAD_SPEC, download_reranker

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "reranker"

            def _touch(**kwargs: object) -> None:
                dest = kwargs["dest_path"]
                assert isinstance(dest, Path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"x")

            events: list[int] = []
            with patch("core.bootstrap_search_download.reranker_dir", return_value=base), patch(
                "core.bootstrap_download._download_hf_hub_file", side_effect=_touch
            ) as stream:
                download_reranker(lambda _s, _f, pct, _src: events.append(pct))
            self.assertEqual(stream.call_count, len(RERANKER_DOWNLOAD_SPEC.files))
            self.assertEqual(
                {c.kwargs["repo_id"] for c in stream.call_args_list}, {"Xenova/ms-marco-MiniLM-L-6-v2"}
            )
            self.assertEqual(events[-1], 100)

    def test_download_failure_raises_runtime_error(self) -> None:
        import tempfile

        from core.bootstrap_search_download import download_reranker

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.bootstrap_search_download.reranker_dir", return_value=Path(tmp) / "rr"
        ), patch("core.bootstrap_download._download_hf_hub_file", side_effect=OSError("net down")):
            with self.assertRaises(RuntimeError):
                download_reranker(lambda *_a: None)


class ShadowRerankTests(FakeScorerTestCase):
    POOL = [
        {"title": "aa", "url": "https://x/1"},
        {"title": "aaaa", "url": "https://x/2"},
        {"title": "a", "url": "https://x/3"},
    ]

    def test_selection_unchanged_and_scores_attached(self) -> None:
        set_reranker_scorer_for_tests(lambda _q, docs: [1.0 if len(d) > 2 else -5.0 for d in docs])
        kept_in = [self.POOL[2], self.POOL[0]]
        kept, diag = rerank.shadow_rerank("what is it about pumps", self.POOL, kept_in, budget=3)
        self.assertEqual([r["url"] for r in kept], ["https://x/3", "https://x/1"])
        self.assertTrue(all("_rerank_score" in r for r in kept))
        self.assertNotIn("_rerank_score", self.POOL[0])  # copies; cached rows untouched
        self.assertEqual(diag["would_keep_count"], 1)
        self.assertEqual(diag["would_state"], "partial")
        self.assertEqual(diag["kept_below_threshold"], 2)
        self.assertEqual(diag["mode"], "shadow")

    def test_cold_model_skips_and_starts_warmup(self) -> None:
        set_reranker_scorer_for_tests(None)
        rerank.unload_reranker()
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True), patch(
            "core.bootstrap_search_download.resolve_reranker_path", return_value="/some/path"
        ), patch.object(rerank, "warm_up_reranker_async", return_value=True) as warm:
            kept, diag = rerank.shadow_rerank("what is a heat pump", self.POOL, self.POOL[:1], budget=3)
        self.assertEqual(diag["skipped_reason"], rerank.SKIP_COLD)
        warm.assert_called_once()
        self.assertEqual(kept, [dict(self.POOL[0])])

    def test_skipped_when_unavailable(self) -> None:
        set_reranker_scorer_for_tests(None)
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False):
            kept, diag = rerank.shadow_rerank("what is a heat pump", self.POOL, self.POOL, budget=3)
        self.assertEqual(diag["skipped_reason"], SKIP_DISABLED)
        self.assertEqual(len(kept), 3)


_SCI_ROWS = [
    {
        "title": "Semaglutide and cardiovascular outcomes",
        "snippet": "Semaglutide reduced major adverse cardiovascular events.",
        "full_text": "Semaglutide reduced major adverse cardiovascular events in adults with type 2 diabetes.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "_adapter": "pubmed",
        "doi": "10.1000/test.1",
        "publication_date": "2024-01",
        "peer_reviewed": True,
    },
    {
        "title": "Semaglutide subgroup analysis",
        "snippet": "No significant benefit for semaglutide in this cohort.",
        "full_text": "No significant benefit was observed for semaglutide cardiovascular endpoints in this cohort.",
        "url": "https://openalex.org/W123",
        "_adapter": "openalex",
        "doi": "10.1000/test.2",
        "publication_date": "2023",
        "peer_reviewed": True,
    },
    {
        "title": "Ant colony optimisation for routing",
        "snippet": "Swarm heuristics for vehicle routing.",
        "url": "https://arxiv.org/abs/0000.0001",
        "_adapter": "arxiv",
    },
]


class CeEvalRankerTests(FakeScorerTestCase):
    def test_unavailable_raises(self) -> None:
        from core.knowledge.quality_eval import RankerUnavailable, get_fixture_ranker

        set_reranker_scorer_for_tests(None)
        with patch.object(rerank, "unavailable_reason", return_value=rerank.SKIP_MODEL_MISSING):
            with self.assertRaises(RankerUnavailable):
                get_fixture_ranker("ce")

    def test_orders_by_score_gates_and_abstains(self) -> None:
        from core.knowledge.quality_eval import get_fixture_ranker

        set_reranker_scorer_for_tests(
            lambda _q, docs: [{"good": 3.0, "better": 6.0, "bad": -4.0}[d.split(".")[0]] for d in docs]
        )
        rank = get_fixture_ranker("ce")
        rows = [{"title": t, "snippet": "x"} for t in ("good", "bad", "better")]
        out = rank("heat pump efficiency", rows, 3, "general_web")
        self.assertEqual([r["title"] for r in out.kept], ["better", "good"])
        self.assertEqual([r["title"] for r in out.rejected], ["bad"])
        self.assertEqual(out.state, "sufficient")
        out = rank("heat pump efficiency", [{"title": "bad", "snippet": "x"}], 3, "general_web")
        self.assertEqual(out.kept, [])
        self.assertEqual(out.state, "insufficient")


class ScientificPipelineShadowTests(unittest.TestCase):
    """S3 acceptance: kept sources identical with shadow scoring on and off."""

    def _run(self):
        from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
        from core.knowledge.web_retrieval import run_v2_web_retrieval

        with patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None), patch(
            "core.knowledge.pipeline_scientific.set_cached_rows"
        ), patch.dict(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": lambda q, max_results=3: [dict(_SCI_ROWS[0])],
                "openalex": lambda q, max_results=3: [dict(_SCI_ROWS[1])],
                "arxiv": lambda q, max_results=3: [dict(_SCI_ROWS[2])],
            },
            clear=False,
        ):
            return run_v2_web_retrieval(
                query="semaglutide cardiovascular outcomes",
                semantic_query="semaglutide cardiovascular outcomes",
                knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
            )

    def test_kept_identical_on_and_off(self) -> None:
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False):
            off = self._run()
        set_reranker_scorer_for_tests(lambda _q, docs: [5.0 if "emaglutide" in d else -8.0 for d in docs])
        try:
            on = self._run()
        finally:
            set_reranker_scorer_for_tests(None)
        assert off.bundle is not None and on.bundle is not None
        self.assertEqual([s.id for s in off.bundle.sources], [s.id for s in on.bundle.sources])
        self.assertEqual(off.relevance_diag["rerank"]["skipped_reason"], SKIP_DISABLED)
        shadow = on.relevance_diag["rerank"]
        self.assertNotIn("skipped_reason", shadow)
        self.assertEqual(shadow["scored"], shadow["pool_size"])
        self.assertEqual(shadow["would_state"], "sufficient")
        import json

        from core.knowledge.observability import build_retrieval_trace, serialize_retrieval_trace

        json.dumps(shadow)  # persisted verbatim in the retrieval trace
        self.assertIn("rerank", json.dumps(serialize_retrieval_trace(
            build_retrieval_trace(on.bundle, relevance_diag=on.relevance_diag)
        )))


@unittest.skipUnless(rerank._model_path() is not None, "reranker model not downloaded")
class RerankModelIntegrationTests(unittest.TestCase):
    def test_real_model_separates_on_and_off_topic(self) -> None:
        rerank.unload_reranker()
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True):
            res = rerank_texts(
                "How does actor-network theory (ANT) analyze artificial intelligence?",
                [
                    "Generative artificial intelligence in jazz improvisation: an Actor-Network Theory perspective.",
                    "Artificial Ant Colonies in Digital Image Habitats - A Mass Behaviour Effect Study on Pattern Recognition.",
                ],
            )
        self.assertIsNotNone(res)
        self.assertGreater(res.scores[0], res.scores[1])
        self.assertGreater(res.scores[0], 0.0)


if __name__ == "__main__":
    unittest.main()
