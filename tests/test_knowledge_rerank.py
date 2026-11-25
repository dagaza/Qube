"""Unit tests for the cross-encoder rerank module (plan S3).

The real model is exercised only in ``RerankModelIntegrationTests`` (skipped
when the reranker is not downloaded); everything else injects a fake scorer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

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
            "core.knowledge.ranking.rerank._model_path", return_value=None
        ):
            self.assertEqual(unavailable_reason(), SKIP_MODEL_MISSING)
            self.assertFalse(rerank.reranker_available())
            self.assertFalse(rerank.warm_up_reranker())
            self.assertIsNone(rerank_texts("what is a heat pump", ["doc"]))

    def test_load_failure_is_swallowed_and_backed_off(self) -> None:
        rerank.unload_reranker()
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True), patch(
            "core.knowledge.ranking.rerank._model_path", return_value="/nonexistent"
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


@unittest.skipUnless(rerank._model_path() is not None, "reranker model not downloaded")
@pytest.mark.real_reranker
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
