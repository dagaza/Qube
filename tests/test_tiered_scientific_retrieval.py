"""Tests for tiered scientific adapter fan-out (HTTP resilience Slice 8)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.pipeline_scientific import ScientificEvidencePipeline  # noqa: E402
from core.knowledge.tiered_scientific_retrieval import (  # noqa: E402
    split_adapter_tiers,
    tiered_scientific_retrieval_enabled,
)
from core.knowledge.types import RetrievalBudget, RetrievalContext  # noqa: E402


def _row(title: str, adapter: str) -> dict:
    return {
        "title": title,
        "snippet": title,
        "full_text": title,
        "url": f"https://example.test/{adapter}",
        "_adapter": adapter,
        "document_type": "journal_abstract",
    }


class TestTieredScientificHelpers(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_TIERED_SCIENTIFIC_RETRIEVAL", None)

    def test_flag_defaults_off(self) -> None:
        os.environ.pop("QUBE_TIERED_SCIENTIFIC_RETRIEVAL", None)
        self.assertFalse(tiered_scientific_retrieval_enabled())

    def test_flag_enabled(self) -> None:
        os.environ["QUBE_TIERED_SCIENTIFIC_RETRIEVAL"] = "1"
        self.assertTrue(tiered_scientific_retrieval_enabled())

    def test_split_sociology_openalex_primary(self) -> None:
        primary, fallback = split_adapter_tiers(
            ("openalex",),
            discipline="sociology",
        )
        self.assertEqual(primary, ("openalex",))
        self.assertEqual(fallback, ())

    def test_split_biomedical_pubmed_primary(self) -> None:
        primary, fallback = split_adapter_tiers(
            ("pubmed", "openalex", "arxiv"),
            discipline="biomedical",
        )
        self.assertEqual(primary, ("pubmed",))
        self.assertEqual(fallback, ("openalex", "arxiv"))


class TestTieredScientificPipeline(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_TIERED_SCIENTIFIC_RETRIEVAL", None)

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_sociology_skips_fallback_when_primary_sufficient(
        self, _set_cache, _get_cache
    ) -> None:
        os.environ["QUBE_TIERED_SCIENTIFIC_RETRIEVAL"] = "1"
        calls: dict[str, int] = {"openalex": 0, "arxiv": 0}

        def _search(aid: str):
            def fn(q: str, max_results: int = 3) -> list[dict]:
                calls[aid] += 1
                if aid == "openalex":
                    return [
                        _row("Social stratification study A", aid),
                        _row("Social stratification study B", aid),
                        _row("Social stratification study C", aid),
                    ]
                return [_row("Fallback arxiv hit", aid)]

            return fn

        with patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters",
            return_value=("openalex", "arxiv"),
        ), patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "openalex": _search("openalex"),
                "arxiv": _search("arxiv"),
            },
        ):
            bundle, rel_diag, _ = ScientificEvidencePipeline().run(
                RetrievalContext(
                    query="social stratification income inequality sociology survey methods",
                    semantic_query="social stratification income inequality sociology survey methods",
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(calls["openalex"], 1)
        self.assertEqual(calls["arxiv"], 0)
        assert rel_diag is not None
        tiered = rel_diag["scientific_tiered_retrieval"]
        self.assertTrue(tiered["enabled"])
        self.assertEqual(tiered["phase1_adapters"], ["openalex"])
        self.assertEqual(tiered["phase2_adapters"], [])
        self.assertTrue(tiered["phase2_skipped"])
        self.assertIn("openalex", bundle.adapter_calls)

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_biomedical_skips_openalex_when_pubmed_sufficient(
        self, _set_cache, _get_cache
    ) -> None:
        os.environ["QUBE_TIERED_SCIENTIFIC_RETRIEVAL"] = "1"
        calls: dict[str, int] = {"pubmed": 0, "openalex": 0, "arxiv": 0}

        def _search(aid: str):
            def fn(q: str, max_results: int = 3) -> list[dict]:
                calls[aid] += 1
                if aid == "pubmed":
                    return [
                        _row("Semaglutide cardiovascular outcomes", "pubmed"),
                        _row("GLP-1 heart failure trial", "pubmed"),
                        _row("Ozempic outcomes review", "pubmed"),
                    ]
                return [_row("Fallback hit", aid)]

            return fn

        with patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters",
            return_value=("pubmed", "openalex", "arxiv"),
        ), patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _search("pubmed"),
                "openalex": _search("openalex"),
                "arxiv": _search("arxiv"),
            },
        ):
            bundle, rel_diag, _ = ScientificEvidencePipeline().run(
                RetrievalContext(
                    query="Ozempic semaglutide cardiovascular outcomes randomized trial",
                    semantic_query="Ozempic semaglutide cardiovascular outcomes randomized trial",
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(calls["pubmed"], 1)
        self.assertEqual(calls["openalex"], 0)
        self.assertEqual(calls["arxiv"], 0)
        assert rel_diag is not None
        self.assertTrue(rel_diag["scientific_tiered_retrieval"]["phase2_skipped"])
        self.assertIn("pubmed", bundle.adapter_calls)

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_biomedical_invokes_fallback_when_primary_sparse(
        self, _set_cache, _get_cache
    ) -> None:
        os.environ["QUBE_TIERED_SCIENTIFIC_RETRIEVAL"] = "1"
        calls: dict[str, int] = {"pubmed": 0, "openalex": 0}

        def _search(aid: str):
            def fn(q: str, max_results: int = 3) -> list[dict]:
                calls[aid] += 1
                if aid == "pubmed":
                    return []
                return [_row("OpenAlex semaglutide paper", "openalex")]

            return fn

        with patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters",
            return_value=("pubmed", "openalex"),
        ), patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _search("pubmed"),
                "openalex": _search("openalex"),
            },
        ):
            bundle, rel_diag, _ = ScientificEvidencePipeline().run(
                RetrievalContext(
                    query="Ozempic semaglutide cardiovascular outcomes randomized trial",
                    semantic_query="Ozempic semaglutide cardiovascular outcomes randomized trial",
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(calls["pubmed"], 1)
        self.assertEqual(calls["openalex"], 1)
        assert rel_diag is not None
        tiered = rel_diag["scientific_tiered_retrieval"]
        self.assertFalse(tiered["phase2_skipped"])
        self.assertEqual(tiered["phase2_adapters"], ["openalex"])
        self.assertIn("openalex", bundle.adapter_calls)

    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    def test_flag_off_calls_all_adapters_in_parallel(
        self, _set_cache, _get_cache
    ) -> None:
        os.environ.pop("QUBE_TIERED_SCIENTIFIC_RETRIEVAL", None)
        calls: dict[str, int] = {"pubmed": 0, "openalex": 0}

        def _search(aid: str):
            def fn(q: str, max_results: int = 3) -> list[dict]:
                calls[aid] += 1
                return [_row(f"{aid} hit", aid)]

            return fn

        with patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters",
            return_value=("pubmed", "openalex"),
        ), patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _search("pubmed"),
                "openalex": _search("openalex"),
            },
        ):
            _bundle, rel_diag, _ = ScientificEvidencePipeline().run(
                RetrievalContext(
                    query="Ozempic semaglutide cardiovascular outcomes randomized trial",
                    semantic_query="Ozempic semaglutide cardiovascular outcomes randomized trial",
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(calls["pubmed"], 1)
        self.assertEqual(calls["openalex"], 1)
        assert rel_diag is not None
        self.assertFalse(rel_diag["scientific_tiered_retrieval"]["enabled"])


if __name__ == "__main__":
    unittest.main()
