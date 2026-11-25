"""Plan S22 (D35): adapters that need a missing API key are skipped and named.

Key-required adapters returned ``[]`` without a key (IEEE Xplore, NICE, EPO OPS
in the S13 cases), which a trace could not tell from "found nothing". The
scientific, finance, legal and generic pipelines now leave them out before any
task is built and list them in ``adapter_skips``.
"""

from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.adapter_key_gate import SKIP_NO_KEY, missing_required_key, split_keyless  # noqa: E402
from core.knowledge.ranking.rerank import set_reranker_scorer_for_tests  # noqa: E402
from core.knowledge.retrieval_profiles import get_profile_spec  # noqa: E402
from core.knowledge.types import RetrievalBudget, RetrievalContext  # noqa: E402

_KEY_ENV = (
    "QUBE_IEEE_XPLORE_API_KEY",
    "QUBE_NICE_API_KEY",
    "QUBE_EPO_OPS_CONSUMER_SECRET",
    "QUBE_FRED_API_KEY",
    "QUBE_CANLII_API_KEY",
    "QUBE_SEMANTIC_SCHOLAR_API_KEY",
    "QUBE_KNOWLEDGE_FIXTURES",
)


class _NoKeys:
    """No credential from the environment or settings, unless ``env`` sets one."""

    def __init__(self, **env: str) -> None:
        self.env = env
        self.stack = ExitStack()

    def __enter__(self):
        clean = {k: v for k, v in os.environ.items() if k not in _KEY_ENV}
        clean.update(self.env)
        self.stack.enter_context(patch.dict(os.environ, clean, clear=True))
        self.stack.enter_context(patch("core.knowledge.credentials._user_secret", return_value=None))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        return False


def _row(title: str, adapter: str) -> dict:
    return {
        "title": title,
        "snippet": title,
        "full_text": title,
        "url": f"https://example.test/{adapter}/{abs(hash(title)) % 10_000}",
        "_adapter": adapter,
        "document_type": "journal_abstract",
    }


class GateTests(unittest.TestCase):
    def test_key_required_adapters_without_a_key_are_gated(self) -> None:
        with _NoKeys():
            for aid in ("ieee_xplore", "nice", "epo_espacenet", "fred", "canlii", "semantic_scholar"):
                with self.subTest(adapter=aid):
                    self.assertTrue(missing_required_key(aid))

    def test_a_configured_key_opens_the_gate(self) -> None:
        with _NoKeys(QUBE_IEEE_XPLORE_API_KEY="test-key"):
            self.assertFalse(missing_required_key("ieee_xplore"))
            self.assertTrue(missing_required_key("nice"))

    def test_optional_key_and_keyless_adapters_are_never_gated(self) -> None:
        with _NoKeys():
            for aid in ("openalex", "pubmed", "nist", "us_census", "crossref", "arxiv", "ietf_rfc", "not_an_adapter"):
                with self.subTest(adapter=aid):
                    self.assertFalse(missing_required_key(aid))

    def test_fixture_mode_gates_nothing(self) -> None:
        with _NoKeys(QUBE_KNOWLEDGE_FIXTURES="1"):
            self.assertFalse(missing_required_key("ieee_xplore"))

    def test_split_keeps_order_and_names_each_skip(self) -> None:
        with _NoKeys():
            runnable, skips = split_keyless(("ieee_xplore", "openalex", "nice", "arxiv"))
        self.assertEqual(runnable, ("openalex", "arxiv"))
        self.assertEqual(
            skips,
            [{"adapter": "ieee_xplore", "reason": SKIP_NO_KEY}, {"adapter": "nice", "reason": SKIP_NO_KEY}],
        )


class ScientificPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        set_reranker_scorer_for_tests(lambda q, texts: [5.0 for _ in texts])
        self.calls: list[str] = []

    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def _search(self, aid: str):
        def fn(q: str, max_results: int = 3) -> list[dict]:
            self.calls.append(aid)
            return [_row(f"{aid} paper {i} on heat pumps", aid) for i in range(max_results)]

        return fn

    def _run(self, adapters):
        from core.knowledge.pipeline_scientific import ScientificEvidencePipeline

        table = {aid: self._search(aid) for aid in adapters}
        with patch(
            "core.knowledge.pipeline_scientific.resolve_service_adapters", return_value=adapters
        ), patch("core.knowledge.adapters.registry.SEARCH_FUNCTIONS", table), patch(
            "core.knowledge.pipeline_scientific.get_profile_spec", return_value=get_profile_spec("balanced")
        ), patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None), patch(
            "core.knowledge.pipeline_scientific.set_cached_rows"
        ):
            _bundle, diag, _raw = ScientificEvidencePipeline().run(
                RetrievalContext(query="heat pump efficiency", semantic_query="heat pump efficiency",
                                 budget=RetrievalBudget(max_results=3))
            )
        return diag

    def test_gated_adapter_is_not_searched_and_is_named(self) -> None:
        with _NoKeys():
            diag = self._run(("ieee_xplore", "openalex"))
        self.assertNotIn("ieee_xplore", self.calls)
        self.assertIn("openalex", self.calls)
        self.assertEqual(diag["scientific_pool"]["adapter_skips"], [{"adapter": "ieee_xplore", "reason": "no_key"}])
        self.assertNotIn("ieee_xplore", diag["scientific_adapters_selected"])
        tasks = [t["adapter"] for c in diag["scientific_pool"]["collection"] for t in c["tasks"]]
        self.assertNotIn("ieee_xplore", tasks)

    def test_with_a_key_the_adapter_is_searched(self) -> None:
        with _NoKeys(QUBE_IEEE_XPLORE_API_KEY="test-key"):
            diag = self._run(("ieee_xplore", "openalex"))
        self.assertIn("ieee_xplore", self.calls)
        self.assertEqual(diag["scientific_pool"]["adapter_skips"], [])


class ServicePipelineTests(unittest.TestCase):
    def _run(self, module_name: str, pipeline_cls: str, service: str, adapters: tuple[str, ...]):
        import importlib

        module = importlib.import_module(f"core.knowledge.{module_name}")
        calls: list[str] = []

        def search_for(aid):
            def fn(*_a, **_k):
                calls.append(aid)
                return []

            return fn

        with patch.object(module, "resolve_service_adapters", return_value=adapters), patch.object(
            module, "get_search_function", side_effect=search_for
        ), patch.object(module, "get_knowledge_source_preferences", return_value={}):
            ctx = RetrievalContext(query="Apple 10-K revenue", semantic_query="Apple 10-K revenue", knowledge_service=service)
            _bundle, diag, _ = getattr(module, pipeline_cls)().run(ctx)
        return calls, diag

    def test_finance(self) -> None:
        with _NoKeys():
            calls, diag = self._run("pipeline_finance", "FinanceEvidencePipeline", "finance_knowledge", ("fred", "world_bank"))
        self.assertEqual(calls, ["world_bank"])
        self.assertEqual(diag["adapter_skips"], [{"adapter": "fred", "reason": "no_key"}])
        self.assertEqual(diag["finance_adapters_selected"], ["world_bank"])

    def test_legal(self) -> None:
        with _NoKeys():
            calls, diag = self._run("pipeline_legal", "LegalEvidencePipeline", "legal_knowledge", ("canlii", "courtlistener"))
        self.assertEqual(calls, ["courtlistener"])
        self.assertEqual(diag["adapter_skips"], [{"adapter": "canlii", "reason": "no_key"}])

    def test_nothing_gated_leaves_no_skip_field(self) -> None:
        with _NoKeys():
            _calls, diag = self._run("pipeline_finance", "FinanceEvidencePipeline", "finance_knowledge", ("world_bank",))
        self.assertNotIn("adapter_skips", diag)

    def test_generic(self) -> None:
        from core.knowledge import pipeline_generic

        seen: list[tuple[str, ...]] = []

        def fake_collect(adapter_ids, **_kw):
            seen.append(tuple(adapter_ids))
            return [], [], [], []

        pipeline = pipeline_generic.GenericOrchestrationPipeline(
            base_service="finance_knowledge", adapter_ids=("fred", "world_bank")
        )
        with _NoKeys(), patch.object(pipeline_generic, "collect_adapter_rows", side_effect=fake_collect):
            _bundle, diag, _ = pipeline.run(RetrievalContext(query="inflation", semantic_query="inflation"))
        self.assertEqual(seen, [("world_bank",)])
        self.assertEqual(diag["adapter_skips"], [{"adapter": "fred", "reason": "no_key"}])


if __name__ == "__main__":
    unittest.main()
