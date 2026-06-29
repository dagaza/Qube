"""Tests for user-configurable knowledge source preferences."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.scientific_adapters import apply_scientific_adapter_policy  # noqa: E402
from core.knowledge.source_preferences import (  # noqa: E402
    get_effective_enabled_adapters,
    resolve_service_adapters,
    set_adapter_enabled,
)
from core.knowledge.types import (  # noqa: E402
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
)


class TestKnowledgeSourcePreferences(unittest.TestCase):
    def test_default_scientific_adapters(self) -> None:
        enabled = get_effective_enabled_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            stored_preferences={},
        )
        self.assertIn("pubmed", enabled)
        self.assertIn("openalex", enabled)
        self.assertIn("arxiv", enabled)

    def test_toggle_off_pubmed(self) -> None:
        prefs = set_adapter_enabled(
            {},
            service_id=SERVICE_SCIENTIFIC_EVIDENCE,
            adapter_id="pubmed",
            enabled=False,
        )
        enabled = get_effective_enabled_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            stored_preferences=prefs,
        )
        self.assertNotIn("pubmed", enabled)
        self.assertIn("openalex", enabled)

    def test_resolve_respects_composer_override(self) -> None:
        resolved = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query="heart failure",
            composer_adapter_filter=("arxiv",),
            stored_preferences={},
        )
        self.assertEqual(resolved, ("arxiv",))

    def test_resolve_non_medical_skips_pubmed_even_if_enabled(self) -> None:
        resolved = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query="transformer attention mechanism",
            stored_preferences={
                SERVICE_SCIENTIFIC_EVIDENCE: ["pubmed", "openalex", "arxiv"],
            },
        )
        self.assertNotIn("pubmed", resolved)
        self.assertIn("openalex", resolved)

    def test_resolve_medical_includes_pubmed_when_enabled(self) -> None:
        resolved = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query="ACE inhibitors heart failure",
            stored_preferences={
                SERVICE_SCIENTIFIC_EVIDENCE: ["pubmed", "openalex", "arxiv"],
            },
        )
        self.assertIn("pubmed", resolved)

    def test_finance_defaults_sec_edgar(self) -> None:
        enabled = get_effective_enabled_adapters(
            SERVICE_FINANCE_KNOWLEDGE,
            stored_preferences={},
        )
        self.assertEqual(enabled, ("sec_edgar",))

    def test_apply_scientific_policy_user_disabled_all_non_medical(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("arxiv",),
            query="machine learning",
        )
        self.assertEqual(resolved, ("arxiv",))


class TestScientificPipelineSourcePreferences(unittest.TestCase):
    @patch("core.knowledge.pipeline_scientific.get_cached_rows", return_value=None)
    @patch("core.knowledge.pipeline_scientific.set_cached_rows")
    @patch("core.knowledge.pipeline_scientific.get_knowledge_source_preferences")
    def test_honors_disabled_pubmed(self, mock_prefs, _set_cache, _get_cache) -> None:
        from core.knowledge.pipeline_scientific import ScientificEvidencePipeline
        from core.knowledge.types import RetrievalBudget, RetrievalContext

        mock_prefs.return_value = {
            SERVICE_SCIENTIFIC_EVIDENCE: ["openalex", "arxiv"],
        }
        pubmed_calls: list[str] = []

        def _pubmed(q: str, max_results: int = 3) -> list[dict]:
            pubmed_calls.append(q)
            return []

        with patch(
            "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
            {
                "pubmed": _pubmed,
                "openalex": lambda q, max_results=3: [
                    {
                        "title": "HF outcomes",
                        "snippet": "text",
                        "full_text": "text",
                        "_adapter": "openalex",
                    }
                ],
                "arxiv": lambda q, max_results=3: [],
            },
        ):
            pipeline = ScientificEvidencePipeline()
            bundle, rel_diag, _ = pipeline.run(
                RetrievalContext(
                    query="ACE inhibitors heart failure",
                    semantic_query="ACE inhibitors heart failure",
                    budget=RetrievalBudget(max_results=3),
                )
            )

        self.assertEqual(pubmed_calls, [])
        self.assertIn("openalex", bundle.adapter_calls)
        assert rel_diag is not None
        self.assertNotIn("pubmed", rel_diag["scientific_adapters_selected"])


if __name__ == "__main__":
    unittest.main()
