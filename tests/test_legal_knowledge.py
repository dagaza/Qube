"""Tests for legal knowledge service (Phase 6 Slice 5b)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.legal_query_planner import plan_legal_query  # noqa: E402
from core.knowledge.registry import resolve_turn_knowledge_service  # noqa: E402
from core.knowledge.services.legal_knowledge import LegalKnowledgeService  # noqa: E402
from core.knowledge.types import RetrievalContext, SERVICE_LEGAL_KNOWLEDGE  # noqa: E402


class TestLegalQueryPlanner(unittest.TestCase):
    def test_strips_legal_noise_and_adds_scotus_filter(self) -> None:
        plan = plan_legal_query(
            "What is the Supreme Court precedent for Miranda rights during interrogation?"
        )
        self.assertIn("Miranda", plan.search_query)
        self.assertIn("court_id:scotus", plan.search_query)

    def test_extracts_case_name_from_conversational_prompt(self) -> None:
        plan = plan_legal_query(
            "What did the Supreme Court hold in Miranda v Arizona about police interrogation?"
        )
        self.assertEqual(plan.search_query, "Miranda v. Arizona court_id:scotus")

    def test_extracts_case_name_from_short_prompt(self) -> None:
        plan = plan_legal_query("Miranda v Arizona police interrogation verdict")
        self.assertEqual(plan.search_query, "Miranda v. Arizona")

    def test_preserves_case_name(self) -> None:
        plan = plan_legal_query("Brown v Board of Education school segregation")
        self.assertIn("Brown", plan.search_query)


class TestLegalKnowledgeService(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_registry_routes_legal_tool(self) -> None:
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="legal"),
            SERVICE_LEGAL_KNOWLEDGE,
        )

    @patch.dict(
        "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
        {
            "courtlistener": lambda q, max_results=3: [
                {
                    "_adapter": "courtlistener",
                    "title": "Miranda v. Arizona",
                    "snippet": "The person in custody must be warned prior to questioning.",
                    "url": "https://www.courtlistener.com/opinion/example/",
                    "document_type": "court_opinion",
                    "publication_date": "1966-06-13",
                    "court": "Supreme Court of the United States",
                    "court_id": "scotus",
                    "citation": ["384 U.S. 436"],
                    "authority_score": 0.95,
                }
            ],
        },
        clear=False,
    )
    @patch(
        "core.knowledge.pipeline_legal.get_knowledge_source_preferences",
        return_value={"legal_knowledge": ["courtlistener"]},
    )
    def test_retrieve_builds_bundle_with_disclaimer(self, _mock_prefs) -> None:
        service = LegalKnowledgeService()
        ctx = RetrievalContext(
            query="Miranda rights police interrogation",
            semantic_query="Miranda rights police interrogation",
            knowledge_service=SERVICE_LEGAL_KNOWLEDGE,
        )
        bundle, rel_diag, _raw = service.retrieve(ctx)
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.knowledge_service, SERVICE_LEGAL_KNOWLEDGE)
        self.assertIn("not_legal_advice", bundle.warnings)
        self.assertEqual(bundle.sources[0].adapter, "courtlistener")
        self.assertGreaterEqual(bundle.sources[0].authority_score, 0.9)
        self.assertIn("legal_search_query", rel_diag or {})


if __name__ == "__main__":
    unittest.main()
