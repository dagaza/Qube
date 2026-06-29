"""Tests for finance knowledge service (Phase 6 Slice 5a)."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.finance_query_planner import plan_finance_query  # noqa: E402
from core.knowledge.registry import (  # noqa: E402
    resolve_turn_knowledge_service,
)
from core.knowledge.services.finance_knowledge import FinanceKnowledgeService  # noqa: E402
from core.knowledge.types import (  # noqa: E402
    RetrievalContext,
    SERVICE_FINANCE_KNOWLEDGE,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"
_SUBMISSIONS = json.loads((_FIXTURES / "sec_submissions_aapl.json").read_text(encoding="utf-8"))


class TestFinanceQueryPlanner(unittest.TestCase):
    def test_strips_sec_noise_and_extracts_form(self) -> None:
        plan = plan_finance_query("What are Apple Inc 10-K SEC filings?")
        self.assertIn("Apple", plan.search_query)
        self.assertIn("10-K", plan.form_types)

    def test_ticker_query(self) -> None:
        plan = plan_finance_query("AMZN recent SEC filings")
        self.assertIn("AMZN", plan.search_query.upper())


class TestFinanceKnowledgeService(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_registry_routes_finance_tool(self) -> None:
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="finance"),
            SERVICE_FINANCE_KNOWLEDGE,
        )

    @patch.dict(
        "core.knowledge.adapters.registry.SEARCH_FUNCTIONS",
        {
            "sec_edgar": lambda q, max_results=3, form_filter=(): [
                {
                    "_adapter": "sec_edgar",
                    "title": "10-K — Apple Inc.",
                    "snippet": "10-K filed 2025-10-31 for Apple Inc.",
                    "url": "https://www.sec.gov/example",
                    "document_type": "sec_filing",
                    "publication_date": "2025-10-31",
                    "form": "10-K",
                    "company": "Apple Inc.",
                    "cik": "320193",
                    "accession_number": "0000320193-25-000079",
                }
            ],
        },
        clear=False,
    )
    def test_retrieve_builds_bundle_with_disclaimer(self) -> None:
        service = FinanceKnowledgeService()
        ctx = RetrievalContext(
            query="Apple 10-K",
            semantic_query="Apple 10-K",
            knowledge_service=SERVICE_FINANCE_KNOWLEDGE,
        )
        bundle, rel_diag, _raw = service.retrieve(ctx)
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.knowledge_service, SERVICE_FINANCE_KNOWLEDGE)
        self.assertIn("not_financial_advice", bundle.warnings)
        self.assertEqual(bundle.sources[0].adapter, "sec_edgar")
        self.assertGreaterEqual(bundle.sources[0].authority_score, 0.9)
        self.assertIn("finance_search_query", rel_diag or {})


if __name__ == "__main__":
    unittest.main()
