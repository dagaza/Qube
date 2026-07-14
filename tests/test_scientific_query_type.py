"""Tests for Slice 19 scientific query-type routing."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.scientific_adapters import apply_scientific_adapter_policy  # noqa: E402
from core.knowledge.scientific_query_type import (  # noqa: E402
    QUERY_TYPE_GUIDELINE,
    QUERY_TYPE_LITERATURE,
    QUERY_TYPE_PATENT,
    QUERY_TYPE_STANDARD,
    QUERY_TYPE_STATISTICS,
    detect_scientific_query_type,
    query_type_routing_enabled,
    reorder_adapters_for_query_type,
)
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402
from core.knowledge.source_preferences import resolve_service_adapters  # noqa: E402


class TestScientificQueryTypeDetection(unittest.TestCase):
    def test_guideline_treatment_query(self) -> None:
        match = detect_scientific_query_type("How is hypertension treated in adults?")
        self.assertEqual(match.query_type, QUERY_TYPE_GUIDELINE)

    def test_statistics_unemployment_query(self) -> None:
        match = detect_scientific_query_type("What is the current US unemployment rate?")
        self.assertEqual(match.query_type, QUERY_TYPE_STATISTICS)

    def test_standard_rfc_query(self) -> None:
        match = detect_scientific_query_type("What does RFC 8446 specify for TLS?")
        self.assertEqual(match.query_type, QUERY_TYPE_STANDARD)

    def test_patent_query(self) -> None:
        match = detect_scientific_query_type("lithium ion battery electrode patent search")
        self.assertEqual(match.query_type, QUERY_TYPE_PATENT)

    def test_research_query_stays_literature(self) -> None:
        match = detect_scientific_query_type(
            "GDP inflation monetary policy econometric VAR model published studies"
        )
        self.assertEqual(match.query_type, QUERY_TYPE_LITERATURE)

    def test_medical_patent_ductus_not_patent_type(self) -> None:
        match = detect_scientific_query_type("patent ductus arteriosus neonatal treatment")
        self.assertEqual(match.query_type, QUERY_TYPE_LITERATURE)


class TestQueryTypeAdapterReorder(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_QUERY_TYPE_ROUTING", None)

    def test_guideline_boosts_nice_before_pubmed(self) -> None:
        enabled = (
            "pubmed",
            "openalex",
            "nice",
            "cdc",
            "who",
            "clinicaltrials_gov",
            "openfda",
        )
        resolved = apply_scientific_adapter_policy(
            enabled,
            query="How is hypertension treated according to clinical guidelines?",
        )
        self.assertEqual(resolved[0], "nice")
        self.assertIn("pubmed", resolved)

    def test_statistics_boosts_bls_for_economics(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("repec", "openalex", "world_bank", "eurostat", "oecd", "bls"),
            query="What is the current US unemployment rate from official statistics?",
        )
        self.assertEqual(resolved[0], "bls")

    def test_standard_boosts_ietf_rfc(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("arxiv", "openalex", "ieee_xplore", "nist", "ietf_rfc"),
            query="IETF standard for HTTP/3 internet protocol specification",
        )
        self.assertEqual(resolved[0], "ietf_rfc")

    def test_literature_query_unchanged(self) -> None:
        before = ("pubmed", "openalex", "arxiv")
        after = reorder_adapters_for_query_type(
            before,
            query="semaglutide cardiovascular outcomes randomized trial publication",
        )
        self.assertEqual(after, before)

    def test_flag_off_skips_reorder(self) -> None:
        os.environ["QUBE_QUERY_TYPE_ROUTING"] = "0"
        self.assertFalse(query_type_routing_enabled())
        before = ("pubmed", "nice", "cdc")
        after = reorder_adapters_for_query_type(
            before,
            query="How is hypertension treated in adults?",
        )
        self.assertEqual(after, before)


class TestQueryTypeResolveService(unittest.TestCase):
    def test_resolve_hypertension_guideline_order(self) -> None:
        resolved = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query="How is hypertension treated in adults?",
            stored_preferences={
                SERVICE_SCIENTIFIC_EVIDENCE: [
                    "pubmed",
                    "openalex",
                    "nice",
                    "cdc",
                    "who",
                ],
            },
        )
        self.assertEqual(resolved[0], "nice")


if __name__ == "__main__":
    unittest.main()
