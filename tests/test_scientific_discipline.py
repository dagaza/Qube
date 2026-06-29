"""Tests for Phase 6 Slice 6a scientific discipline routing."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.scientific_adapters import apply_scientific_adapter_policy  # noqa: E402
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL,
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_GENERAL,
    SCIENTIFIC_DISCIPLINE_PHYSICS,
    detect_scientific_discipline,
    preferred_adapters_for_discipline,
)
from core.knowledge.source_preferences import resolve_service_adapters  # noqa: E402
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402


class TestScientificDisciplineDetection(unittest.TestCase):
    def test_biomedical_query(self) -> None:
        match = detect_scientific_discipline("ACE inhibitors heart failure")
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_BIOMEDICAL)
        self.assertEqual(match.ui_group, "Science")

    def test_computer_science_query(self) -> None:
        match = detect_scientific_discipline(
            "transformer attention mechanism neural machine translation"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        self.assertEqual(match.ui_group, "Computer Science")

    def test_economics_query(self) -> None:
        match = detect_scientific_discipline(
            "monetary policy inflation econometric VAR model"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_ECONOMICS)
        self.assertEqual(match.ui_group, "Economics")

    def test_physics_query(self) -> None:
        match = detect_scientific_discipline(
            "gravitational wave detection LIGO binary black hole"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_PHYSICS)

    def test_general_science_fallback(self) -> None:
        match = detect_scientific_discipline(
            "climate change Arctic sea ice extent satellite observations"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_GENERAL)


class TestDisciplineAdapterPolicy(unittest.TestCase):
    def test_cs_prefers_arxiv_before_openalex(self) -> None:
        enabled = ("pubmed", "openalex", "arxiv")
        resolved = apply_scientific_adapter_policy(
            enabled,
            query="transformer attention mechanism",
        )
        self.assertEqual(resolved, ("arxiv", "openalex"))

    def test_economics_openalex_only_when_repec_unimplemented(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv"),
            query="GDP inflation monetary policy econometric",
        )
        self.assertEqual(resolved, ("openalex",))

    def test_biomedical_includes_pubmed_first(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv"),
            query="semaglutide cardiovascular outcomes",
        )
        self.assertEqual(resolved[0], "pubmed")

    def test_catalog_cs_group_order(self) -> None:
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        self.assertEqual(order, ("arxiv", "openalex"))

    def test_resolve_service_adapters_cs_order(self) -> None:
        resolved = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query="deep learning neural network GPU training",
            stored_preferences={
                SERVICE_SCIENTIFIC_EVIDENCE: ["pubmed", "openalex", "arxiv"],
            },
        )
        self.assertEqual(resolved, ("arxiv", "openalex"))


if __name__ == "__main__":
    unittest.main()
