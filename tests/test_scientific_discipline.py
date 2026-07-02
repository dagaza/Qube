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
    SCIENTIFIC_DISCIPLINE_BIOLOGY,
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL,
    SCIENTIFIC_DISCIPLINE_CHEMISTRY,
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_GENERAL,
    SCIENTIFIC_DISCIPLINE_PHYSICS,
    SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE,
    SCIENTIFIC_DISCIPLINE_PSYCHOLOGY,
    SCIENTIFIC_DISCIPLINE_SOCIOLOGY,
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
            "reproducibility crisis open science preregistration meta-analysis"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_GENERAL)

    def test_biology_query_without_clinical_framing(self) -> None:
        match = detect_scientific_discipline(
            "microbiome diversity soil ecology metagenomics"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_BIOLOGY)
        self.assertEqual(match.ui_group, "Biology")

    def test_clinical_query_stays_biomedical_not_biology(self) -> None:
        match = detect_scientific_discipline(
            "Ozempic semaglutide cardiovascular outcomes randomized trial"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_BIOMEDICAL)

    def test_crispr_gene_editing_routes_biology(self) -> None:
        match = detect_scientific_discipline(
            "CRISPR Cas9 gene editing off-target effects in human cells"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_BIOLOGY)

    def test_chemistry_compound_binding_query(self) -> None:
        match = detect_scientific_discipline(
            "aspirin acetylsalicylic acid binding COX-2 cyclooxygenase enzyme kinetics"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_CHEMISTRY)
        self.assertEqual(match.ui_group, "Chemistry")

    def test_clinical_drug_query_not_chemistry(self) -> None:
        match = detect_scientific_discipline(
            "aspirin cardiovascular outcomes randomized clinical trial patients"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_BIOMEDICAL)

    def test_psychology_cognitive_query(self) -> None:
        match = detect_scientific_discipline(
            "working memory cognitive load dual-task experiment psychology"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_PSYCHOLOGY)
        self.assertEqual(match.ui_group, "Psychology")

    def test_sociology_stratification_query(self) -> None:
        match = detect_scientific_discipline(
            "social stratification income inequality sociology survey methods"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_SOCIOLOGY)
        self.assertEqual(match.ui_group, "Social Science")

    def test_political_science_electoral_query(self) -> None:
        match = detect_scientific_discipline(
            "voter turnout electoral reform democracy comparative political science"
        )
        self.assertEqual(match.discipline, SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE)
        self.assertEqual(match.ui_group, "Social Science")


class TestDisciplineAdapterPolicy(unittest.TestCase):
    def test_cs_prefers_arxiv_before_openalex(self) -> None:
        enabled = ("pubmed", "openalex", "arxiv")
        resolved = apply_scientific_adapter_policy(
            enabled,
            query="transformer attention mechanism",
        )
        self.assertEqual(resolved, ("arxiv", "openalex"))

    def test_economics_repec_before_openalex(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv", "repec"),
            query="GDP inflation monetary policy econometric",
        )
        self.assertEqual(resolved, ("repec", "openalex"))

    def test_biomedical_includes_pubmed_first(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv"),
            query="semaglutide cardiovascular outcomes",
        )
        self.assertEqual(resolved[0], "pubmed")

    def test_biology_prefers_pubmed_then_biorxiv(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv", "biorxiv"),
            query="microbiome diversity soil ecology metagenomics",
        )
        self.assertEqual(resolved[:2], ("pubmed", "biorxiv"))

    def test_chemistry_prefers_pubchem_first(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv", "pubchem"),
            query="aspirin acetylsalicylic acid binding COX-2 cyclooxygenase",
        )
        self.assertEqual(resolved[0], "pubchem")

    def test_physics_prefers_arxiv_then_inspire_hep(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv", "inspire_hep"),
            query="gravitational wave detection LIGO binary black hole",
        )
        self.assertEqual(resolved[:2], ("arxiv", "inspire_hep"))

    def test_psychology_prefers_pubmed_first(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv"),
            query="working memory cognitive load dual-task experiment psychology",
        )
        self.assertEqual(resolved[:2], ("pubmed", "openalex"))

    def test_sociology_openalex_only(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv"),
            query="social stratification income inequality sociology survey methods",
        )
        self.assertEqual(resolved, ("openalex",))

    def test_political_science_openalex_only(self) -> None:
        resolved = apply_scientific_adapter_policy(
            ("pubmed", "openalex", "arxiv"),
            query="voter turnout electoral reform democracy comparative political science",
        )
        self.assertEqual(resolved, ("openalex",))

    def test_catalog_cs_group_order(self) -> None:
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        self.assertEqual(
            order,
            ("arxiv", "dblp", "openreview", "acl_anthology", "openalex", "acm_dl"),
        )

    def test_resolve_service_adapters_cs_order(self) -> None:
        resolved = resolve_service_adapters(
            SERVICE_SCIENTIFIC_EVIDENCE,
            query="deep learning neural network GPU training",
            stored_preferences={
                SERVICE_SCIENTIFIC_EVIDENCE: ["pubmed", "openalex", "arxiv"],
            },
        )
        self.assertEqual(resolved, ("arxiv", "dblp", "openalex"))


if __name__ == "__main__":
    unittest.main()
