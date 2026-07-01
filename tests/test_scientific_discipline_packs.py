"""Tests for Phase 6c scientific discipline pack registry."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.scientific_discipline import (  # noqa: E402
    preferred_adapters_for_discipline,
)
from core.knowledge.scientific_discipline_packs import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_BIOLOGY,
    SCIENTIFIC_DISCIPLINE_BIOMEDICAL,
    SCIENTIFIC_DISCIPLINE_CHEMISTRY,
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
    SCIENTIFIC_DISCIPLINE_ECONOMICS,
    SCIENTIFIC_DISCIPLINE_MEDICINE,
    SCIENTIFIC_DISCIPLINE_PHYSICS,
    SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE,
    SCIENTIFIC_DISCIPLINE_PSYCHOLOGY,
    SCIENTIFIC_DISCIPLINE_SOCIOLOGY,
    get_discipline_pack,
    normalize_discipline_id,
    planned_primary_adapter_ids,
)
from core.knowledge.types import (  # noqa: E402
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
)


class TestScientificDisciplinePacks(unittest.TestCase):
    def test_biomedical_alias_resolves_to_medicine_pack(self) -> None:
        self.assertEqual(normalize_discipline_id("biomedical"), SCIENTIFIC_DISCIPLINE_MEDICINE)
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_BIOMEDICAL)
        assert pack is not None
        self.assertEqual(pack.id, SCIENTIFIC_DISCIPLINE_MEDICINE)

    def test_economics_pack_not_finance_service(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_ECONOMICS)
        assert pack is not None
        self.assertEqual(pack.knowledge_service, SERVICE_SCIENTIFIC_EVIDENCE)
        self.assertNotEqual(pack.knowledge_service, SERVICE_FINANCE_KNOWLEDGE)

    def test_finance_and_legal_are_not_discipline_packs(self) -> None:
        self.assertIsNone(get_discipline_pack(SERVICE_FINANCE_KNOWLEDGE))
        self.assertIsNone(get_discipline_pack(SERVICE_LEGAL_KNOWLEDGE))

    def test_active_cs_pack_adapter_order(self) -> None:
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        self.assertEqual(order, ("arxiv", "dblp", "openalex"))

    def test_active_biology_pack_adapter_order(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_BIOLOGY)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_BIOLOGY)
        self.assertEqual(order, ("pubmed", "biorxiv", "openalex"))

    def test_active_chemistry_pack_adapter_order(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_CHEMISTRY)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_CHEMISTRY)
        self.assertEqual(order, ("pubchem", "openalex", "pubmed"))

    def test_active_psychology_pack_adapter_order(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_PSYCHOLOGY)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_PSYCHOLOGY)
        self.assertEqual(order, ("pubmed", "openalex"))

    def test_active_sociology_pack_adapter_order(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_SOCIOLOGY)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_SOCIOLOGY)
        self.assertEqual(order, ("openalex",))

    def test_active_political_science_pack_adapter_order(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_POLITICAL_SCIENCE)
        self.assertEqual(order, ("openalex",))

    def test_active_economics_pack_prefers_repec(self) -> None:
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_ECONOMICS)
        self.assertEqual(order[0], "repec")

    def test_planned_adapters_include_future_sources(self) -> None:
        planned = planned_primary_adapter_ids()
        self.assertIn("pubchem", planned)
        self.assertIn("biorxiv", planned)
        self.assertIn("inspire_hep", planned)

    def test_physics_pack_active(self) -> None:
        pack = get_discipline_pack(SCIENTIFIC_DISCIPLINE_PHYSICS)
        assert pack is not None
        self.assertEqual(pack.status, "active")
        order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_PHYSICS)
        self.assertEqual(order, ("arxiv", "inspire_hep", "openalex"))


if __name__ == "__main__":
    unittest.main()
