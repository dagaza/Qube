"""Tests for Slice 16 adapters (ChEMBL, UniProt, PDB, ChemRxiv)."""

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

from core.knowledge.adapters import chembl, chemrxiv, pdb, uniprot  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import adapter_credentials_hint  # noqa: E402
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_BIOLOGY,
    SCIENTIFIC_DISCIPLINE_CHEMISTRY,
    preferred_adapters_for_discipline,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_SLICE16_ADAPTER_IDS = ("chembl", "uniprot", "pdb", "chemrxiv")


class TestSlice16Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _SLICE16_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_anonymous_adapters_have_no_required_hint(self) -> None:
        for adapter_id in _SLICE16_ADAPTER_IDS:
            self.assertIsNone(adapter_credentials_hint(adapter_id), adapter_id)

    def test_discipline_pack_updates(self) -> None:
        chemistry_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_CHEMISTRY)
        biology_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_BIOLOGY)
        self.assertIn("chembl", chemistry_order)
        self.assertIn("chemrxiv", chemistry_order)
        self.assertIn("uniprot", biology_order)
        self.assertIn("pdb", biology_order)


class TestSlice16AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_chembl_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "chembl_search_aspirin.json").read_text(encoding="utf-8")
        )
        with patch.object(chembl, "fetch_search_results", return_value=fixture):
            rows = chembl.search_chembl("aspirin")
        self.assertEqual(rows[0]["_adapter"], "chembl")
        self.assertEqual(rows[0]["document_type"], "bioactive_compound")

    def test_uniprot_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "uniprot_search_insulin.json").read_text(encoding="utf-8")
        )
        with patch.object(uniprot, "fetch_search_results", return_value=fixture):
            rows = uniprot.search_uniprot("insulin")
        self.assertEqual(rows[0]["_adapter"], "uniprot")
        self.assertEqual(rows[0]["document_type"], "protein_record")

    def test_pdb_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "pdb_search_hemoglobin.json").read_text(encoding="utf-8")
        )
        with patch.object(pdb, "fetch_search_results", return_value=fixture):
            rows = pdb.search_pdb("hemoglobin")
        self.assertEqual(rows[0]["_adapter"], "pdb")
        self.assertEqual(rows[0]["document_type"], "protein_structure")

    def test_chemrxiv_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "chemrxiv_search_battery.json").read_text(encoding="utf-8")
        )
        with patch.object(chemrxiv, "fetch_search_results", return_value=fixture):
            rows = chemrxiv.search_chemrxiv("lithium battery")
        self.assertEqual(rows[0]["_adapter"], "chemrxiv")
        self.assertEqual(rows[0]["document_type"], "preprint")


if __name__ == "__main__":
    unittest.main()
