"""Tests for Slice 15 adapters (OpenReview, ACL Anthology)."""

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

from core.knowledge.adapters import acl_anthology, openreview  # noqa: E402
from core.knowledge.adapters.registry import get_search_function  # noqa: E402
from core.knowledge.provider_credentials import adapter_credentials_hint  # noqa: E402
from core.knowledge.scientific_discipline import (  # noqa: E402
    SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE,
    preferred_adapters_for_discipline,
)

_FIXTURES = Path(_WS_ROOT) / "eval" / "fixtures" / "knowledge"

_SLICE15_ADAPTER_IDS = ("openreview", "acl_anthology")


class TestSlice15Registry(unittest.TestCase):
    def test_adapters_registered(self) -> None:
        for adapter_id in _SLICE15_ADAPTER_IDS:
            self.assertIsNotNone(get_search_function(adapter_id), adapter_id)

    def test_anonymous_adapters_have_no_required_hint(self) -> None:
        for adapter_id in _SLICE15_ADAPTER_IDS:
            self.assertIsNone(adapter_credentials_hint(adapter_id), adapter_id)

    def test_discipline_pack_updates(self) -> None:
        cs_order = preferred_adapters_for_discipline(SCIENTIFIC_DISCIPLINE_COMPUTER_SCIENCE)
        self.assertIn("openreview", cs_order)
        self.assertIn("acl_anthology", cs_order)


class TestSlice15AdapterFixtures(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUBE_KNOWLEDGE_FIXTURES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("QUBE_KNOWLEDGE_FIXTURES", None)

    def test_openreview_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "openreview_search_transformer.json").read_text(encoding="utf-8")
        )
        with patch.object(openreview, "fetch_search_results", return_value=fixture):
            rows = openreview.search_openreview("transformer")
        self.assertEqual(rows[0]["_adapter"], "openreview")
        self.assertIn(rows[0]["document_type"], ("conference_paper", "preprint"))

    def test_acl_anthology_fixture_search(self) -> None:
        fixture = json.loads(
            (_FIXTURES / "acl_anthology_search_transformer.json").read_text(encoding="utf-8")
        )
        with patch.object(acl_anthology, "fetch_search_results", return_value=fixture):
            rows = acl_anthology.search_acl_anthology("transformer")
        self.assertEqual(rows[0]["_adapter"], "acl_anthology")
        self.assertEqual(rows[0]["document_type"], "conference_paper")


if __name__ == "__main__":
    unittest.main()
