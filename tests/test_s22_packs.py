"""Plan S22 (D36–D38): institutional adapters that returned 0 rows (V30).

Live diagnosis (2026-11-25): PatentsView's PatentSearch host no longer
resolves, BLS has no series search and Eurostat's discovery endpoint returns
404, so all three are retired. WHO GHO (health indicators, first-word query)
and NIST NVD (a CVE database) leave their packs but stay registered for
explicit selection.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.adapter_key_gate import split_keyless  # noqa: E402
from core.knowledge.adapter_readiness import get_adapter_readiness_meta  # noqa: E402
from core.knowledge.adapters.catalog import ADAPTER_CATALOG  # noqa: E402
from core.knowledge.adapters.registry import RETIRED_ADAPTERS, SEARCH_FUNCTIONS, get_search_function  # noqa: E402
from core.knowledge.provider_credentials import adapter_credentials_hint  # noqa: E402
from core.knowledge.scientific_discipline import preferred_adapters_for_discipline  # noqa: E402
from core.knowledge.scientific_discipline_packs import SCIENTIFIC_DISCIPLINE_PACKS  # noqa: E402
from core.knowledge.source_preferences import get_effective_enabled_adapters  # noqa: E402
from core.knowledge.types import SERVICE_FINANCE_KNOWLEDGE, SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402

RETIRED = frozenset({"bls", "eurostat", "uspto_patentsview"})


class RetiredAdapterTests(unittest.TestCase):
    def test_retired_adapters_are_not_registered(self) -> None:
        self.assertEqual(RETIRED_ADAPTERS, RETIRED)
        for aid in RETIRED:
            with self.subTest(adapter=aid):
                self.assertNotIn(aid, SEARCH_FUNCTIONS)
                self.assertIsNone(get_search_function(aid))
                self.assertTrue((ROOT / "core" / "knowledge" / "adapters" / f"{aid}.py").is_file())

    def test_no_catalog_entry_offers_them(self) -> None:
        entries = [e for e in ADAPTER_CATALOG if e.id in RETIRED]
        self.assertEqual(len(entries), 7)
        for entry in entries:
            with self.subTest(adapter=entry.id, service=entry.knowledge_service, group=entry.ui_group):
                self.assertFalse(entry.implemented)
                self.assertFalse(entry.default_enabled)

    def test_readiness_says_why(self) -> None:
        for aid in RETIRED:
            with self.subTest(adapter=aid):
                meta = get_adapter_readiness_meta(aid)
                self.assertEqual(meta.readiness, "stub")
                self.assertIn("Retired in plan S22", meta.production_strategy)
                self.assertIsNone(adapter_credentials_hint(aid))

    def test_stored_preferences_cannot_enable_them(self) -> None:
        for service in (SERVICE_SCIENTIFIC_EVIDENCE, SERVICE_FINANCE_KNOWLEDGE):
            with self.subTest(service=service):
                enabled = get_effective_enabled_adapters(
                    service, stored_preferences={service: ["eurostat", "bls", "uspto_patentsview", "world_bank"]}
                )
                self.assertFalse(RETIRED & set(enabled))
                self.assertFalse(RETIRED & set(get_effective_enabled_adapters(service, stored_preferences={})))

    def test_no_pack_lists_them(self) -> None:
        for pack in SCIENTIFIC_DISCIPLINE_PACKS:
            with self.subTest(pack=pack.id):
                self.assertFalse(RETIRED & set(pack.resolved_adapter_order()))


class PackTests(unittest.TestCase):
    def test_who_and_nist_left_their_packs_but_stay_selectable(self) -> None:
        self.assertNotIn("who", preferred_adapters_for_discipline("medicine"))
        self.assertNotIn("nist", preferred_adapters_for_discipline("engineering"))
        for pack in SCIENTIFIC_DISCIPLINE_PACKS:
            with self.subTest(pack=pack.id):
                self.assertFalse({"who", "nist"} & set(pack.resolved_adapter_order()))
        for aid in ("who", "nist"):
            self.assertIsNotNone(get_search_function(aid))
            self.assertIn(aid, get_effective_enabled_adapters(SERVICE_SCIENTIFIC_EVIDENCE, stored_preferences={}))

    def test_routes_after_s22(self) -> None:
        enabled = set(get_effective_enabled_adapters(SERVICE_SCIENTIFIC_EVIDENCE, stored_preferences={}))
        routes = {
            pid: tuple(a for a in preferred_adapters_for_discipline(pid) if a in enabled)
            for pid in ("medicine", "chemistry", "economics", "engineering")
        }
        self.assertEqual(
            routes,
            {
                "medicine": ("pubmed", "nice", "cdc", "clinicaltrials_gov", "openfda", "openalex", "europe_pmc"),
                "chemistry": ("pubchem", "chembl", "chemrxiv", "epo_espacenet", "openalex", "pubmed"),
                "economics": ("repec", "world_bank", "oecd", "openalex"),
                "engineering": ("ieee_xplore", "ietf_rfc", "epo_espacenet", "arxiv", "openalex"),
            },
        )

    def test_keyless_engineering_searches_ietf_arxiv_and_openalex(self) -> None:
        # Acceptance: the engineering pack's primaries are IEEE Xplore (key) and
        # IETF RFC; a keyless turn still has IETF RFC plus the fallbacks.
        clean = {k: v for k, v in os.environ.items() if not k.startswith("QUBE_") or not k.endswith(("_KEY", "_SECRET", "_TOKEN"))}
        clean.pop("QUBE_KNOWLEDGE_FIXTURES", None)
        enabled = set(get_effective_enabled_adapters(SERVICE_SCIENTIFIC_EVIDENCE, stored_preferences={}))
        route = tuple(a for a in preferred_adapters_for_discipline("engineering") if a in enabled)
        with patch.dict(os.environ, clean, clear=True), patch(
            "core.knowledge.credentials._user_secret", return_value=None
        ):
            runnable, skips = split_keyless(route)
        self.assertEqual(runnable, ("ietf_rfc", "arxiv", "openalex"))
        self.assertEqual([s["adapter"] for s in skips], ["ieee_xplore", "epo_espacenet"])


if __name__ == "__main__":
    unittest.main()
