"""Plan S23 (D40): Crossref joins the physics pack, and only that pack.

In the S13 A/B, all five discipline-routing losses with the reranker were
Crossref rows that only the general route reached (Bell tests, QE, euro-area
unemployment, DAPA-HF, PARADIGM-HF). Measured on shared pools against
``kq_s22_*``, adding Crossref to physics fixes the Bell-test loss and changes
nothing else. Adding it to economics also fixes two CE losses but costs the
token fallback 2 fixtures. Adding it to medicine makes the fictional-drug
abstain case ``partial`` with the reranker. Both fail rule 7, so they were not
taken.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import quality_eval as qe  # noqa: E402
from core.knowledge.scientific_adapters import apply_scientific_adapter_policy  # noqa: E402
from core.knowledge.scientific_discipline import preferred_adapters_for_discipline  # noqa: E402
from core.knowledge.source_preferences import get_effective_enabled_adapters  # noqa: E402
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402

CORPUS = "eval/retrieval_corpus/v1_knowledge_quality.json"
BASELINES = ROOT / "eval" / "retrieval_corpus" / "baselines"
# Packs that already listed Crossref before S23 (plan S8).
_HAD_CROSSREF = {"psychology", "sociology", "political_science", "general_science"}


class PackTests(unittest.TestCase):
    def test_crossref_is_the_last_physics_fallback(self) -> None:
        self.assertEqual(preferred_adapters_for_discipline("physics")[-1], "crossref")

    def test_no_other_pack_gained_crossref(self) -> None:
        for pid in ("medicine", "biology", "chemistry", "computer_science", "economics", "earth_environment", "engineering"):
            with self.subTest(pack=pid):
                self.assertNotIn("crossref", preferred_adapters_for_discipline(pid))
        for pid in _HAD_CROSSREF:
            with self.subTest(pack=pid):
                self.assertIn("crossref", preferred_adapters_for_discipline(pid))

    def test_a_physics_question_routes_to_crossref(self) -> None:
        enabled = get_effective_enabled_adapters(SERVICE_SCIENTIFIC_EVIDENCE, stored_preferences={})
        route = apply_scientific_adapter_policy(
            enabled, query="What do loophole-free Bell test experiments show about quantum entanglement?"
        )
        self.assertEqual(route, ("arxiv", "inspire_hep", "openalex", "crossref"))


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = qe.load_json(CORPUS)
        cls.entries = [e for e in corpus["queries"] if e.get("fixture")]
        cls.fixtures = {e["id"]: qe.load_json(e["fixture"]) for e in cls.entries}

    def test_the_s23_refresh_moved_six_crossref_hits_in(self) -> None:
        moved: dict[str, int] = {}
        for fid, fx in self.fixtures.items():
            for cand in fx.get("candidates") or []:
                for hit in cand.get("retrieved") or []:
                    for move in hit.get("pool_moves") or ():
                        if move.get("refresh") != "S23":
                            continue
                        self.assertEqual((hit["adapter"], hit["query"], move["moved"]), ("crossref", "production", "in"))
                        moved[fid] = moved.get(fid, 0) + 1
        self.assertEqual(moved, {"s13_phys_bell_tests": 3, "s13_phys_ligo_detection": 3})

    def test_physics_pools_before_and_after_s23(self) -> None:
        for fid in ("s13_phys_bell_tests", "s13_phys_ligo_detection"):
            with self.subTest(fixture=fid):
                fx = self.fixtures[fid]
                before = {r.get("_adapter") for r in qe.fixture_pool(fx, as_of="S22")}
                after = {r.get("_adapter") for r in qe.fixture_pool(fx)}
                self.assertNotIn("crossref", before)
                self.assertEqual(after - before, {"crossref"})

    def test_s23_token_baselines_are_todays_replay(self) -> None:
        for mode in ("production", "standalone"):
            with self.subTest(mode=mode):
                base = json.loads((BASELINES / f"kq_s23_token_{mode}.json").read_text(encoding="utf-8"))
                stored = {r["id"]: r for r in base["results"]}
                for entry in self.entries:
                    rep = qe.evaluate_fixture(self.fixtures[entry["id"]], ranker="token", query_mode=mode)
                    want = stored[entry["id"]]
                    self.assertEqual(rep["kept_titles"], want["kept_titles"], entry["id"])
                    self.assertEqual(rep["pool_size"], want["pool_size"], entry["id"])
                    self.assertEqual(want["pool_refresh"], "S23")


if __name__ == "__main__":
    unittest.main()
