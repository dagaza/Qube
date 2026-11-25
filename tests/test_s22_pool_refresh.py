"""Plan S22 (D39): a route refresh stamps the hits it moves.

S22 dropped WHO from the medicine pack, and ``--refresh-routes`` moved its 17
captured hits out of the default pool. Without a record of that move, no run
stored before S22 could rebuild the pool it ranked, so shared-pool scoring
skipped those fixtures. Each moved hit now records the refresh in
``pool_moves``; reports record ``pool_refresh``; ``as_of`` rebuilds a pool as
it stood after a given refresh (``None``: before every stamped one).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import quality_eval as qe  # noqa: E402
from core.knowledge.scientific_adapters import ROUTE_NO_DISCIPLINE, ROUTE_PRODUCTION  # noqa: E402


def _load_capture_tool():
    spec = importlib.util.spec_from_file_location(
        "capture_knowledge_fixture_s22", ROOT / "tools" / "capture_knowledge_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(title: str, adapter: str, rank: int, *, label: str = "relevant", added: bool = False) -> dict:
    hit = {"query": "production", "adapter": adapter, "rank": rank}
    if added:
        hit["route_added"] = True
    return {"title": title, "label": label, "label_source": "model", "retrieved": [hit]}


class RouteAddedAsOfTests(unittest.TestCase):
    def test_a_hit_never_moved_keeps_its_flag(self) -> None:
        for as_of in (None, "S22"):
            self.assertTrue(qe.route_added_as_of({"route_added": True}, as_of))
            self.assertFalse(qe.route_added_as_of({}, as_of))

    def test_moved_out_is_in_the_pool_before_the_refresh(self) -> None:
        hit = {"route_added": True, "pool_moves": [{"refresh": "S22", "moved": "out"}]}
        self.assertFalse(qe.route_added_as_of(hit, None))
        self.assertTrue(qe.route_added_as_of(hit, "S22"))
        self.assertTrue(qe.route_added_as_of(hit))

    def test_moved_in_is_out_of_the_pool_before_the_refresh(self) -> None:
        hit = {"pool_moves": [{"refresh": "S22", "moved": "in"}]}
        self.assertTrue(qe.route_added_as_of(hit, None))
        self.assertFalse(qe.route_added_as_of(hit, "S22"))

    def test_the_last_move_at_or_before_the_refresh_decides(self) -> None:
        hit = {"pool_moves": [{"refresh": "S22", "moved": "out"}, {"refresh": "S99", "moved": "in"}]}
        with patch.object(qe, "POOL_REFRESHES", ("S22", "S99")):
            self.assertFalse(qe.route_added_as_of(hit, None))
            self.assertTrue(qe.route_added_as_of(hit, "S22"))
            self.assertFalse(qe.route_added_as_of(hit, "S99"))

    def test_unknown_refresh_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            qe.route_added_as_of({"pool_moves": [{"refresh": "S22", "moved": "out"}]}, "S1")

    def test_stored_runs_name_the_pool_they_ranked(self) -> None:
        self.assertIsNone(qe.run_pool_spec({}, {"query_mode": "production"})["as_of"])
        self.assertEqual(qe.run_pool_spec({}, {"pool_refresh": "S22"})["as_of"], "S22")
        self.assertEqual(qe.CURRENT_POOL_REFRESH, qe.POOL_REFRESHES[-1])


class RefreshStampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_capture_tool()

    def _fixture(self) -> dict:
        return {
            "id": "s22_refresh",
            "path": "scientific_evidence",
            "budget": 3,
            "retrieval_query": "q",
            "production_depth": {"who": 3, "pubmed": 3, "crossref": 3},
            "adapter_queries": {"production": {"who": "q", "pubmed": "q", "crossref": "q"}},
            "routes": {"production": {ROUTE_PRODUCTION: ["pubmed", "who"], ROUTE_NO_DISCIPLINE: ["pubmed", "crossref"]}},
            "candidates": [
                _row("who0", "who", 0, label="irrelevant"),
                _row("pm0", "pubmed", 0),
                _row("cr0", "crossref", 0, added=True),
            ],
        }

    def _refresh(self, fx: dict, production: list[str], stage: str | None) -> dict:
        routes = {ROUTE_PRODUCTION: production, ROUTE_NO_DISCIPLINE: ["pubmed", "crossref"]}
        with patch.object(self.tool, "_scientific_routes", lambda _q: {k: list(v) for k, v in routes.items()}):
            return self.tool.refresh_routes(fx, stage=stage)

    def test_moves_are_stamped_and_older_pools_rebuild(self) -> None:
        fx = self._fixture()
        counts = self._refresh(fx, ["pubmed", "crossref"], "S22")
        self.assertEqual(counts, {"left_pool": 1, "joined_pool": 1})
        hits = {c["title"]: c["retrieved"][0] for c in fx["candidates"]}
        self.assertEqual(hits["who0"]["pool_moves"], [{"refresh": "S22", "moved": "out"}])
        self.assertEqual(hits["cr0"]["pool_moves"], [{"refresh": "S22", "moved": "in"}])
        self.assertNotIn("pool_moves", hits["pm0"])
        self.assertEqual([r["title"] for r in qe.fixture_pool(fx)], ["pm0", "cr0"])
        self.assertEqual([r["title"] for r in qe.fixture_pool(fx, as_of=None)], ["who0", "pm0"])
        rep = qe.evaluate_fixture(fx, ranker="token")
        self.assertEqual((rep["pool_refresh"], rep["pool_size"]), ("S22", 2))

    def test_unstamped_refresh_records_nothing(self) -> None:
        fx = self._fixture()
        self._refresh(fx, ["pubmed"], None)
        self.assertFalse(any("pool_moves" in h for c in fx["candidates"] for h in c["retrieved"]))

    def test_cli_requires_a_known_stage(self) -> None:
        for extra in ([], ["--refresh-stage", "S1"]):
            with self.subTest(extra=extra):
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / "capture_knowledge_fixture.py"), "--refresh-routes",
                     "--path", "scientific_evidence", *extra],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("needs --refresh-stage", proc.stderr)


if __name__ == "__main__":
    unittest.main()
