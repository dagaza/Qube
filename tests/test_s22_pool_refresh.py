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
import json
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

CORPUS = "eval/retrieval_corpus/v1_knowledge_quality.json"
BASELINES = ROOT / "eval" / "retrieval_corpus" / "baselines"


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
        rep = qe.evaluate_fixture(fx, ranker="token", as_of="S22")
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


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = qe.load_json(CORPUS)
        cls.entries = [e for e in corpus["queries"] if e.get("fixture")]
        cls.fixtures = {e["id"]: qe.load_json(e["fixture"]) for e in cls.entries}

    def test_the_s22_refresh_moved_the_17_who_hits_out(self) -> None:
        moved: dict[str, int] = {}
        for fid, fx in self.fixtures.items():
            for cand in fx.get("candidates") or []:
                for hit in cand.get("retrieved") or []:
                    for move in hit.get("pool_moves") or ():
                        if move.get("refresh") != "S22":
                            continue  # later refreshes (S23) are tested with their stage
                        self.assertEqual((hit["adapter"], move), ("who", {"refresh": "S22", "moved": "out"}))
                        moved[fid] = moved.get(fid, 0) + 1
        self.assertEqual(
            moved, {"kq_ace_hf_single": 8, "sci_abstain_fictional_drug": 6, "s13_qt_trial_registry_psilocybin": 3}
        )

    def test_every_committed_baseline_rescores_on_its_own_pool(self) -> None:
        files = sorted(BASELINES.glob("kq_*.json"))
        self.assertGreaterEqual(len(files), 64)
        for path in files:
            with self.subTest(baseline=path.name):
                run = json.loads(path.read_text(encoding="utf-8"))
                (again, _), skipped = qe.shared_pool_reports(self.fixtures, [run, run])
                self.assertEqual(skipped, [])
                self.assertEqual(len(again), run["evaluated"])
                stored = {r["id"]: r for r in run["results"]}
                for rep in again:
                    for metric in qe.POOL_RELATIVE_METRICS:
                        self.assertTrue(qe._same_metric(rep[metric], stored[rep["id"]][metric]), (rep["id"], metric))

    def test_s22_token_baselines_are_the_replay_as_of_s22(self) -> None:
        # S23's refresh (D40) changed today's pools; ``as_of="S22"`` replays them
        # as S22 left them.
        for mode in ("production", "standalone"):
            with self.subTest(mode=mode):
                base = json.loads((BASELINES / f"kq_s22_token_{mode}.json").read_text(encoding="utf-8"))
                stored = {r["id"]: r for r in base["results"]}
                for entry in self.entries:
                    rep = qe.evaluate_fixture(
                        self.fixtures[entry["id"]], ranker="token", query_mode=mode, as_of="S22"
                    )
                    want = stored[entry["id"]]
                    self.assertEqual(rep["kept_titles"], want["kept_titles"], entry["id"])
                    self.assertEqual(rep["pool_size"], want["pool_size"], entry["id"])
                    self.assertEqual(want["pool_refresh"], "S22")

    def test_who_rows_left_three_pools(self) -> None:
        s21 = json.loads((BASELINES / "kq_s21_token_production.json").read_text(encoding="utf-8"))
        s22 = json.loads((BASELINES / "kq_s22_token_production.json").read_text(encoding="utf-8"))
        before = {r["id"]: r["pool_size"] for r in s21["results"] if not r.get("skipped")}
        after = {r["id"]: r["pool_size"] for r in s22["results"] if not r.get("skipped")}
        shrunk = {fid: (before[fid], after[fid]) for fid in before if before[fid] != after[fid]}
        self.assertEqual(
            shrunk,
            {"kq_ace_hf_single": (15, 12), "sci_abstain_fictional_drug": (8, 5),
             "s13_qt_trial_registry_psilocybin": (12, 9)},
        )


if __name__ == "__main__":
    unittest.main()
