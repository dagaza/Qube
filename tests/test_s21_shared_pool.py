"""Plan S21: shared-pool scoring for runs that rank different pools.

``ndcg_at_5`` and ``relevant_recall`` are scored against the pool a run ranked,
so a run with a deeper pool or another routing arm is measured on another
scale. Shared-pool scoring re-scores every run against the union of the runs'
pools (predecessor S8 results, the "like-for-like" table).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import quality_eval as qe  # noqa: E402
from core.knowledge.scientific_adapters import ROUTE_NO_DISCIPLINE, ROUTE_PRODUCTION  # noqa: E402

BASELINES = ROOT / "eval" / "retrieval_corpus" / "baselines"
CORPUS = "eval/retrieval_corpus/v1_knowledge_quality.json"


def _hit(adapter: str, rank: int, query: str = "production", **extra) -> dict:
    return {"query": query, "adapter": adapter, "rank": rank, **extra}


def _row(title: str, label: str | None, *hits: dict) -> dict:
    return {"title": title, "snippet": title, "label": label, "retrieved": list(hits)}


def _stub_ranker(keep_titles: list[str]):
    def rank(query: str, rows: list[dict], budget: int, path: str = "") -> qe.RankerOutput:
        kept = [r for r in rows if r["title"] in keep_titles][:budget]
        return qe.RankerOutput(kept=kept, rejected=[r for r in rows if r not in kept])

    return rank


def _depth_fixture() -> dict:
    """Two relevant rows; the second is beyond the production depth."""
    return {
        "id": "depth",
        "path": "general_web",
        "category": "how_to",
        "query_clean": "q",
        "budget": 3,
        "production_depth": {"ddg": 2},
        "candidates": [
            _row("rel near", "relevant", _hit("ddg", 0)),
            _row("irr near", "irrelevant", _hit("ddg", 1)),
            _row("rel deep", "relevant", _hit("ddg", 2)),
            _row("part deep", "partial", _hit("ddg", 3)),
        ],
    }


def _routed_fixture() -> dict:
    return {
        "id": "routed",
        "path": "scientific_evidence",
        "category": "heuristic_trigger",
        "query_clean": "q",
        "production_depth": {"a": 2, "b": 2, "c": 2},
        "routes": {"production": {ROUTE_PRODUCTION: ["b", "a"], ROUTE_NO_DISCIPLINE: ["c", "a"]}},
        "candidates": [
            _row("a0", "partial", _hit("a", 0)),
            _row("b0", "irrelevant", _hit("b", 0)),
            _row("c0", "relevant", _hit("c", 0, route_added=True)),
            _row("a2", "relevant", _hit("a", 2)),  # beyond depth: in no pool
        ],
    }


def _run(fixture: dict, keep: list[str], **pool_kw) -> dict:
    report = qe.evaluate_fixture(fixture, ranker=_stub_ranker(keep), **pool_kw)
    report["id"] = fixture["id"]
    return report


def _summary(results: list[dict], **settings) -> dict:
    return {"query_mode": "production", "pool_mode": "production", "include_synthetic": False,
            **settings, "results": results}


class SharedPoolTests(unittest.TestCase):
    def test_union_of_the_runs_pools(self) -> None:
        fx = _routed_fixture()
        pool = qe.shared_pool(fx, [{"route_arm": ROUTE_PRODUCTION}, {"route_arm": ROUTE_NO_DISCIPLINE}])
        self.assertEqual([r["title"] for r in pool], ["a0", "b0", "c0"])

    def test_one_run_shares_its_own_pool(self) -> None:
        fx = _depth_fixture()
        for mode in qe.POOL_MODES:
            with self.subTest(pool_mode=mode):
                own = qe.fixture_pool(fx, pool_mode=mode)
                shared = qe.shared_pool(fx, [{"pool_mode": mode}])
                self.assertEqual(sorted(r["title"] for r in shared), sorted(r["title"] for r in own))

    def test_rows_are_copies(self) -> None:
        fx = _depth_fixture()
        qe.shared_pool(fx, [{}])[0]["title"] = "mutated"
        self.assertEqual(fx["candidates"][0]["title"], "rel near")

    def test_reports_carry_kept_labels(self) -> None:
        rep = _run(_depth_fixture(), ["irr near", "rel near"])
        self.assertEqual(rep["kept_labels"], ["relevant", "irrelevant"])


class RescoreTests(unittest.TestCase):
    def test_rescoring_on_the_own_pool_reproduces_the_report(self) -> None:
        fx = _depth_fixture()
        for mode, keep in (("production", ["rel near"]), ("full", ["rel deep", "part deep", "irr near"])):
            with self.subTest(pool_mode=mode):
                rep = _run(fx, keep, pool_mode=mode)
                pool = qe.fixture_pool(fx, pool_mode=mode)
                again = qe.rescore_on_pool(fx, rep, rep["kept_labels"], pool)
                for metric in qe.POOL_RELATIVE_METRICS:
                    self.assertEqual(again[metric], rep[metric], metric)

    def test_deeper_pool_no_longer_looks_like_a_regression(self) -> None:
        fx = _depth_fixture()
        shallow = _run(fx, ["rel near"], pool_mode="production")
        deep = _run(fx, ["rel near", "rel deep"], pool_mode="full")
        # Own pools: the shallow run found its only relevant row, so both score 1.0
        # recall, while nDCG penalises the deep run for the partial row it left out.
        self.assertEqual((shallow["relevant_recall"], deep["relevant_recall"]), (1.0, 1.0))
        self.assertGreater(shallow["ndcg_at_5"], deep["ndcg_at_5"])
        (s_cur, s_base), skipped = qe.shared_pool_reports(
            {"depth": fx}, [_summary([deep], pool_mode="full"), _summary([shallow])]
        )
        self.assertEqual(skipped, [])
        self.assertEqual(s_base[0]["relevant_recall"], 0.5)
        self.assertEqual(s_cur[0]["relevant_recall"], 1.0)
        self.assertGreater(s_cur[0]["ndcg_at_5"], s_base[0]["ndcg_at_5"])
        self.assertEqual(s_cur[0]["shared_pool_size"], 4)

    def test_derived_expect_follows_the_shared_pool(self) -> None:
        fx = _routed_fixture()
        prod = _run(fx, [], route_arm=ROUTE_PRODUCTION)
        general = _run(fx, ["c0"], route_arm=ROUTE_NO_DISCIPLINE)
        self.assertEqual((prod["expect"], prod["abstain_correct"]), ("partial", False))
        (s_prod, s_gen), _ = qe.shared_pool_reports(
            {"routed": fx},
            [_summary([dict(prod, route_arm=ROUTE_PRODUCTION)]),
             _summary([dict(general, route_arm=ROUTE_NO_DISCIPLINE)])],
        )
        self.assertEqual(s_prod[0]["expect"], "partial")
        self.assertFalse(s_prod[0]["abstain_correct"])
        self.assertTrue(s_gen[0]["abstain_correct"])
        self.assertEqual(s_gen[0]["relevant_recall"], 1.0)

    def test_fixture_missing_from_a_run_is_not_scored(self) -> None:
        fx = _depth_fixture()
        rep = _run(fx, ["rel near"])
        (cur, base), skipped = qe.shared_pool_reports({"depth": fx}, [_summary([rep]), _summary([])])
        self.assertEqual((cur, base, skipped), ([], [], []))

    def test_changed_fixture_is_skipped_with_a_reason(self) -> None:
        fx = _depth_fixture()
        rep = _run(fx, ["rel near"])
        rep["pool_size"] = 5
        (cur, _), skipped = qe.shared_pool_reports({"depth": fx}, [_summary([rep]), _summary([rep])])
        self.assertEqual(cur, [])
        self.assertEqual(skipped[0]["id"], "depth")
        self.assertIn("ranked 5", skipped[0]["reason"])


class KeptLabelsTests(unittest.TestCase):
    """Stored baselines before S21 carry kept titles only."""

    def _twins(self) -> dict:
        return {
            "id": "twins", "path": "trusted_knowledge", "query_clean": "q", "budget": 3,
            "production_depth": {"ddg": 4},
            "candidates": [
                _row("Same title", "relevant", _hit("ddg", 0)),
                _row("Same title", "partial", _hit("ddg", 1)),
                _row("other", "irrelevant", _hit("ddg", 2)),
            ],
        }

    def _stored(self, fx: dict, kept_index: int) -> dict:
        pool = qe.fixture_pool(fx)
        kept = [pool[kept_index], pool[2]]
        rep = {k: v for k, v in _run(fx, []).items() if k != "kept_labels"}
        rep.update(
            kept_titles=[r["title"] for r in kept],
            precision_at_k=qe.precision_at_k(kept, 3),
            ndcg_at_5=qe.ndcg_at_k(kept, pool, 5),
            irrelevant_kept_rate=qe.irrelevant_kept_rate(kept),
            relevant_recall=qe.relevant_recall(kept, pool, budget=3),
        )
        return rep

    def test_labels_field_is_used_when_present(self) -> None:
        self.assertEqual(qe.kept_labels({"kept_labels": ["partial"], "kept_titles": ["x"]}, []), ["partial"])

    def test_title_shared_by_two_labels_is_settled_by_the_stored_metrics(self) -> None:
        fx = self._twins()
        pool = qe.fixture_pool(fx)
        self.assertEqual(qe.kept_labels(self._stored(fx, 0), pool), ["relevant", "irrelevant"])
        self.assertEqual(qe.kept_labels(self._stored(fx, 1), pool), ["partial", "irrelevant"])

    def test_unsettled_or_missing_title_raises(self) -> None:
        fx = self._twins()
        pool = qe.fixture_pool(fx)
        stored = self._stored(fx, 0)
        stored["ndcg_at_5"] = 0.123
        with self.assertRaises(ValueError):
            qe.kept_labels(stored, pool)
        with self.assertRaises(ValueError):
            qe.kept_labels({"kept_titles": ["not in pool"]}, pool)
        with self.assertRaises(ValueError):
            qe.kept_labels({"kept_titles": ["other", "other"]}, pool)


class CommittedBaselineTests(unittest.TestCase):
    """Acceptance: the stored runs re-score faithfully and reproduce the S8 table."""

    @classmethod
    def setUpClass(cls) -> None:
        corpus = qe.load_json(CORPUS)
        cls.fixtures = {e["id"]: qe.load_json(e["fixture"]) for e in corpus["queries"] if e.get("fixture")}

    def _load(self, name: str) -> dict:
        return json.loads((BASELINES / name).read_text(encoding="utf-8"))

    def test_s13_baselines_rescored_on_their_own_pools_are_unchanged(self) -> None:
        for ranker in ("token", "embed", "ce"):
            for mode in ("production", "standalone"):
                with self.subTest(ranker=ranker, mode=mode):
                    run = self._load(f"kq_s13_{ranker}_{mode}.json")
                    (again, _), skipped = qe.shared_pool_reports(self.fixtures, [run, run])
                    self.assertEqual(skipped, [])
                    self.assertEqual(len(again), run["evaluated"])
                    stored = {r["id"]: r for r in run["results"]}
                    for rep in again:
                        for metric in qe.POOL_RELATIVE_METRICS:
                            self.assertTrue(
                                qe._same_metric(rep[metric], stored[rep["id"]][metric]), (rep["id"], metric)
                            )

    def test_reproduces_the_s8_like_for_like_table(self) -> None:
        # Predecessor "S8 results": S7's and S8's kept sets on the same 8-row web pools.
        expected = {"production": ((0.62, 0.64), (0.75, 0.79)), "standalone": ((0.61, 0.64), (0.76, 0.82))}
        for mode, (ndcg, recall) in expected.items():
            with self.subTest(mode=mode):
                s7 = self._load(f"kq_s7_ce_{mode}.json")
                s8 = self._load(f"kq_s8_ce_{mode}.json")
                (r7, r8), skipped = qe.shared_pool_reports(self.fixtures, [s7, s8])
                self.assertEqual(skipped, [])
                a7 = qe.aggregate_reports(r7, group_by="path")["general_web"]
                a8 = qe.aggregate_reports(r8, group_by="path")["general_web"]
                self.assertEqual(a8["fixtures"], 20)
                self.assertEqual((round(a7["ndcg_at_5"], 2), round(a8["ndcg_at_5"], 2)), ndcg)
                self.assertEqual((round(a7["relevant_recall"], 2), round(a8["relevant_recall"], 2)), recall)
                # Each scored on its own pool, S8 looks worse (the false regression).
                self.assertGreater(s7["aggregates"]["path"]["general_web"]["ndcg_at_5"], 0.9)
                self.assertLess(s8["aggregates"]["path"]["general_web"]["ndcg_at_5"], 0.7)


class CliTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "evaluate_knowledge_quality.py"), "--corpus", CORPUS, *args],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )

    def test_shared_scoring_needs_a_baseline(self) -> None:
        proc = self._run("--pool-scoring", "shared")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("needs --compare", proc.stderr)

    def test_compare_on_shared_pools(self) -> None:
        # The S13 token baseline is today's token run, so shared scoring finds no
        # regression and scores every fixture.
        proc = self._run("--json", "--compare", str(BASELINES / "kq_s13_token_production.json"),
                         "--pool-scoring", "shared")
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertEqual(summary["pool_scoring"], "shared")
        shared = summary["shared_pool"]
        self.assertEqual((shared["fixtures"], shared["skipped"]), (summary["evaluated"], []))
        self.assertEqual(summary["regressions"], [])
        cur, base = shared["current"]["all"]["all"], shared["baseline_aggregates"]["all"]["all"]
        for metric in qe.POOL_RELATIVE_METRICS:
            self.assertTrue(qe._same_metric(cur[metric], base[metric]), metric)

    def test_default_output_has_no_shared_section(self) -> None:
        proc = self._run("--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertNotIn("shared_pool", summary)
        self.assertTrue(all("kept_labels" in r for r in summary["results"] if not r.get("skipped")))


class HeuristicAbSharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("evaluate_heuristic_ab_s21", ROOT / "tools" / "evaluate_heuristic_ab.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.rows = module.run(["token"], ["production"])
        cls.summary = module.summarise(cls.rows)

    def test_every_row_carries_shared_scores(self) -> None:
        self.assertTrue(self.rows)
        for r in self.rows:
            self.assertEqual(set(r["shared"]), {"pool_size", "production", "off"})

    def test_ranking_ablation_shares_the_pool_it_ranked(self) -> None:
        corpus = qe.load_json(CORPUS)
        fixtures = {e["id"]: qe.load_json(e["fixture"]) for e in corpus["queries"] if e.get("fixture")}
        for r in (r for r in self.rows if r["heuristic"] == "trial_grounding"):
            fx = fixtures[r["id"]]
            arm = ROUTE_PRODUCTION if qe.fixture_has_routes(fx) else None
            own = qe.evaluate_fixture(fx, ranker="token", route_arm=arm)
            self.assertEqual(r["shared"]["pool_size"], own["pool_size"], r["id"])
            self.assertEqual(r["shared"]["production"]["relevant_recall"], own["relevant_recall"], r["id"])

    def test_routing_arm_pool_holds_both_arms(self) -> None:
        routing = [r for r in self.rows if r["heuristic"] == "discipline_routing"]
        self.assertTrue(routing)
        for r in routing:
            self.assertGreaterEqual(r["shared"]["pool_size"], max(r["production"]["pool_size"], r["off"]["pool_size"]))
        s = self.summary["discipline_routing"]["token/production"]
        self.assertIsNotNone(s["shared_recall_on"])
        self.assertIsNotNone(s["shared_recall_off"])


if __name__ == "__main__":
    unittest.main()
