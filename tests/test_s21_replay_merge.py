"""Plan S21 (D34): the scientific replay merges its pool as production does.

Production fuses scholarly rows that share a DOI or normalised title
(``adapter_collection.rrf_merge``). Fixtures capture such rows as separate
candidates when their DOI or URL differ, so the replay drops the ones the merge
fuses into an earlier row before ranking (V54).
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
from core.knowledge.adapter_collection import first_seen_unique, merge_keys, rrf_merge  # noqa: E402
from core.knowledge.scientific_adapters import ROUTE_ARMS, ROUTE_NO_DISCIPLINE, ROUTE_PRODUCTION  # noqa: E402

BASELINES = ROOT / "eval" / "retrieval_corpus" / "baselines"
CORPUS = "eval/retrieval_corpus/v1_knowledge_quality.json"


def _row(title: str, adapter: str, rank: int, label: str = "relevant", **extra) -> dict:
    hit = {"query": "production", "adapter": adapter, "rank": rank}
    if extra.pop("added", False):
        hit["route_added"] = True
    return {"title": title, "snippet": title, "label": label, "_adapter": adapter, "retrieved": [hit], **extra}


def _fixture(path: str = "scientific_evidence") -> dict:
    return {
        "id": "dup",
        "path": path,
        "category": "physics",
        "query_clean": "q",
        "production_depth": {"openalex": 3, "arxiv": 3},
        "candidates": [
            _row("What is the Evidence for Dark Matter?", "arxiv", 0, url="https://arxiv.org/abs/x"),
            _row("What is the evidence for dark matter", "openalex", 1, doi="10.1/a"),
            _row("Another paper", "openalex", 0, label="partial", doi="10.1/b"),
        ],
    }


class FirstSeenUniqueTests(unittest.TestCase):
    def test_keeps_the_copies_rrf_merge_keeps(self) -> None:
        rows = [
            {"title": "Paper A", "doi": "10.1/x"},
            {"title": "Paper B"},
            {"title": "paper a!"},  # same title as the first row
            {"title": "Other title", "doi": "10.1/X"},  # same DOI as the first row
            {"title": "Other title"},  # matches a key the dropped row brought
        ]
        kept = first_seen_unique(rows)
        self.assertEqual([r["title"] for r in kept], ["Paper A", "Paper B"])
        self.assertTrue(all(any(k is r for r in rows) for k in kept))
        self.assertEqual(sorted(r["title"] for r in rrf_merge([rows])), ["Paper A", "Paper B"])

    def test_rows_without_keys_are_kept(self) -> None:
        rows = [{"title": ""}, {"title": ""}]
        self.assertEqual(len(first_seen_unique(rows)), 2)
        self.assertEqual(merge_keys(rows[0]), [])


class ReplayPoolTests(unittest.TestCase):
    def test_scientific_pool_drops_the_fused_copy(self) -> None:
        fx = _fixture()
        # Adapter order is production_depth's: openalex first, so its copy is the one kept.
        self.assertEqual(
            [r["title"] for r in qe.fixture_pool(fx)],
            ["Another paper", "What is the evidence for dark matter"],
        )
        self.assertEqual(len(qe.fixture_pool(fx, merge=False)), 3)

    def test_other_paths_do_not_merge(self) -> None:
        self.assertEqual(len(qe.fixture_pool(_fixture("general_web"))), 3)

    def test_ranker_sees_the_merged_pool(self) -> None:
        seen: list[list[str]] = []

        def rank(query, rows, budget, path=""):
            seen.append([r["title"] for r in rows])
            return qe.RankerOutput(kept=rows[:budget], rejected=[])

        report = qe.evaluate_fixture(_fixture(), ranker=rank)
        self.assertEqual(len(seen[0]), 2)
        self.assertEqual(report["pool_size"], 2)
        self.assertTrue(report["replay_merge"])
        # One relevant paper, not two: the ideal ranking no longer counts the copy.
        self.assertEqual(report["relevant_in_pool"], 1)

    def test_shared_pool_merges_copies_two_arms_saw(self) -> None:
        fx = {
            "id": "arms", "path": "scientific_evidence", "query_clean": "q",
            "production_depth": {"a": 2, "b": 2, "c": 2},
            "routes": {"production": {ROUTE_PRODUCTION: ["b", "a"], ROUTE_NO_DISCIPLINE: ["c", "a"]}},
            "candidates": [
                _row("Shared Paper", "b", 0, doi="10.1/b"),
                _row("Shared paper.", "c", 0, doi="10.1/c", added=True),
                _row("a0", "a", 0, label="partial"),
            ],
        }
        arms = [{"route_arm": ROUTE_PRODUCTION}, {"route_arm": ROUTE_NO_DISCIPLINE}]
        self.assertEqual(len(qe.fixture_pool(fx, route_arm=ROUTE_PRODUCTION)), 2)
        self.assertEqual(len(qe.fixture_pool(fx, route_arm=ROUTE_NO_DISCIPLINE)), 2)
        self.assertEqual(len(qe.shared_pool(fx, arms)), 2)
        self.assertEqual(len(qe.shared_pool(fx, [dict(a, merge=False) for a in arms])), 3)

    def test_runs_stored_before_s21_rebuild_unmerged_pools(self) -> None:
        self.assertFalse(qe.run_pool_spec({}, {"query_mode": "production"})["merge"])
        self.assertTrue(qe.run_pool_spec({}, {"replay_merge": True})["merge"])
        fx = qe.load_json("eval/fixtures/knowledge_quality/sci_abstain_fictional_drug.json")
        s13 = json.loads((BASELINES / "kq_s13_token_production.json").read_text(encoding="utf-8"))
        stored = next(r for r in s13["results"] if r["id"] == "sci_abstain_fictional_drug")
        spec = qe.run_pool_spec(s13, stored)
        self.assertEqual(len(qe.fixture_pool(fx, **qe._pool_args(spec))), stored["pool_size"])
        self.assertLess(len(qe.fixture_pool(fx)), stored["pool_size"])


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = qe.load_json(CORPUS)
        cls.entries = [e for e in corpus["queries"] if e.get("fixture")]
        cls.fixtures = {e["id"]: qe.load_json(e["fixture"]) for e in cls.entries}

    def test_no_scientific_replay_keeps_two_copies_of_one_paper(self) -> None:
        for fid, fx in self.fixtures.items():
            if qe._normalize_path(fx.get("path")) != qe.PATH_SCIENTIFIC:
                continue
            for mode in ("production", "standalone"):
                arms = list(ROUTE_ARMS) if qe.fixture_has_routes(fx, mode) else [None]
                for arm in arms:
                    with self.subTest(fixture=fid, mode=mode, arm=arm):
                        pool = qe.fixture_pool(fx, query_mode=mode, route_arm=arm)
                        keys = [k for row in pool for k in merge_keys(row)]
                        self.assertEqual(len(keys), len(set(keys)))

    def test_s21_token_baselines_are_todays_replay(self) -> None:
        for mode in ("production", "standalone"):
            with self.subTest(mode=mode):
                base = json.loads((BASELINES / f"kq_s21_token_{mode}.json").read_text(encoding="utf-8"))
                stored = {r["id"]: r for r in base["results"]}
                for entry in self.entries:
                    rep = qe.evaluate_fixture(self.fixtures[entry["id"]], ranker="token", query_mode=mode)
                    want = stored[entry["id"]]
                    self.assertEqual(rep["kept_titles"], want["kept_titles"], entry["id"])
                    self.assertEqual(rep["pool_size"], want["pool_size"], entry["id"])
                    self.assertTrue(want["replay_merge"])


if __name__ == "__main__":
    unittest.main()
