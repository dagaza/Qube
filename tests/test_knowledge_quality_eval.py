"""Stage S1 — generic knowledge-quality eval harness and metric definitions.

Pins the metric conventions documented in ``core/knowledge/quality_eval.py`` so
later stages (reranker, sufficiency gate, query resolution) are measured on a
stable scale.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.quality_eval import (  # noqa: E402
    RankerOutput,
    RankerUnavailable,
    abstain_correct,
    abstained,
    aggregate_reports,
    candidate_gain,
    compare_to_baseline,
    evaluate_fixture,
    fixture_pool,
    fixture_ranking_query,
    get_fixture_ranker,
    irrelevant_kept_rate,
    load_json,
    ndcg_at_k,
    precision_at_k,
    relevant_recall,
    validate_fixture,
)

FIXTURE_DIR = ROOT / "eval" / "fixtures" / "knowledge_quality"


def _row(title: str, label: str | None, **extra) -> dict:
    row = {"title": title, "snippet": title, **extra}
    if label is not None:
        row["label"] = label
    return row


R1 = _row("rel one", "relevant")
R2 = _row("rel two", "relevant")
P1 = _row("partial one", "partial")
I1 = _row("irr one", "irrelevant")
I2 = _row("irr two", "irrelevant")
U1 = _row("unlabelled", None)


class CandidateGainTests(unittest.TestCase):
    def test_graded_gains_and_unknown(self) -> None:
        self.assertEqual(candidate_gain(R1), 2)
        self.assertEqual(candidate_gain(P1), 1)
        self.assertEqual(candidate_gain(I1), 0)
        self.assertIsNone(candidate_gain(U1))
        self.assertIsNone(candidate_gain({"label": "maybe"}))
        self.assertEqual(candidate_gain({"label": "Relevant"}), 2)


class PrecisionTests(unittest.TestCase):
    def test_counts_relevant_and_partial_as_useful(self) -> None:
        self.assertAlmostEqual(precision_at_k([R1, P1, I1], 3), 2 / 3)

    def test_divides_by_returned_rows_not_k(self) -> None:
        self.assertEqual(precision_at_k([R1], 3), 1.0)

    def test_only_top_k_considered(self) -> None:
        self.assertEqual(precision_at_k([I1, R1, R2], 1), 0.0)

    def test_unlabelled_rows_ignored(self) -> None:
        self.assertEqual(precision_at_k([U1, R1], 2), 1.0)

    def test_undefined_cases(self) -> None:
        self.assertIsNone(precision_at_k([], 3))
        self.assertIsNone(precision_at_k([U1], 3))
        self.assertIsNone(precision_at_k([R1], 0))


class NdcgTests(unittest.TestCase):
    def test_ideal_order_scores_one(self) -> None:
        pool = [I1, P1, R1]
        self.assertAlmostEqual(ndcg_at_k([R1, P1], pool, 5), 1.0)

    def test_worse_order_scores_lower(self) -> None:
        pool = [R1, P1, I1]
        ideal = ndcg_at_k([R1, P1, I1], pool, 5)
        worse = ndcg_at_k([I1, P1, R1], pool, 5)
        self.assertAlmostEqual(ideal, 1.0)
        self.assertLess(worse, ideal)
        expected = ((2**1 - 1) / math.log2(3) + (2**2 - 1) / math.log2(4)) / (
            (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
        )
        self.assertAlmostEqual(worse, expected)

    def test_empty_ranked_with_relevant_pool_is_zero(self) -> None:
        self.assertEqual(ndcg_at_k([], [R1, I1], 5), 0.0)

    def test_undefined_when_pool_has_nothing_useful(self) -> None:
        self.assertIsNone(ndcg_at_k([I1], [I1, I2], 5))


class RateAndRecallTests(unittest.TestCase):
    def test_irrelevant_kept_rate(self) -> None:
        self.assertAlmostEqual(irrelevant_kept_rate([R1, I1, I2]), 2 / 3)
        self.assertEqual(irrelevant_kept_rate([R1, P1]), 0.0)
        self.assertIsNone(irrelevant_kept_rate([]))
        self.assertIsNone(irrelevant_kept_rate([U1]))

    def test_relevant_recall_capped_by_budget(self) -> None:
        pool = [R1, R2, _row("rel three", "relevant"), I1]
        # Budget 2 can hold at most 2 relevant rows → 2 kept = perfect recall.
        self.assertEqual(relevant_recall([R1, R2], pool, budget=2), 1.0)
        self.assertAlmostEqual(relevant_recall([R1, I1], pool, budget=2), 0.5)

    def test_relevant_recall_ignores_partial(self) -> None:
        self.assertEqual(relevant_recall([P1], [R1, P1], budget=3), 0.0)

    def test_relevant_recall_undefined_without_relevant(self) -> None:
        self.assertIsNone(relevant_recall([P1], [P1, I1], budget=3))


class AbstentionTests(unittest.TestCase):
    def test_abstained(self) -> None:
        self.assertTrue(abstained([]))
        self.assertTrue(abstained([R1], "insufficient"))
        self.assertTrue(abstained([R1], "none"))
        self.assertFalse(abstained([R1], "unscored"))
        self.assertFalse(abstained([R1], "partial"))

    def test_abstain_correct(self) -> None:
        self.assertTrue(abstain_correct("insufficient", True))
        self.assertFalse(abstain_correct("insufficient", False))
        self.assertTrue(abstain_correct("sufficient", False))
        self.assertFalse(abstain_correct("partial", True))
        self.assertIsNone(abstain_correct(None, True))


class ValidationTests(unittest.TestCase):
    def _valid(self) -> dict:
        return {
            "schema_version": 2,
            "id": "x",
            "query_clean": "q",
            "expect": "sufficient",
            "budget": 3,
            "candidates": [
                {"title": "a", "label": "relevant", "label_source": "model"},
                {"title": "b", "label": "irrelevant"},
            ],
        }

    def test_valid_fixture(self) -> None:
        self.assertEqual(validate_fixture(self._valid()), [])

    def test_v1_fixture_without_new_fields_is_valid(self) -> None:
        fx = {"id": "old", "query_raw": "q", "candidates": [{"title": "a", "label": "relevant"}]}
        self.assertEqual(validate_fixture(fx), [])

    def test_reports_problems(self) -> None:
        fx = self._valid()
        fx["expect"] = "maybe"
        fx["budget"] = 0
        fx["candidates"].append({"title": "a", "label": "relevant"})  # duplicate
        fx["candidates"].append({"title": "c", "label": "great"})
        fx["candidates"].append({"title": "d", "label_source": "llm"})
        errors = validate_fixture(fx)
        joined = " | ".join(errors)
        self.assertIn("invalid expect", joined)
        self.assertIn("invalid budget", joined)
        self.assertIn("duplicate candidate", joined)
        self.assertIn("invalid label 'great'", joined)
        self.assertIn("invalid label_source", joined)

    def test_all_repo_fixtures_validate(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("*.json"))
        self.assertTrue(paths, "no knowledge-quality fixtures found")
        for path in paths:
            with self.subTest(fixture=path.name):
                fixture = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(validate_fixture(fixture), [])

    def test_corpus_fixture_paths_exist(self) -> None:
        corpus = load_json("eval/retrieval_corpus/v1_knowledge_quality.json")
        for entry in corpus.get("queries") or []:
            if entry.get("fixture"):
                with self.subTest(case=entry.get("id")):
                    self.assertTrue((ROOT / entry["fixture"]).is_file())


class PoolAndQueryTests(unittest.TestCase):
    def test_synthetic_rows_excluded_by_default(self) -> None:
        fx = {"candidates": [R1, dict(R2, synthetic=True)]}
        self.assertEqual([r["title"] for r in fixture_pool(fx)], ["rel one"])
        self.assertEqual(len(fixture_pool(fx, include_synthetic=True)), 2)

    def test_pool_returns_copies(self) -> None:
        fx = {"candidates": [dict(R1)]}
        fixture_pool(fx)[0]["title"] = "mutated"
        self.assertEqual(fx["candidates"][0]["title"], "rel one")

    def test_query_modes(self) -> None:
        fx = {"query_raw": "@[tool:x] raw", "query_clean": "clean", "standalone_question": "gold"}
        self.assertEqual(fixture_ranking_query(fx), "clean")
        self.assertEqual(fixture_ranking_query(fx, query_mode="standalone"), "gold")
        self.assertEqual(
            fixture_ranking_query({"query_clean": "clean"}, query_mode="standalone"), "clean"
        )


def _stub_ranker(keep_titles: list[str], state: str = "unscored"):
    def rank(query: str, rows: list[dict], budget: int) -> RankerOutput:
        kept = [r for r in rows if r["title"] in keep_titles][:budget]
        rejected = [r for r in rows if r not in kept]
        return RankerOutput(kept=kept, rejected=rejected, state=state)

    return rank


class EvaluateFixtureTests(unittest.TestCase):
    def _fixture(self) -> dict:
        return {
            "id": "fx",
            "path": "general_web",
            "category": "how_to",
            "expect": "sufficient",
            "query_clean": "q",
            "budget": 2,
            "candidates": [R1, P1, I1, dict(R2, synthetic=True)],
        }

    def test_metrics_with_stub_ranker(self) -> None:
        rep = evaluate_fixture(self._fixture(), ranker=_stub_ranker(["rel one", "irr one"]))
        self.assertEqual(rep["pool_size"], 3)
        self.assertEqual(rep["synthetic_excluded"], 1)
        self.assertEqual(rep["kept_count"], 2)
        self.assertEqual(rep["precision_at_k"], 0.5)
        self.assertEqual(rep["irrelevant_kept_rate"], 0.5)
        self.assertEqual(rep["relevant_recall"], 1.0)
        self.assertFalse(rep["abstained"])
        self.assertTrue(rep["abstain_correct"])
        self.assertEqual(rep["path"], "general_web")
        self.assertEqual(rep["category"], "how_to")

    def test_abstention_from_ranker_state(self) -> None:
        fx = self._fixture()
        fx["expect"] = "insufficient"
        rep = evaluate_fixture(fx, ranker=_stub_ranker(["irr one"], state="insufficient"))
        self.assertTrue(rep["abstained"])
        self.assertTrue(rep["abstain_correct"])

    def test_include_synthetic(self) -> None:
        rep = evaluate_fixture(
            self._fixture(), ranker=_stub_ranker(["rel two"]), include_synthetic=True
        )
        self.assertEqual(rep["pool_size"], 4)
        self.assertEqual(rep["synthetic_excluded"], 0)
        self.assertEqual(rep["kept_count"], 1)

    def test_defaults_for_v1_fixture(self) -> None:
        rep = evaluate_fixture(
            {"id": "v1", "query_raw": "q", "candidates": [R1]}, ranker=_stub_ranker(["rel one"])
        )
        self.assertEqual(rep["path"], "unspecified")
        self.assertEqual(rep["category"], "uncategorized")
        self.assertEqual(rep["budget"], 3)
        self.assertIsNone(rep["abstain_correct"])

    def test_rejects_unknown_query_mode(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_fixture(self._fixture(), ranker=_stub_ranker([]), query_mode="other")

    def test_token_ranker_uses_production_pipeline(self) -> None:
        fx = {
            "id": "tok",
            "query_clean": "actor network theory artificial intelligence",
            "budget": 3,
            "candidates": [
                _row("Actor network theory and artificial intelligence", "relevant"),
                _row("Ant colony optimisation for routing", "irrelevant"),
            ],
        }
        rep = evaluate_fixture(fx, ranker="token")
        self.assertEqual(rep["ranker"], "token")
        self.assertEqual(rep["kept_titles"][0], "Actor network theory and artificial intelligence")

    def test_unknown_ranker(self) -> None:
        with self.assertRaises(RankerUnavailable):
            get_fixture_ranker("nope")


class AntFixtureBaselineTests(unittest.TestCase):
    """The real ANT pool has no relevant row; today's ranker does not abstain."""

    def test_real_pool_expects_abstention_and_token_ranker_misses_it(self) -> None:
        fixture = load_json("eval/fixtures/knowledge_quality/ant_ai_candidates.json")
        self.assertEqual(fixture.get("expect"), "insufficient")
        self.assertTrue(any(c.get("synthetic") for c in fixture["candidates"]))
        rep = evaluate_fixture(fixture, ranker="token")
        self.assertEqual(rep["relevant_in_pool"], 0)
        self.assertEqual(rep["synthetic_excluded"], 1)
        # Known gap until S3/S4: irrelevant rows kept, no abstention.
        self.assertGreater(rep["kept_count"], 0)
        self.assertEqual(rep["irrelevant_kept_rate"], 1.0)
        self.assertFalse(rep["abstain_correct"])


class AggregationTests(unittest.TestCase):
    def _reports(self) -> list[dict]:
        return [
            {"category": "a", "precision_at_k": 1.0, "ndcg_at_5": None, "irrelevant_kept_rate": 0.0,
             "relevant_recall": 1.0, "abstain_correct": True, "expect": "sufficient", "abstained": False},
            {"category": "a", "precision_at_k": 0.5, "ndcg_at_5": 0.5, "irrelevant_kept_rate": 0.5,
             "relevant_recall": None, "abstain_correct": False, "expect": "insufficient", "abstained": False},
            {"category": "b", "precision_at_k": None, "ndcg_at_5": None, "irrelevant_kept_rate": None,
             "relevant_recall": None, "abstain_correct": True, "expect": "insufficient", "abstained": True},
        ]

    def test_overall_means_skip_none(self) -> None:
        agg = aggregate_reports(self._reports())["all"]
        self.assertEqual(agg["fixtures"], 3)
        self.assertAlmostEqual(agg["precision_at_k"], 0.75)
        self.assertAlmostEqual(agg["ndcg_at_5"], 0.5)
        self.assertAlmostEqual(agg["abstain_correct"], 2 / 3)
        self.assertEqual(agg["abstain_precision"], 1.0)
        self.assertAlmostEqual(agg["abstain_recall"], 0.5)

    def test_grouped(self) -> None:
        agg = aggregate_reports(self._reports(), group_by="category")
        self.assertEqual(set(agg), {"a", "b"})
        self.assertEqual(agg["a"]["fixtures"], 2)
        self.assertIsNone(agg["b"]["precision_at_k"])
        self.assertIsNone(agg["a"]["abstain_precision"])  # no abstentions in "a"


class BaselineCompareTests(unittest.TestCase):
    def test_flags_drop_beyond_tolerance(self) -> None:
        base = {"cat": {"fixtures": 40, "precision_at_k": 0.80, "irrelevant_kept_rate": 0.10}}
        cur = {"cat": {"fixtures": 40, "precision_at_k": 0.70, "irrelevant_kept_rate": 0.20}}
        regs = compare_to_baseline(cur, base)
        self.assertEqual({r["metric"] for r in regs}, {"precision_at_k", "irrelevant_kept_rate"})

    def test_within_tolerance_and_improvements_pass(self) -> None:
        base = {"cat": {"fixtures": 40, "precision_at_k": 0.80, "irrelevant_kept_rate": 0.20}}
        cur = {"cat": {"fixtures": 40, "precision_at_k": 0.77, "irrelevant_kept_rate": 0.05}}
        self.assertEqual(compare_to_baseline(cur, base), [])

    def test_one_fixture_tolerance_for_small_groups(self) -> None:
        # 4 fixtures → tolerance 0.25 (one fixture), so a 0.2 drop passes.
        base = {"cat": {"fixtures": 4, "abstain_correct": 1.0}}
        cur = {"cat": {"fixtures": 4, "abstain_correct": 0.8}}
        self.assertEqual(compare_to_baseline(cur, base), [])
        cur = {"cat": {"fixtures": 4, "abstain_correct": 0.5}}
        self.assertEqual(len(compare_to_baseline(cur, base)), 1)

    def test_missing_groups_and_metrics_skipped(self) -> None:
        base = {"old": {"fixtures": 3, "precision_at_k": 1.0}, "cat": {"fixtures": 3}}
        cur = {"cat": {"fixtures": 3, "precision_at_k": 0.0}, "new": {"fixtures": 1}}
        self.assertEqual(compare_to_baseline(cur, base), [])


class CliSmokeTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "evaluate_knowledge_quality.py"), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_default_run_passes_contract_gate(self) -> None:
        proc = self._run("--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertGreaterEqual(summary["evaluated"], 1)
        self.assertIn("category", summary["aggregates"])
        self.assertEqual(summary["failures"], [])

    def test_metric_gate_fails_when_unmet(self) -> None:
        proc = self._run("--max-irrelevant-kept-rate", "0.05")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("gate max_irrelevant_kept_rate", proc.stdout)

    def test_compare_against_own_output_has_no_regressions(self) -> None:
        baseline = self._run("--json")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        tmp = ROOT / "tests" / "_kq_baseline_tmp.json"
        try:
            tmp.write_text(baseline.stdout, encoding="utf-8")
            proc = self._run("--compare", str(tmp))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
