"""Plan S23 (D41, V31): a trial is named only by its whole name, as words.

Before S23, the ranking side counted a trial as named when each of its 3+-letter
name parts appeared anywhere in the text. So "dapagliflozin" was a DAPA-HF hit,
and "Collaborative Development Paradigm Divergence" a PARADIGM-HF hit. The query
side read "the scientific consensus on …" as the 1987 CONSENSUS trial, and it
also cut that question's PubMed keyword query down to "CONSENSUS". One pattern
per trial now serves queries and rows alike. CONSENSUS, the one name that is
also an ordinary word, matches in capitals, or as "consensus trial" in any case.
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
from core.knowledge.entities.trials import trial_name_pattern  # noqa: E402
from core.knowledge.ranking.trial_grounding import (  # noqa: E402
    _trial_in_text,
    extract_trial_signal,
    trial_grounding_boost,
)
from core.knowledge.scientific_query_planner import plan_scientific_query  # noqa: E402

CORPUS = "eval/retrieval_corpus/v1_knowledge_quality.json"
BASELINES = ROOT / "eval" / "retrieval_corpus" / "baselines"
_GLYPHOSATE = "What is the scientific consensus on the safety of glyphosate?"


class QuerySignalTests(unittest.TestCase):
    def test_scientific_consensus_is_not_the_trial(self) -> None:
        self.assertEqual(extract_trial_signal(_GLYPHOSATE), frozenset())
        self.assertEqual(extract_trial_signal("Is there a Delphi consensus study on long COVID care?"), frozenset())

    def test_the_consensus_trial_by_name(self) -> None:
        for q in ("What did CONSENSUS show for enalapril?", "consensus trial enalapril mortality"):
            with self.subTest(q=q):
                self.assertEqual(extract_trial_signal(q), frozenset({"consensus"}))

    def test_named_trials_with_any_separator(self) -> None:
        for q in ("DAPA-HF results", "DAPA HF results", "dapa-hf results", "DAPA‐HF results", "DAPA–HF results", "DAPAHF results"):
            with self.subTest(q=q):
                self.assertEqual(extract_trial_signal(q), frozenset({"dapa-hf"}))
        self.assertEqual(extract_trial_signal("dapagliflozin in heart failure"), frozenset())


class RowMatchTests(unittest.TestCase):
    def test_a_name_part_is_not_a_hit(self) -> None:
        for slug, text in (
            ("dapa-hf", "Efficacy and safety of dapagliflozin in heart failure with reduced ejection fraction"),
            ("paradigm-hf", "From OSS to Open Source AI: an Exploratory Study of Collaborative Development Paradigm Divergence"),
            ("emperor-reduced", "Emperor penguin colonies at reduced sea-ice extent"),
            ("consensus", "Glyphosate and cancer risk: a consensus statement"),
        ):
            with self.subTest(slug=slug):
                self.assertFalse(_trial_in_text(slug, text))

    def test_the_whole_name_is_a_hit(self) -> None:
        for slug, text in (
            ("dapa-hf", "Efficacy and safety of dapagliflozin according to aetiology: insights from the DAPA‐HF trial"),
            ("dapa-hf", "Results of DAPA HF"),
            ("paradigm-hf", "Angiotensin-neprilysin inhibition in PARADIGM-HF"),
            ("emperor-reduced", "Post hoc analysis from EMPEROR-Reduced."),
            ("consensus", "Effects of enalapril on mortality in severe heart failure: the CONSENSUS trial"),
            ("consensus", "Twenty years after the consensus trial"),
        ):
            with self.subTest(slug=slug, text=text):
                self.assertTrue(_trial_in_text(slug, text))

    def test_a_slug_outside_the_table_needs_every_part_as_a_word(self) -> None:
        pattern = trial_name_pattern("dapa-ckd")
        self.assertIsNotNone(pattern)
        self.assertTrue(pattern.search("Dapagliflozin in CKD: the DAPA-CKD trial"))
        self.assertTrue(pattern.search("DAPA CKD"))
        self.assertFalse(pattern.search("dapagliflozin in CKD"))
        self.assertIsNone(trial_name_pattern(" - "))

    def test_a_name_part_earns_the_no_hit_penalty(self) -> None:
        row = {"title": "Efficacy of dapagliflozin in heart failure", "snippet": "Dapagliflozin outcomes.", "_adapter": "pubmed"}
        self.assertAlmostEqual(trial_grounding_boost(row, frozenset({"dapa-hf"})), -0.15)


class PlannerTests(unittest.TestCase):
    def test_the_glyphosate_keyword_query_is_no_longer_the_trial_name(self) -> None:
        plan = plan_scientific_query(_GLYPHOSATE, semantic_query=_GLYPHOSATE)
        self.assertEqual(plan.keyword_query, "What is the scientific consensus on the safety of glyphosate")


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = qe.load_json(CORPUS)
        cls.fixtures = {e["id"]: qe.load_json(e["fixture"]) for e in corpus["queries"] if e.get("fixture")}

    def test_only_the_three_named_trials_carry_a_signal(self) -> None:
        fired = {
            fid: sorted(extract_trial_signal(fx.get("retrieval_query") or ""))
            for fid, fx in self.fixtures.items()
            if extract_trial_signal(fx.get("retrieval_query") or "")
        }
        self.assertEqual(
            fired,
            {"s13_trial_emperor_reduced": ["emperor-reduced"], "s13_trial_dapa_hf": ["dapa-hf"],
             "s13_trial_paradigm_hf": ["paradigm-hf"]},
        )

    def test_token_ranking_of_the_two_reranked_fixtures_improved(self) -> None:
        # Their pools are the same as in S22, so the stored S22 scores compare
        # like for like.
        for mode in ("production", "standalone"):
            stored = {
                r["id"]: r
                for r in json.loads((BASELINES / f"kq_s22_token_{mode}.json").read_text(encoding="utf-8"))["results"]
            }
            for fid in ("s13_trial_emperor_reduced", "s13_trial_false_consensus"):
                with self.subTest(mode=mode, fixture=fid):
                    rep = qe.evaluate_fixture(self.fixtures[fid], ranker="token", query_mode=mode)
                    self.assertEqual(rep["pool_size"], stored[fid]["pool_size"])
                    self.assertGreater(rep["ndcg_at_5"], stored[fid]["ndcg_at_5"] + 0.15)


if __name__ == "__main__":
    unittest.main()
