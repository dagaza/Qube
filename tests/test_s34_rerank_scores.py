"""Plan S34: recorded cross-encoder scores on the S33 corpus, and what they show.

``eval/fixtures/library_memory/rerank_scores.json`` holds the S3
cross-encoder's score for every candidate of every run of the three captures
(the pool the S34 shadow pass sees). ``lm_s34_rerank_*`` are the distribution
reports. These tests check the recording covers the captures, replays through
the production shadow path, and the S34 findings (V71, V73) as recorded.
Today's captures report to ``CURRENT_RERANK`` (re-recorded after S36 changed
the follow-up search texts); ``lm_s34_rerank_*`` stay S34's record.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import library_memory_eval as lme  # noqa: E402
from core.knowledge.ranking import library_memory_shadow as shadow  # noqa: E402
from core.knowledge.ranking import rerank  # noqa: E402

MODES = ("balanced", "fast", "power")
FIXTURE_DIR = ROOT / "eval" / "fixtures" / "library_memory"
BASELINE_DIR = ROOT / "eval" / "library_memory_corpus" / "baselines"
CURRENT_RERANK = "lm_s36_rerank"  # what today's captures and recording report (S36)


def _report(mode: str, stage: str = "lm_s34_rerank") -> dict:
    return lme.load_json(BASELINE_DIR / f"{stage}_{mode}.json")


class TestRecordedScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = lme.load_corpus()
        cls.captures = {m: lme.load_json(FIXTURE_DIR / f"captured_{m}.json") for m in MODES}
        cls.recorded = lme.load_json(lme.DEFAULT_RERANK_SCORES)

    def test_recording_matches_the_model_and_corpus(self):
        self.assertEqual(self.recorded["schema"], lme.RERANK_SCHEMA)
        self.assertEqual(self.recorded["model"], rerank.RERANKER_MODEL_ID)
        self.assertEqual(self.recorded["text_chars"], rerank.DEFAULT_TEXT_CHARS)
        self.assertEqual(self.recorded["corpus_fingerprint"], lme.corpus_fingerprint())

    def test_every_pool_pair_is_recorded_with_todays_passage(self):
        pairs = 0
        for mode, cap in self.captures.items():
            for case in self.corpus["cases"]:
                for run in lme.case_runs(case):
                    question = cap["cases"][case["id"]]["runs"][run]["search_text"]
                    for ref in lme.rerank_pool(cap, case, run):
                        with self.subTest(mode=mode, case=case["id"], run=run, ref=ref):
                            self.assertIn(ref, self.recorded["scores"][question])
                            self.assertEqual(
                                self.recorded["passages"][ref], lme._passage_hash(lme.rerank_passage(cap, case, ref))
                            )
                        pairs += 1
        self.assertGreater(pairs, 6000)

    def test_reports_are_todays_distributions(self):
        for mode, cap in self.captures.items():
            with self.subTest(mode=mode):
                report = json.loads(json.dumps(lme.rerank_report(cap, self.corpus, self.recorded)))
                self.assertEqual(report, _report(mode, CURRENT_RERANK))

    def test_replay_scorer_refuses_an_unrecorded_passage(self):
        scorer = lme.recorded_rerank_scorer(list(self.captures.values()), self.corpus, self.recorded)
        case = next(c for c in self.corpus["cases"] if c["id"] == "lib_omega_only_sky_blue")
        cap = self.captures["balanced"]
        question = cap["cases"][case["id"]]["runs"][lme.RUN_LIBRARY_CHAT]["search_text"]
        ref = lme.rerank_pool(cap, case, lme.RUN_LIBRARY_CHAT)[0]
        self.assertEqual(scorer(question, [lme.rerank_passage(cap, case, ref)]), [self.recorded["scores"][question][ref]])
        with self.assertRaises(lme.ReplayMismatch):
            scorer(question, ["a passage nobody recorded"])

    def test_the_shadow_pass_replays_from_the_recording(self):
        # The production shadow path (rag_search -> offer -> score_job) scores exactly the recorded pairs.
        import mcp.rag_tool as rt

        cap = self.captures["balanced"]
        rerank.set_reranker_scorer_for_tests(lme.recorded_rerank_scorer([cap], self.corpus, self.recorded))
        orig = rt.rag_search
        rt.rag_search = lambda *a, **k: orig(*a, shadow_path=shadow.PATH_LIBRARY_CHAT, **k)
        try:
            for case in [c for c in self.corpus["cases"] if c["path"] == "library"][:20]:
                with self.subTest(case=case["id"]):
                    shadow.begin_turn()
                    kept = lme.replay_run(cap, case, lme.RUN_LIBRARY_CHAT)
                    (job,) = shadow.end_turn()
                    rec = shadow.score_job(job)
                    self.assertIsNone(rec["skipped_reason"])
                    self.assertEqual(rec["kept"], len(kept))
                    question = cap["cases"][case["id"]]["runs"][lme.RUN_LIBRARY_CHAT]["search_text"]
                    recorded = self.recorded["scores"][question]
                    expected = [round(recorded[k], 2) for k in kept]
                    self.assertEqual(rec["kept_scores"], expected)
        finally:
            rt.rag_search = orig
            shadow.end_turn()
            rerank.set_reranker_scorer_for_tests(None)


class TestS34Findings(unittest.TestCase):
    """V71 and V73, read from the committed reports."""

    def test_v71_cross_encoder_separates_chunks_better_than_the_embedder(self):
        for mode in MODES:
            for run in lme.LIBRARY_RUNS:
                with self.subTest(mode=mode, run=run):
                    s = _report(mode)["runs"][run]
                    self.assertGreaterEqual(s["auc_ce_relevant_vs_irrelevant"], 0.95)
                    self.assertGreater(s["auc_ce_relevant_vs_irrelevant"], s["auc_semantic_relevant_vs_irrelevant"] + 0.05)

    def test_v71_one_threshold_reads_the_same_on_every_embedder(self):
        # The cross-encoder reads text, so a relevant chunk scores the same whichever embedder retrieved it.
        p50 = {mode: _report(mode)["runs"][lme.RUN_LIBRARY_CHAT]["ce"]["relevant"]["p50"] for mode in MODES}
        self.assertLess(max(p50.values()) - min(p50.values()), 0.5)

    def test_v73_core_and_narrative_memories_score_like_unrelated_ones(self):
        for mode in MODES:
            for run in ("memory_core", "memory_narrative"):
                with self.subTest(mode=mode, run=run):
                    s = _report(mode)["runs"][run]
                    self.assertLess(s["ce"]["relevant"]["p50"], -10.0)
                    self.assertEqual(s["at_or_above"]["-4"]["relevant"], 1)
            recall = _report(mode)["runs"]["memory_recall"]
            self.assertGreater(recall["ce"]["relevant"]["p50"], -1.0)
            self.assertEqual(recall["at_or_above"]["-4"]["irrelevant"], 0)


if __name__ == "__main__":
    unittest.main()
