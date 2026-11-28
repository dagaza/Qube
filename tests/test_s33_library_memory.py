"""Plan S33: library and memory fixtures, and a baseline of today's floors.

The corpus (``eval/library_memory_corpus/v1_library_memory.json``) is synthetic:
15 invented documents and 33 invented memories, with labels written from their
text before any scoring. Each shipped embedder (Balanced, Fast, Power) has a
capture of what the production calls read (``eval/fixtures/library_memory``)
and a baseline report (``eval/library_memory_corpus/baselines/lm_s33_*``).

These tests check the fixtures, that the offline replay reproduces what the
production calls kept at capture time, and the S33 findings as recorded in
the S33 baselines (V63–V65, V68 in the plan). They do not pin the memory
shadowing bug (V66): S41 fixes it.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import library_memory_eval as lme  # noqa: E402

MODES = ("balanced", "fast", "power")
FIXTURE_DIR = ROOT / "eval" / "fixtures" / "library_memory"
BASELINE_DIR = ROOT / "eval" / "library_memory_corpus" / "baselines"
CURRENT_BASELINE = "lm_s36"  # the baseline today's captures replay to (S36)


_V63_PROBE = """
import json, math, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from rag.store import DocumentStore
dim = 16
q = np.zeros(dim, dtype=np.float32)
q[0] = 1.0
rows = []
for i, cos in enumerate((0.95, 0.65, 0.5)):
    v = np.zeros(dim, dtype=np.float32)
    v[0], v[1] = cos, math.sqrt(1 - cos * cos)
    rows.append({"vector": v.tolist(), "text": str(cos), "source": f"s{i}.md", "chunk_id": 0})
with tempfile.TemporaryDirectory() as tmp:
    store = DocumentStore(Path(tmp), expected_vector_dim=dim, quiet=True)
    store.add_chunks(rows, rebuild_fts=False)
    hits = store.table.search(q).limit(3).to_list()
print(json.dumps([[float(h["text"]), float(h["_distance"])] for h in hits]))
"""


def _capture(mode: str) -> dict:
    return lme.load_json(FIXTURE_DIR / f"captured_{mode}.json")


def _s33(mode: str) -> dict:
    return lme.load_json(BASELINE_DIR / f"lm_s33_{mode}.json")


def _case_report(report: dict, case_id: str, run: str) -> dict:
    return next(r for r in report["cases"] if r["id"] == case_id and r["run"] == run)


class TestCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = lme.load_corpus()

    def test_corpus_validates(self):
        self.assertEqual(lme.validate_corpus(self.corpus), [])

    def test_every_label_phrase_matches_one_captured_chunk(self):
        for mode in MODES:
            chunks = _capture(mode)["stores"]["full"]["chunks"]
            with self.subTest(mode=mode):
                self.assertEqual(lme.validate_corpus(self.corpus, chunks), [])

    def test_stage_case_kinds_are_present(self):
        cats = {(c["path"], c["category"]) for c in self.corpus["cases"]}
        for needed in (
            ("library", "offtopic_abstain"),
            ("library", "near_miss_abstain"),
            ("library", "multi_turn"),
            ("memory", "recall_abstain"),
            ("memory", "core_abstain"),
            ("memory", "multi_turn"),
        ):
            self.assertIn(needed, cats)
        modes = {c.get("mode") for c in self.corpus["cases"] if c["path"] == "memory"}
        self.assertEqual(modes, set(lme.MEMORY_MODES))

    def test_t41_scenario_is_a_case(self):
        # T4.1: "Why is the sky blue?" against a library holding only the blue jay study.
        case = next(c for c in self.corpus["cases"] if c["id"] == "lib_omega_only_sky_blue")
        self.assertEqual(lme.case_store(case), "omega_only")
        self.assertFalse(case.get("labels"))
        self.assertEqual(self.corpus["stores"]["omega_only"]["docs"], ["project_omega_bluejay.md"])


class TestPrivacy(unittest.TestCase):
    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    HOME_PATH = re.compile(r"(/home/|/Users/|[A-Za-z]:\\\\?Users\\\\?)[^\s\"'/\\]+", re.IGNORECASE)

    def _files(self):
        for base in (FIXTURE_DIR, ROOT / "eval" / "library_memory_corpus"):
            yield from sorted(p for p in base.rglob("*") if p.is_file())

    def test_no_email_addresses_or_home_paths(self):
        files = list(self._files())
        self.assertGreaterEqual(len(files), 15 + 3 + 1 + 3)
        for path in files:
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertIsNone(self.EMAIL.search(text))
                self.assertIsNone(self.HOME_PATH.search(text))


class TestCaptures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = lme.load_corpus()
        cls.captures = {mode: _capture(mode) for mode in MODES}

    def test_captures_match_corpus_and_floors(self):
        from mcp.memory_tool import MIN_SEMANTIC_SCORE, SOFT_DISTANCE_CUTOFF
        from mcp.rag_tool import MIN_RAG_SEMANTIC_SCORE

        fingerprint = lme.corpus_fingerprint()
        for mode, cap in self.captures.items():
            with self.subTest(mode=mode):
                self.assertEqual(cap["schema"], lme.CAPTURE_SCHEMA)
                self.assertEqual(cap["embedding_mode"], mode)
                self.assertEqual(cap["corpus_fingerprint"], fingerprint, "corpus changed: re-capture")
                self.assertEqual(
                    cap["floors"],
                    {
                        "MIN_RAG_SEMANTIC_SCORE": MIN_RAG_SEMANTIC_SCORE,
                        "MIN_SEMANTIC_SCORE": MIN_SEMANTIC_SCORE,
                        "SOFT_DISTANCE_CUTOFF": SOFT_DISTANCE_CUTOFF,
                    },
                )
                self.assertEqual(cap["library_tool_top_k"], lme.library_tool_top_k())
                self.assertEqual(set(cap["cases"]), {c["id"] for c in self.corpus["cases"]})

    def test_replay_reproduces_what_production_kept(self):
        for mode, cap in self.captures.items():
            for case in self.corpus["cases"]:
                for run in lme.case_runs(case):
                    with self.subTest(mode=mode, case=case["id"], run=run):
                        self.assertEqual(
                            lme.replay_run(cap, case, run),
                            cap["cases"][case["id"]]["runs"][run]["kept"],
                        )

    def test_current_baseline_is_todays_replay(self):
        for mode, cap in self.captures.items():
            with self.subTest(mode=mode):
                baseline = lme.load_json(BASELINE_DIR / f"{CURRENT_BASELINE}_{mode}.json")
                report = json.loads(json.dumps(lme.build_report(cap, self.corpus, stage=baseline["stage"])))
                self.assertEqual(report, baseline)
                self.assertEqual(lme.compare_reports(report, baseline), [])

    def test_replay_refuses_a_changed_query(self):
        cap = self.captures["balanced"]
        case = copy.deepcopy(next(c for c in self.corpus["cases"] if c["id"] == "lib_omega_only_sky_blue"))
        case["query"] = "Why is the sea blue?"
        with self.assertRaises(lme.ReplayMismatch):
            lme.replay_run(cap, case, lme.RUN_LIBRARY_CHAT)


class TestCorpusFingerprint(unittest.TestCase):
    def test_windows_checkout_gives_the_recorded_fingerprint(self):
        """Git on Windows checks the corpus out with CRLF line endings (``core.autocrlf``)."""
        from unittest import mock

        read_bytes = Path.read_bytes

        def crlf_checkout(path):
            data = read_bytes(path)
            if b"\r" in data:
                return data
            return data.replace(b"\n", b"\r\n")

        recorded = _capture("balanced")["corpus_fingerprint"]
        with mock.patch.object(Path, "read_bytes", crlf_checkout):
            self.assertEqual(lme.corpus_fingerprint(), recorded)
        self.assertEqual(lme.corpus_fingerprint(), recorded)


class TestS33Findings(unittest.TestCase):
    """The S33 findings, read from the S33 baselines (they stay true after later stages)."""

    def test_v63_distance_is_squared_l2(self):
        # The documents table uses LanceDB's default metric: _distance = 2 - 2cos on unit vectors,
        # so the floors' "1 - _distance" is 2cos - 1, not the cosine. It runs in a child process:
        # several test files install a lancedb stub at import when the real one is not loaded yet.
        out = subprocess.run(
            [sys.executable, "-c", _V63_PROBE, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        hits = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(len(hits), 3)
        for cos, distance in hits:
            self.assertAlmostEqual(distance, 2 - 2 * cos, places=4)
            self.assertAlmostEqual(lme.cosine_from_distance(distance), cos, places=4)

    def test_v64_library_floor_drops_nothing_on_balanced(self):
        report = _s33("balanced")
        for run in lme.LIBRARY_RUNS:
            summary = report["summary"][run]
            self.assertEqual(summary["floor"]["below_floor"], {"relevant": 0, "partial": 0, "irrelevant": 0})
            self.assertGreater(summary["scores"]["semantic"]["irrelevant"]["min"], 0.30)
            abstain = [r for r in report["cases"] if r["run"] == run and r["expect"] == "insufficient"]
            self.assertEqual(len(abstain), 13)
            self.assertTrue(all(r["kept"] for r in abstain), "an abstain case abstained")
        t41 = _case_report(report, "lib_omega_only_sky_blue", lme.RUN_LIBRARY_CHAT)
        self.assertEqual(len(t41["kept"]), 5)

    def test_v64_fast_and_power_floor_cuts_relevant_chunks(self):
        expected = {"fast": (16, 498, 5), "power": (16, 604, 12)}
        for mode, (relevant, irrelevant, emptied) in expected.items():
            with self.subTest(mode=mode):
                report = _s33(mode)
                below = report["summary"][lme.RUN_LIBRARY_CHAT]["floor"]["below_floor"]
                self.assertEqual((below["relevant"], below["irrelevant"]), (relevant, irrelevant))
                answerable_emptied = [
                    r
                    for r in report["cases"]
                    if r["run"] == lme.RUN_LIBRARY_CHAT and r["expect"] != "insufficient" and not r["kept"]
                ]
                self.assertEqual(len(answerable_emptied), emptied)

    def test_v65_rag_returns_memory_rows(self):
        report = _s33("balanced")
        for run, rows, cases in ((lme.RUN_LIBRARY_CHAT, 48, 33), (lme.RUN_LIBRARY_TOOL, 35, 30)):
            with self.subTest(run=run):
                self.assertEqual(report["summary"][run]["memory_rows_kept"], rows)
                hit = [r for r in report["cases"] if r["run"] == run and r["memory_rows_kept"]]
                self.assertEqual(len(hit), cases)

    def test_v68_memory_floor_depends_on_the_embedder(self):
        expected = {"balanced": 0, "fast": 11, "power": 12}
        for mode, below in expected.items():
            with self.subTest(mode=mode):
                floor = _s33(mode)["summary"]["memory_recall"]["floor"]
                total = floor["below_floor"]["relevant"] + floor["above_floor"]["relevant"]
                self.assertEqual((floor["below_floor"]["relevant"], total), (below, 13))


if __name__ == "__main__":
    unittest.main()
