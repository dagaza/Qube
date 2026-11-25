"""Plan S21: the scholarly merge fuses chapter-numbered duplicate titles.

A book chapter indexed as "2 EVIDENCE FOR DARK MATTER FROM ASTRONOMY" and the
same work without the number passed the DOI / title dedupe in the S8 live run.
``adapter_collection._normalize_title`` now strips a leading section number
when at least four words remain.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import quality_eval as qe  # noqa: E402
from core.knowledge.adapter_collection import _normalize_title, first_seen_unique, rrf_merge  # noqa: E402
from core.knowledge.scientific_adapters import ROUTE_NO_DISCIPLINE, ROUTE_PRODUCTION  # noqa: E402


def _before_s21(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


class NormalizeTitleTests(unittest.TestCase):
    def test_leading_section_number_is_stripped(self) -> None:
        plain = _normalize_title("Evidence for Dark Matter from Astronomy")
        for numbered in (
            "2 EVIDENCE FOR DARK MATTER FROM ASTRONOMY",
            "2. Evidence for dark matter from astronomy",
            "2.1 Evidence for Dark Matter from Astronomy",
            "  12 Evidence for dark matter from astronomy",
        ):
            with self.subTest(title=numbered):
                self.assertEqual(_normalize_title(numbered), plain)

    def test_titles_that_only_start_with_a_number_keep_it(self) -> None:
        for title in (
            "10-K — Apple Inc.",
            "27-hour battery life: Framework Laptop 13 Pro beats Apple MacBook Pro 14",
            "2001: A Space Odyssey",
            "100 Years of Solitude",
            "3.15_LEARN.TIME.TEACHER.STUDY — Last study on effective learning time",
            "2 Methods",
            "4 Results and discussion",
            "Evidence for dark matter",
        ):
            with self.subTest(title=title):
                self.assertEqual(_normalize_title(title), _before_s21(title))

    def test_different_works_stay_apart(self) -> None:
        self.assertNotEqual(_normalize_title("2 Methods for measuring dark matter"),
                            _normalize_title("Methods for measuring dark energy"))
        self.assertNotEqual(_normalize_title("2 Methods"), _normalize_title("Methods"))


class MergeTests(unittest.TestCase):
    def test_rrf_merge_fuses_the_chapter_with_its_book(self) -> None:
        crossref = [{"title": "2 EVIDENCE FOR DARK MATTER FROM ASTRONOMY", "doi": "10.1/ch2", "_adapter": "crossref"}]
        openalex = [{"title": "Evidence for Dark Matter from Astronomy", "doi": "10.1/book", "_adapter": "openalex"},
                    {"title": "Another paper", "_adapter": "openalex"}]
        merged = rrf_merge([openalex, crossref])
        self.assertEqual([r["_adapter"] for r in merged], ["openalex", "openalex"])
        self.assertEqual(merged[0]["title"], "Evidence for Dark Matter from Astronomy")
        self.assertGreater(merged[0]["_rrf_score"], merged[1]["_rrf_score"])
        self.assertEqual(len(first_seen_unique(openalex + crossref)), 2)

    def test_numbered_generic_chapters_are_not_fused(self) -> None:
        rows = [{"title": "2 Methods", "doi": "10.1/a"}, {"title": "Methods", "doi": "10.1/b"}]
        self.assertEqual(len(rrf_merge([rows])), 2)


class CorpusTests(unittest.TestCase):
    def test_false_consensus_general_route_holds_one_copy(self) -> None:
        # The corpus's one chapter-numbered pair: both irrelevant rows, found only
        # on the general route (Crossref), so the default replays are unchanged.
        fx = qe.load_json("eval/fixtures/knowledge_quality/s13_trial_false_consensus.json")
        for mode in ("production", "standalone"):
            with self.subTest(mode=mode):
                general = [r["title"] for r in qe.fixture_pool(fx, query_mode=mode, route_arm=ROUTE_NO_DISCIPLINE)]
                self.assertEqual(sum("Scientific Consensus &amp; the Counterfactual" in t for t in general), 1)
                unmerged = qe.fixture_pool(fx, query_mode=mode, route_arm=ROUTE_NO_DISCIPLINE, merge=False)
                self.assertEqual(len(unmerged), len(general) + 1)
                production = qe.fixture_pool(fx, query_mode=mode, route_arm=ROUTE_PRODUCTION)
                self.assertEqual(len(production), len(qe.fixture_pool(fx, query_mode=mode, route_arm=ROUTE_PRODUCTION, merge=False)))


if __name__ == "__main__":
    unittest.main()
