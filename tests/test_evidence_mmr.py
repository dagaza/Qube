"""MMR evidence selection (core.knowledge.ranking.diversity)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.ranking.diversity import mmr_select_rows  # noqa: E402


def _row(title: str, rel: float, adapter: str = "a") -> dict:
    return {"title": title, "snippet": "", "_adapter": adapter, "_scientific_relevance": rel}


class MmrRelevanceAlignmentTests(unittest.TestCase):
    def test_candidate_uses_its_own_relevance_after_picks(self) -> None:
        # Regression: relevance was looked up by position in the shrinking
        # ``remaining`` list, so after the first pick the 0.10 row inherited the
        # 0.95 row's score and displaced it.
        rows = [
            _row("semaglutide cardiovascular outcomes trial", 1.0),
            _row("semaglutide cardiovascular outcomes trials", 0.95),
            _row("ant colony routing", 0.10),
        ]
        picked = [r["title"] for r in mmr_select_rows(rows, max_results=2)]
        self.assertEqual(picked, [rows[0]["title"], rows[1]["title"]])

    def test_diversity_still_wins_between_close_scores(self) -> None:
        rows = [
            _row("semaglutide cardiovascular outcomes trial", 1.0),
            _row("semaglutide cardiovascular outcomes trials", 0.62),
            _row("glp-1 agonists and heart failure hospitalisation", 0.60, adapter="b"),
        ]
        picked = [r["title"] for r in mmr_select_rows(rows, max_results=2)]
        self.assertEqual(picked[1], rows[2]["title"])

    def test_small_pools_pass_through(self) -> None:
        rows = [_row("x", 0.1), _row("y", 0.9)]
        self.assertEqual(mmr_select_rows(rows, max_results=3), rows)
        self.assertEqual(mmr_select_rows([], max_results=3), [])


if __name__ == "__main__":
    unittest.main()
