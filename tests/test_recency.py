"""Recency need and source freshness (plan S7, R9)."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.ranking.recency import (  # noqa: E402
    FRESH,
    STALE,
    UNDATED,
    SourceDate,
    annotate_freshness,
    assess_freshness,
    mentioned_date,
    parse_date_string,
    url_date,
)
from core.retrieval_query_cleanup import (  # noqa: E402
    NEWS_WINDOW_DAYS,
    RECENCY_WINDOW_DAYS,
    recency_need,
)

TODAY = date(2026, 11, 23)


class RecencyNeedTests(unittest.TestCase):
    def test_table(self) -> None:
        cases = [
            # (question, expected window or None, reason)
            ("What is the latest US inflation rate?", RECENCY_WINDOW_DAYS, "recent_wording"),
            ("What is the newest iPhone model?", RECENCY_WINDOW_DAYS, "recent_wording"),
            ("Tesla latest quarterly report 10-Q", RECENCY_WINDOW_DAYS, "recent_wording"),
            ("What is the current ECB deposit facility rate?", RECENCY_WINDOW_DAYS, "current"),
            ("Who is the current UK prime minister?", RECENCY_WINDOW_DAYS, "current"),
            ("Is it currently legal to fly drones here?", RECENCY_WINDOW_DAYS, "current"),
            ("What is the price of gold?", RECENCY_WINDOW_DAYS, "market"),
            ("euro to dollar exchange rate", RECENCY_WINDOW_DAYS, "market"),
            ("Who won the match?", RECENCY_WINDOW_DAYS, "market"),
            ("Best laptops 2026", RECENCY_WINDOW_DAYS, "recent_year"),
            ("Roth IRA limits for 2025", RECENCY_WINDOW_DAYS, "recent_year"),
            ("News about the election today", NEWS_WINDOW_DAYS, "news"),
            ("What happened this week in tech?", NEWS_WINDOW_DAYS, "news"),
            ("breaking headlines", NEWS_WINDOW_DAYS, "news"),
            # Not recency: physics / ocean "current", concepts, history, plain facts.
            ("What is electric current?", None, None),
            ("How is current density defined?", None, None),
            ("What are ocean currents?", None, None),
            ("the current flowing through a resistor", None, None),
            ("price elasticity of demand", None, None),
            ("What is a normal heart rate?", None, None),
            ("Who won the 2022 World Cup?", None, None),
            ("What was the inflation rate in 2019?", None, None),
            ("What is a Roth IRA?", None, None),
            ("How much does it cost to install a heat pump?", None, None),
            ("", None, None),
        ]
        for question, window, reason in cases:
            with self.subTest(question=question):
                need = recency_need(question, today=TODAY)
                if window is None:
                    self.assertIsNone(need)
                else:
                    self.assertIsNotNone(need)
                    self.assertEqual((need.window_days, need.reason), (window, reason))

    def test_recent_year_follows_today(self) -> None:
        self.assertIsNotNone(recency_need("Best laptops 2026", today=date(2027, 3, 1)))
        self.assertIsNone(recency_need("Best laptops 2026", today=date(2028, 3, 1)))


class SourceDateTests(unittest.TestCase):
    def test_parse_date_string(self) -> None:
        self.assertEqual(parse_date_string("2026-10-29"), SourceDate(2026, 10, 29))
        self.assertEqual(parse_date_string("2026-10"), SourceDate(2026, 10))
        self.assertEqual(parse_date_string("2026"), SourceDate(2026))
        self.assertEqual(parse_date_string("29 October 2026"), SourceDate(2026, 10, 29))
        self.assertIsNone(parse_date_string(""))
        self.assertIsNone(parse_date_string("n/a"))

    def test_url_dates(self) -> None:
        cases = [
            ("https://www.cnbc.com/2026/10/29/ecb-holds.html", SourceDate(2026, 10, 29)),
            ("https://blog.python.org/2026/10/python-3150-final/", SourceDate(2026, 10)),
            ("https://example.com/news/2024/story", SourceDate(2024)),
            ("https://example.com/2026-03-04-release-notes", SourceDate(2026, 3, 4)),
            ("https://example.com/item/9789240034228", None),
            ("https://example.com/docs/3.14/whatsnew", None),
            ("https://example.com/2030/01/future/", None),  # after today: ignored
        ]
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(url_date(url, today=TODAY), expected)

    def test_mentioned_dates_take_the_latest_non_future_date(self) -> None:
        cases = [
            ("CPI-U +0.2% in Oct 2026", SourceDate(2026, 10)),
            ("data from 1999-01-01 to 2026-11-23", SourceDate(2026, 11, 23)),
            ("rate 2.50 % since 16 September 2026", SourceDate(2026, 9, 16)),
            ("released November 10, 2026", SourceDate(2026, 11, 10)),
            ("Best iPhone 2026", SourceDate(2026)),
            ("limits for 2025-2026", SourceDate(2026)),
            ("forecast for 2027-2030", None),  # only future years
            ("Python 3.14 release", None),
            ("ISBN 9789240034228", None),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(mentioned_date(text, today=TODAY), expected)

    def test_ambiguous_slash_date_keeps_the_earlier_reading(self) -> None:
        # 03/04/2026 could be 3 April or 4 March: never overstate freshness.
        self.assertEqual(mentioned_date("updated 03/04/2026", today=TODAY), SourceDate(2026, 3, 4))
        self.assertEqual(mentioned_date("since 16/09/2026", today=TODAY), SourceDate(2026, 9, 16))


class FreshnessTests(unittest.TestCase):
    def _f(self, row, window=365):
        return assess_freshness(row, window_days=window, today=TODAY)

    def test_published_date_proves_fresh_or_stale(self) -> None:
        self.assertEqual(self._f({"publication_date": "2026-11-03"}).status, FRESH)
        self.assertEqual(self._f({"publication_date": "2023-05-01"}).status, STALE)
        self.assertEqual(self._f({"url": "https://x.org/2026/10/29/a"}).status, FRESH)
        self.assertEqual(self._f({"url": "https://x.org/2021/01/a"}).status, STALE)

    def test_year_precision_counts_the_latest_day_for_published_dates(self) -> None:
        # "2025" may mean 31 Dec 2025, inside a 12-month window from 23 Nov 2026.
        self.assertEqual(self._f({"publication_date": "2025"}).status, FRESH)
        self.assertEqual(self._f({"publication_date": "2024"}).status, STALE)

    def test_mentions_prove_fresh_but_never_stale(self) -> None:
        row = {"title": "Inflation", "snippet": "CPI rose in Oct 2026."}
        self.assertEqual(self._f(row).status, FRESH)
        old = {"title": "Inflation in 2019", "snippet": "CPI rose in March 2019."}
        f = self._f(old)
        self.assertEqual(f.status, UNDATED)
        self.assertEqual(f.source_date, SourceDate(2019, 3))

    def test_mention_newer_than_url_date_wins(self) -> None:
        row = {"url": "https://x.org/2021/05/guide", "snippet": "Updated 2 November 2026."}
        self.assertEqual(self._f(row).status, FRESH)

    def test_news_window_needs_a_recent_day(self) -> None:
        self.assertEqual(self._f({"snippet": "Nov 2026 update"}, window=7).status, UNDATED)
        self.assertEqual(self._f({"snippet": "20 November 2026: vote"}, window=7).status, FRESH)
        self.assertEqual(self._f({"url": "https://x.org/2026/11/01/a"}, window=7).status, STALE)

    def test_no_date_is_undated(self) -> None:
        self.assertEqual(self._f({"title": "Roth IRA", "snippet": "A retirement account."}).status, UNDATED)

    def test_annotate_sets_row_fields(self) -> None:
        rows = [{"url": "https://x.org/2021/05/a"}, {"snippet": "Oct 2026"}, {"title": "t"}]
        annotate_freshness(rows, window_days=365, today=TODAY)
        self.assertEqual([r["_freshness"] for r in rows], [STALE, FRESH, UNDATED])
        self.assertEqual(rows[0]["_source_date"], "2021-05")
        self.assertEqual(rows[0]["_source_date_basis"], "url")
        self.assertEqual(rows[1]["_source_date_basis"], "mentioned")
        self.assertIsNone(rows[2]["_source_date"])


if __name__ == "__main__":
    unittest.main()
