"""DuckDuckGo HTML inspection and sentinel classification."""
from __future__ import annotations

import unittest

from core.knowledge.adapters.duckduckgo import failure_sentinel_reason, is_failure_sentinel
from mcp.internet_tool import (
    DDG_BOT_CHALLENGE_SNIPPET,
    DDG_NO_RESULTS_SNIPPET,
    inspect_ddg_html_response,
    parse_ddg_html_results,
    score_ddg_bot_challenge,
)

SAMPLE_SERP_HTML = """
<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">Example Title</a>
<a class="result__snippet">Example snippet text about ducks.</a>
"""

SAMPLE_BOT_CHALLENGE_HTML = """
<div class="anomaly-modal__box" data-index="1">
  <label for="image-check_abc" data-testid="anomaly-modal-tile-1">
    <img class="anomaly-modal__image" src="../assets/anomaly/images/challenge/abc.jpg">
  </label>
</div>
"""

SAMPLE_VERIFY_HUMAN_HTML = """
<html><body>
<p>Please verify you are human before continuing.</p>
<p>Unusual traffic from your network.</p>
</body></html>
"""


class TestInternetToolDdg(unittest.TestCase):
    def test_inspect_serp_html(self) -> None:
        inspection = inspect_ddg_html_response(
            SAMPLE_SERP_HTML,
            http_status=200,
            max_results=3,
        )
        self.assertEqual(inspection["response_kind"], "serp")
        self.assertEqual(inspection["parsed_rows"], 1)
        self.assertEqual(inspection["urls_with_http"], 1)
        self.assertEqual(inspection["bot_challenge_signals"], [])

    def test_inspect_bot_challenge_html(self) -> None:
        inspection = inspect_ddg_html_response(
            SAMPLE_BOT_CHALLENGE_HTML,
            http_status=202,
            max_results=3,
        )
        self.assertEqual(inspection["response_kind"], "bot_challenge")
        self.assertGreater(len(inspection["bot_challenge_signals"]), 0)

    def test_score_bot_challenge_with_verify_human_keywords(self) -> None:
        is_bot, signals = score_ddg_bot_challenge(
            SAMPLE_VERIFY_HUMAN_HTML,
            http_status=202,
            link_matches=0,
            snippet_matches=0,
        )
        self.assertTrue(is_bot)
        self.assertIn("http_202", signals)
        self.assertIn("no_serp_markers", signals)

    def test_score_not_bot_when_serp_present(self) -> None:
        is_bot, signals = score_ddg_bot_challenge(
            SAMPLE_SERP_HTML,
            http_status=200,
            link_matches=1,
            snippet_matches=1,
        )
        self.assertFalse(is_bot)
        self.assertEqual(signals, [])

    def test_inspect_empty_parse_html(self) -> None:
        inspection = inspect_ddg_html_response(
            "<html><body><p>No results here</p></body></html>",
            http_status=200,
            max_results=3,
        )
        self.assertEqual(inspection["response_kind"], "empty_parse")
        self.assertEqual(inspection["parsed_rows"], 0)

    def test_parse_ddg_html_results_extracts_url(self) -> None:
        rows = parse_ddg_html_results(SAMPLE_SERP_HTML, max_results=3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], "https://example.com/article")
        self.assertIn("Example snippet", rows[0]["snippet"])

    def test_is_failure_sentinel_recognizes_bot_challenge(self) -> None:
        rows = [{"title": "", "snippet": DDG_BOT_CHALLENGE_SNIPPET}]
        self.assertTrue(is_failure_sentinel(rows))

    def test_is_failure_sentinel_recognizes_empty_results(self) -> None:
        rows = [{"title": "", "snippet": DDG_NO_RESULTS_SNIPPET}]
        self.assertTrue(is_failure_sentinel(rows))

    def test_is_failure_sentinel_false_for_real_rows(self) -> None:
        rows = [{"title": "A", "snippet": "Real result", "url": "https://example.com"}]
        self.assertFalse(is_failure_sentinel(rows))

    def test_failure_sentinel_reason_bot_challenge(self) -> None:
        rows = [{"title": "", "snippet": DDG_BOT_CHALLENGE_SNIPPET}]
        self.assertEqual(failure_sentinel_reason(rows), "ddg_bot_challenge")


if __name__ == "__main__":
    unittest.main()
