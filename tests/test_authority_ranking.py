"""Tests for authority tier ranking."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.ranking.authority import (  # noqa: E402
    authority_score_for_url,
    is_allowlisted_url,
    is_wikipedia_url,
)


class TestAuthorityRanking(unittest.TestCase):
    def test_wikipedia_tier(self) -> None:
        url = "https://en.wikipedia.org/wiki/Bucharest"
        self.assertTrue(is_wikipedia_url(url))
        self.assertTrue(is_allowlisted_url(url))
        self.assertGreaterEqual(authority_score_for_url(url), 0.9)

    def test_gov_edu_allowlist(self) -> None:
        self.assertTrue(is_allowlisted_url("https://www.nasa.gov/about"))
        self.assertTrue(is_allowlisted_url("https://mit.edu/research"))
        self.assertFalse(is_allowlisted_url("https://example.com/page"))

    def test_default_web_tier(self) -> None:
        self.assertLess(authority_score_for_url("https://example.com"), 0.5)


if __name__ == "__main__":
    unittest.main()
