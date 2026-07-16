"""Tests for fetch blockers and engine (M2)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.fetch.blockers import detect_blocker, detect_cloudflare  # noqa: E402
from core.knowledge.fetch.engine import fetch_html_string, fetch_url  # noqa: E402
from core.knowledge.fetch.types import BlockerReason  # noqa: E402
from core.knowledge.egress_policy import EgressPolicyError  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "fetch"


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


class TestFetchBlockers(unittest.TestCase):
    def test_cloudflare_fixture_detected(self) -> None:
        html = _read_fixture("cloudflare_challenge.html")
        self.assertTrue(detect_cloudflare(html))
        self.assertEqual(
            detect_blocker(html, status_code=200, content_type_header="text/html"),
            BlockerReason.CLOUDFLARE,
        )

    def test_clean_article_fixture_passes(self) -> None:
        html = _read_fixture("article_clean.html")
        self.assertIsNone(
            detect_blocker(html, status_code=200, content_type_header="text/html")
        )

    def test_fetch_html_string_cloudflare_failure(self) -> None:
        html = _read_fixture("cloudflare_challenge.html")
        result = fetch_html_string(html, url="https://example.com/blocked")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, BlockerReason.CLOUDFLARE)
        self.assertIsNone(result.html)

    def test_fetch_html_string_clean_success(self) -> None:
        html = _read_fixture("article_clean.html")
        result = fetch_html_string(html, url="https://example.com/birds")
        self.assertTrue(result.success)
        self.assertIsNone(result.failure_reason)
        self.assertIn("Dust Bathing", result.html or "")

    @patch("core.knowledge.fetch.engine.knowledge_get")
    def test_fetch_url_success(self, mock_get) -> None:
        response = MagicMock()
        response.status_code = 200
        response.url = "https://example.com/birds"
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.text = _read_fixture("article_clean.html")
        response.reason = "OK"
        response.encoding = "utf-8"
        mock_get.return_value = response

        result = fetch_url("https://example.com/birds")
        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertIn("bird species", (result.html or "").lower())

    @patch("core.knowledge.fetch.engine.knowledge_get")
    def test_fetch_url_egress_blocked(self, mock_get) -> None:
        mock_get.side_effect = EgressPolicyError("blocked by policy")
        result = fetch_url("https://internal.example/page")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, BlockerReason.EGRESS_BLOCKED)


if __name__ == "__main__":
    unittest.main()
