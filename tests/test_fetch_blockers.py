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


_ARTICLE_BODY = "<p>" + ("Heat pumps move heat from a cold space to a warm one. " * 60) + "</p>"


class TestBlockerFalsePositives(unittest.TestCase):
    """S12: markup seen on real article pages that used to be discarded."""

    def test_cloudflare_scripts_on_a_real_page_are_not_a_challenge(self) -> None:
        for script in (
            '<script type="module" src="https://static.cloudflareinsights.com/beacon.min.js"></script>',
            '<link rel="preconnect" href="https://cdnjs.cloudflare.com">',
            "<script>var u='/cdn-cgi/challenge-platform/scripts/jsd/main.js?ray=1';</script>",
            '<script>window.cfg={"cloudflare":{"version":"1.0.0"}}</script>',
        ):
            with self.subTest(script=script[:40]):
                html = f"<html><head>{script}</head><body>{_ARTICLE_BODY}</body></html>"
                self.assertFalse(detect_cloudflare(html))
                self.assertIsNone(detect_blocker(html, status_code=200, content_type_header="text/html"))

    def test_real_challenge_shapes_still_detected(self) -> None:
        orchestrate = (
            "<html><head><title>Just a moment...</title></head><body>"
            "<noscript>Enable JavaScript and cookies to continue</noscript>"
            "<script>window._cf_chl_opt={cvId:'3'};</script>"
            "<script src='/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1'></script>"
            "</body></html>"
        )
        self.assertTrue(detect_cloudflare(orchestrate))
        blocked = "<html><head><title>Attention Required! | Cloudflare</title></head><body><h1>Sorry, you have been blocked</h1></body></html>"
        self.assertTrue(detect_cloudflare(blocked))

    def test_wikipedia_citation_lock_is_not_a_paywall(self) -> None:
        lock = '<span class="id-lock-subscription" title="paid subscription required"><a href="#">doi</a></span>'
        html = f"<html><body>{_ARTICLE_BODY}{lock}</body></html>"
        self.assertIsNone(detect_blocker(html, status_code=200, content_type_header="text/html"))

    def test_visible_paywall_text_still_detected(self) -> None:
        html = "<html><body><p>Intro.</p><div>Subscribe to read the full article.</div></body></html>"
        self.assertEqual(
            detect_blocker(html, status_code=200, content_type_header="text/html"),
            BlockerReason.PAYWALL,
        )


class TestOversizeTruncation(unittest.TestCase):
    def _resp(self, total: int):
        resp = MagicMock()
        resp.iter_content.return_value = [b"a" * 65536] * (total // 65536)
        return resp

    def test_api_reads_still_raise_over_the_limit(self) -> None:
        from core.knowledge.http_client import _read_bounded_response

        with self.assertRaises(EgressPolicyError):
            _read_bounded_response(self._resp(655360), max_bytes=524288)

    def test_page_fetch_keeps_the_head(self) -> None:
        from core.knowledge.http_client import _read_bounded_response

        resp = self._resp(655360)
        content = _read_bounded_response(resp, max_bytes=524288, truncate=True)
        self.assertEqual(len(content), 524288)
        self.assertTrue(resp.qube_truncated)
        small = self._resp(131072)
        self.assertEqual(len(_read_bounded_response(small, max_bytes=524288, truncate=True)), 131072)
        self.assertIsNot(getattr(small, "qube_truncated", None), True)

    @patch("core.knowledge.fetch.engine.knowledge_get")
    def test_fetch_url_asks_for_truncation(self, mock_get) -> None:
        response = MagicMock()
        response.status_code = 200
        response.url = "https://en.wikipedia.org/wiki/Heat_pump"
        response.headers = {"Content-Type": "text/html"}
        response.text = f"<html><body>{_ARTICLE_BODY}</body></html>"
        response.qube_truncated = True
        mock_get.return_value = response
        result = fetch_url("https://en.wikipedia.org/wiki/Heat_pump")
        self.assertTrue(result.success)
        self.assertTrue(mock_get.call_args.kwargs["truncate_oversize"])
        self.assertTrue(result.raw_metadata.get("truncated"))


if __name__ == "__main__":
    unittest.main()
