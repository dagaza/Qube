"""Browser-like DuckDuckGo HTTP session (offline; no live DDG)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.knowledge.discovery import ddg_http
from core.knowledge.discovery.backoff import reset_discovery_backoff
from core.knowledge.discovery.pacing import reset_discovery_pacing
from core.knowledge.discovery.session_budget import reset_ddg_session_budget
from mcp.internet_tool import execute_internet_search

SERP_HTML = """
<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Example</a>
<a class="result__snippet">Example snippet.</a>
"""

CHALLENGE_HTML = '<div class="anomaly-modal__box"><img class="anomaly-modal__image"></div>'


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeSession:
    instances: list["_FakeSession"] = []

    def __init__(self, *, post_status: int = 200, post_text: str = SERP_HTML) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.closed = False
        self._post_status = post_status
        self._post_text = post_text
        _FakeSession.instances.append(self)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return _FakeResponse(200, "<html></html>")

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse(self._post_status, self._post_text)

    def close(self):
        self.closed = True


class TestDdgBrowserSession(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSession.instances.clear()
        ddg_http.reset_ddg_browser_session()
        reset_discovery_backoff()
        reset_discovery_pacing()
        reset_ddg_session_budget()
        self._env = patch.dict(
            "os.environ",
            {"QUBE_DISCOVERY_PACE_ENABLED": "0"},
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        ddg_http.reset_ddg_browser_session()

    def _patch_session(self, **kwargs):
        def _factory():
            return _FakeSession(**kwargs)

        return patch.object(ddg_http._DdgBrowserSession, "_ensure_session_locked", _ensure_with(_factory))

    def test_warm_up_once_then_form_post_reuses_cookies(self) -> None:
        with self._patch_session():
            ddg_http.ddg_html_search("first query")
            ddg_http.ddg_html_search("second query")
        self.assertEqual(len(_FakeSession.instances), 1)
        methods = [c[0] for c in _FakeSession.instances[0].calls]
        self.assertEqual(methods, ["GET", "POST", "POST"])
        _, url, kwargs = _FakeSession.instances[0].calls[1]
        self.assertEqual(url, ddg_http.DDG_HTML_ENDPOINT)
        self.assertEqual(kwargs["data"], {"q": "first query"})
        self.assertEqual(kwargs["headers"]["Referer"], ddg_http.DDG_HTML_ENDPOINT)
        self.assertNotIn("User-Agent", kwargs["headers"])

    def test_challenge_resets_session_for_fresh_cookies(self) -> None:
        with self._patch_session(post_status=202, post_text=CHALLENGE_HTML):
            response = execute_internet_search("blocked query")
        self.assertEqual(response.inspection["response_kind"], "bot_challenge")
        self.assertTrue(_FakeSession.instances[0].closed)
        self.assertIsNone(ddg_http._browser_session._session)

    def test_serp_parsed_through_browser_session(self) -> None:
        with self._patch_session():
            response = execute_internet_search("ok query")
        self.assertEqual(response.inspection["response_kind"], "serp")
        self.assertEqual(response.rows[0]["url"], "https://example.com/a")

    def test_impersonation_can_be_disabled_by_env(self) -> None:
        with patch.dict("os.environ", {"QUBE_DDG_IMPERSONATE": "0"}):
            self.assertFalse(ddg_http.browser_impersonation_available())

    def test_requests_fallback_does_not_advertise_brotli_without_decoder(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _no_brotli(name, *args, **kwargs):
            if name == "brotli":
                raise ImportError("no brotli")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_brotli):
            self.assertEqual(ddg_http._requests_accept_encoding(), "gzip, deflate")


def _ensure_with(factory):
    def _ensure(self) -> None:
        if not self._expired_locked():
            return
        self._close_locked()
        self._session = factory()
        self._transport = "fake"
        self._created_at = ddg_http.time.time()

    return _ensure


if __name__ == "__main__":
    unittest.main()
