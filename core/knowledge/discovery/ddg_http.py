"""Browser-like HTTP session for DuckDuckGo HTML discovery.

DuckDuckGo challenges requests whose TLS/HTTP2 fingerprint does not match a real
browser, regardless of the User-Agent header. When ``curl_cffi`` is installed the
session impersonates Chrome end-to-end (TLS, HTTP/2 settings, header order and
client hints) and behaves like a person using the page: it opens the search page
once, keeps cookies, and submits the search form from that page.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Qube.Knowledge.Discovery.DdgHttp")

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_HTML_ORIGIN = "https://html.duckduckgo.com"

DEFAULT_IMPERSONATE_TARGET = "chrome"
DEFAULT_SESSION_MAX_AGE_SEC = 1800.0
DEFAULT_SESSION_MAX_SEARCHES = 25
DEFAULT_REQUEST_TIMEOUT_SEC = 8.0

_FALLBACK_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class DdgHttpResponse:
    status_code: int
    text: str
    transport: str
    # Raw ``Retry-After`` header (S12: bounded backoff on HTTP 429).
    retry_after: str | None = None


def browser_impersonation_available() -> bool:
    if str(os.getenv("QUBE_DDG_IMPERSONATE", "1")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    try:
        import curl_cffi  # noqa: F401
    except Exception:
        return False
    return True


def _impersonate_target() -> str:
    return (
        str(os.getenv("QUBE_DDG_IMPERSONATE_TARGET") or "").strip()
        or DEFAULT_IMPERSONATE_TARGET
    )


def _requests_accept_encoding() -> str:
    try:
        import brotli  # noqa: F401

        return "gzip, deflate, br"
    except Exception:
        return "gzip, deflate"


class _DdgBrowserSession:
    """One logical browsing session: cookies + warm-up state + age limits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Any = None
        self._transport = ""
        self._created_at = 0.0
        self._searches = 0
        self._warmed = False
        self._warned_fallback = False

    def reset(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        session = self._session
        self._session = None
        self._transport = ""
        self._created_at = 0.0
        self._searches = 0
        self._warmed = False
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def _expired_locked(self) -> bool:
        if self._session is None:
            return True
        if time.time() - self._created_at > DEFAULT_SESSION_MAX_AGE_SEC:
            return True
        return self._searches >= DEFAULT_SESSION_MAX_SEARCHES

    def _ensure_session_locked(self) -> None:
        if not self._expired_locked():
            return
        self._close_locked()
        if browser_impersonation_available():
            from curl_cffi import requests as curl_requests

            self._session = curl_requests.Session(impersonate=_impersonate_target())
            self._transport = f"curl_cffi:{_impersonate_target()}"
        else:
            import requests

            if not self._warned_fallback:
                logger.warning(
                    "[DDG] curl_cffi unavailable; using plain requests "
                    "(DuckDuckGo is much more likely to show bot challenges)."
                )
                self._warned_fallback = True
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": _FALLBACK_USER_AGENT,
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;q=0.9,"
                        "*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": _requests_accept_encoding(),
                    "Upgrade-Insecure-Requests": "1",
                }
            )
            self._transport = "requests"
        self._created_at = time.time()

    def _warm_up_locked(self, timeout: float) -> None:
        """Open the search page once per session, as a person would before searching."""
        if self._warmed:
            return
        self._warmed = True
        try:
            self._session.get(DDG_HTML_ENDPOINT, timeout=timeout)
        except Exception as exc:
            logger.debug("[DDG] warm-up GET failed (continuing): %s", exc)

    def search(self, query: str, *, timeout: float) -> DdgHttpResponse:
        with self._lock:
            self._ensure_session_locked()
            self._warm_up_locked(timeout)
            headers = {
                "Origin": DDG_HTML_ORIGIN,
                "Referer": DDG_HTML_ENDPOINT,
                "Sec-Fetch-Site": "same-origin",
            }
            if self._transport == "requests":
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            self._searches += 1
            response = self._session.post(
                DDG_HTML_ENDPOINT,
                data={"q": query},
                headers=headers,
                timeout=timeout,
            )
            headers = getattr(response, "headers", None) or {}
            try:
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
            except Exception:
                retry_after = None
            return DdgHttpResponse(
                status_code=int(response.status_code),
                text=response.text or "",
                transport=self._transport,
                retry_after=str(retry_after) if retry_after else None,
            )


_browser_session = _DdgBrowserSession()


def ddg_html_search(
    query: str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SEC,
) -> DdgHttpResponse:
    """Submit one DuckDuckGo HTML search through the shared browser-like session."""
    return _browser_session.search(query, timeout=timeout)


def reset_ddg_browser_session() -> None:
    """Drop cookies and connection state; the next search starts a fresh visit."""
    _browser_session.reset()
