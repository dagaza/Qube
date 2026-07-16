"""Blocker heuristics for fetched HTML — ADR 004 invariant."""

from __future__ import annotations

import re

from core.knowledge.fetch.types import BlockerReason

_CLOUDFLARE_MARKERS = (
    "just a moment",
    "checking your browser",
    "cf-browser-verification",
    "challenge-platform",
    "cloudflare",
    "attention required",
)
_JS_SHELL_MARKERS = (
    '<div id="root"></div>',
    '<div id="app"></div>',
    'id="__next"',
    "data-reactroot",
)
_PAYWALL_MARKERS = (
    "subscribe to read",
    "subscription required",
    "sign in to continue reading",
    "this article is for subscribers",
    "metered paywall",
    "you have reached your limit of free articles",
)
_COOKIE_WALL_MARKERS = (
    "cookie consent",
    "accept cookies",
    "we use cookies",
    "gdpr consent",
    "privacy preference center",
)
_MIN_MAIN_TEXT_CHARS = 120


def _normalized(html: str) -> str:
    return (html or "").lower()


def _visible_text_length(html: str) -> int:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html or "", flags=re.I | re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return len(text)


def detect_cloudflare(html: str) -> bool:
    norm = _normalized(html)
    return any(marker in norm for marker in _CLOUDFLARE_MARKERS)


def detect_js_rendered(html: str) -> bool:
    if not (html or "").strip():
        return True
    norm = _normalized(html)
    if "<noscript>" in norm and "enable javascript" in norm:
        return True
    if any(marker.lower() in norm for marker in _JS_SHELL_MARKERS):
        return _visible_text_length(html) < _MIN_MAIN_TEXT_CHARS
    return False


def detect_cookie_wall(html: str) -> bool:
    norm = _normalized(html)
    if not any(marker in norm for marker in _COOKIE_WALL_MARKERS):
        return False
    return _visible_text_length(html) < 400


def detect_paywall(html: str) -> bool:
    norm = _normalized(html)
    return any(marker in norm for marker in _PAYWALL_MARKERS)


def detect_blocker(
    html: str | None,
    *,
    status_code: int | None = None,
    content_type_header: str | None = None,
) -> str | None:
    """Return a BlockerReason value when the response should not be extracted."""
    if status_code is not None and status_code >= 400:
        return BlockerReason.HTTP_ERROR

    body = html or ""
    if not body.strip():
        return BlockerReason.EMPTY_EXTRACT

    if detect_cloudflare(body):
        return BlockerReason.CLOUDFLARE
    if detect_paywall(body):
        return BlockerReason.PAYWALL
    if detect_cookie_wall(body):
        return BlockerReason.COOKIE_WALL
    if detect_js_rendered(body):
        return BlockerReason.JS_RENDERED

    ctype = (content_type_header or "").lower()
    if ctype and "text/html" not in ctype and "application/xhtml" not in ctype:
        if "text/" not in ctype and "application/json" not in ctype:
            return BlockerReason.EMPTY_EXTRACT

    if _visible_text_length(body) < 40:
        return BlockerReason.EMPTY_EXTRACT

    return None
