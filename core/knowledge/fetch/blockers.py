"""Blocker heuristics for fetched HTML — ADR 004 invariant."""

from __future__ import annotations

import re

from core.knowledge.fetch.types import BlockerReason

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
    return len(_visible_text(html))


# S12: a challenge interstitial is a short page that says so. The bare word
# "cloudflare" or "challenge-platform" also appears in the scripts Cloudflare
# injects into every page it proxies (Insights beacon, cdnjs, the passive
# /cdn-cgi/challenge-platform/scripts/jsd/ bot script), so matching raw HTML
# discarded 8 of 41 real article pages in the S12 measurement.
_CHALLENGE_TEXT_MARKERS = (
    "just a moment",
    "checking your browser",
    "attention required",
    "verify you are human",
    "enable javascript and cookies to continue",
)
_CHALLENGE_MARKUP_MARKERS = (
    "cf-browser-verification",
    "cf_chl_opt",
    "/cdn-cgi/challenge-platform/h/",
)
_CHALLENGE_MAX_VISIBLE_CHARS = 2000


def _visible_text(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html or "", flags=re.I | re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_cloudflare(html: str) -> bool:
    visible = _visible_text(html)
    if len(visible) >= _CHALLENGE_MAX_VISIBLE_CHARS:
        return False
    if any(marker in visible for marker in _CHALLENGE_TEXT_MARKERS):
        return True
    norm = _normalized(html)
    return any(marker in norm for marker in _CHALLENGE_MARKUP_MARKERS)


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
    """Paywall wording in the page's visible text (S12).

    Attributes are not text: every Wikipedia citation to a paywalled source
    carries ``title="paid subscription required"`` and used to match.
    """
    visible = _visible_text(html)
    return any(marker in visible for marker in _PAYWALL_MARKERS)


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
