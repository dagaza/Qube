"""Authority tiers for web and encyclopedia sources."""

from __future__ import annotations

from urllib.parse import urlparse

TIER_WIKIPEDIA = 0.95
TIER_GOV = 0.88
TIER_EDU = 0.82
TIER_DEFAULT = 0.35

_WIKIPEDIA_HOSTS = frozenset(
    {
        "wikipedia.org",
        "www.wikipedia.org",
        "en.wikipedia.org",
        "wikimedia.org",
    }
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def is_wikipedia_url(url: str | None) -> bool:
    host = _host(url or "")
    if not host:
        return False
    return host in _WIKIPEDIA_HOSTS or host.endswith(".wikipedia.org")


def is_gov_url(url: str | None) -> bool:
    host = _host(url or "")
    return bool(host.endswith(".gov") or host.endswith(".gov.uk"))


def is_edu_url(url: str | None) -> bool:
    host = _host(url or "")
    return bool(host.endswith(".edu") or host.endswith(".ac.uk"))


def is_allowlisted_url(url: str | None) -> bool:
    """Trusted-knowledge allowlist: Wikipedia, government, and academic domains."""
    return is_wikipedia_url(url) or is_gov_url(url) or is_edu_url(url)


def authority_score_for_url(url: str | None) -> float:
    if is_wikipedia_url(url):
        return TIER_WIKIPEDIA
    if is_gov_url(url):
        return TIER_GOV
    if is_edu_url(url):
        return TIER_EDU
    return TIER_DEFAULT


def tier_label_for_url(url: str | None) -> str:
    if is_wikipedia_url(url):
        return "wikipedia"
    if is_gov_url(url):
        return "government"
    if is_edu_url(url):
        return "academic"
    return "web"
