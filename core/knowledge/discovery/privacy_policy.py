"""Privacy-first discovery tiers and routing (R6/R7)."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.adapters.brave_search import brave_search_configured
from core.knowledge.discovery.policy import (
    BRAVE_DISCOVERY_PROVIDER_ID,
    PRIMARY_DISCOVERY_PROVIDER_ID,
    WIKIPEDIA_DISCOVERY_PROVIDER_ID,
)
from core.knowledge.discovery.searxng import SEARXNG_DISCOVERY_PROVIDER_ID

TIER_PRIVATE = "private"
TIER_BALANCED = "balanced"
TIER_ENHANCED = "enhanced"
TIER_SEARXNG = "searxng"

VALID_PRIVACY_TIERS = frozenset(
    {TIER_PRIVATE, TIER_BALANCED, TIER_ENHANCED, TIER_SEARXNG}
)

DEFAULT_PRIVACY_TIER = TIER_PRIVATE

_TIER_LABELS: dict[str, str] = {
    TIER_PRIVATE: "Private search (recommended)",
    TIER_BALANCED: "Private + API fallback",
    TIER_ENHANCED: "Maximum reliability",
    TIER_SEARXNG: "Self-hosted SearXNG",
}

_TIER_DESCRIPTIONS: dict[str, str] = {
    TIER_PRIVATE: (
        "DuckDuckGo and Wikipedia only — no API keys, no third-party SERP vendors."
    ),
    TIER_BALANCED: (
        "Same private primary path; Brave Search API is used as fallback when "
        "configured (including site-biased @recipe queries)."
    ),
    TIER_ENHANCED: (
        "Same as balanced — prioritizes configured API fallbacks after DDG blocks. "
        "Optional query alternation stays off by default."
    ),
    TIER_SEARXNG: (
        "Queries go to your SearXNG instance when configured; otherwise behaves "
        "like balanced private search with SearXNG in the fallback chain."
    ),
}


def normalize_privacy_tier(tier: str | None) -> str:
    value = (tier or DEFAULT_PRIVACY_TIER).strip().lower()
    return value if value in VALID_PRIVACY_TIERS else DEFAULT_PRIVACY_TIER


def get_active_privacy_tier() -> str:
    from core.app_settings import get_discovery_privacy_tier

    return normalize_privacy_tier(get_discovery_privacy_tier())


def privacy_tier_label(tier: str | None) -> str:
    return _TIER_LABELS.get(normalize_privacy_tier(tier), normalize_privacy_tier(tier))


def privacy_tier_description(tier: str | None) -> str:
    return _TIER_DESCRIPTIONS.get(normalize_privacy_tier(tier), "")


def discovery_api_fallback_enabled() -> bool:
    """True when API SERP providers (Brave) may be used per user tier."""
    tier = get_active_privacy_tier()
    if tier == TIER_PRIVATE:
        return False
    from core.app_settings import get_discovery_api_fallback_enabled

    return bool(get_discovery_api_fallback_enabled())


def _has_site_bias(site_bias: tuple[str, ...] | None) -> bool:
    return bool(tuple(s.strip() for s in (site_bias or ()) if (s or "").strip()))


@dataclass(frozen=True)
class DiscoveryRoute:
    privacy_tier: str
    primary_id: str
    fallback_ids: tuple[str, ...]
    site_bias_brave_primary: bool = False


def resolve_discovery_route(
    *,
    site_bias: tuple[str, ...] | None = None,
) -> DiscoveryRoute:
    """Choose primary provider and ordered fallbacks for the active privacy tier."""
    from core.knowledge.discovery.searxng import searxng_configured

    tier = get_active_privacy_tier()
    api_fallback = discovery_api_fallback_enabled()
    brave_ready = api_fallback and brave_search_configured()
    searxng_ready = searxng_configured()
    site_bias_active = _has_site_bias(site_bias)

    if site_bias_active and brave_ready:
        fallbacks: list[str] = [PRIMARY_DISCOVERY_PROVIDER_ID, WIKIPEDIA_DISCOVERY_PROVIDER_ID]
        if tier == TIER_SEARXNG and searxng_ready:
            fallbacks.insert(1, SEARXNG_DISCOVERY_PROVIDER_ID)
        return DiscoveryRoute(
            privacy_tier=tier,
            primary_id=BRAVE_DISCOVERY_PROVIDER_ID,
            fallback_ids=_dedupe_preserve_order(fallbacks),
            site_bias_brave_primary=True,
        )

    if tier == TIER_SEARXNG and searxng_ready:
        fallbacks = []
        if brave_ready:
            fallbacks.append(BRAVE_DISCOVERY_PROVIDER_ID)
        fallbacks.append(WIKIPEDIA_DISCOVERY_PROVIDER_ID)
        return DiscoveryRoute(
            privacy_tier=tier,
            primary_id=SEARXNG_DISCOVERY_PROVIDER_ID,
            fallback_ids=tuple(fallbacks),
        )

    fallbacks = []
    if brave_ready:
        fallbacks.append(BRAVE_DISCOVERY_PROVIDER_ID)
    if tier == TIER_SEARXNG and searxng_ready:
        fallbacks.append(SEARXNG_DISCOVERY_PROVIDER_ID)
    fallbacks.append(WIKIPEDIA_DISCOVERY_PROVIDER_ID)

    return DiscoveryRoute(
        privacy_tier=tier,
        primary_id=PRIMARY_DISCOVERY_PROVIDER_ID,
        fallback_ids=_dedupe_preserve_order(fallbacks),
    )


def bot_challenge_fallback_chain(
    *,
    site_bias: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Ordered fallback providers after primary failure."""
    return resolve_discovery_route(site_bias=site_bias).fallback_ids


def _dedupe_preserve_order(provider_ids: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for pid in provider_ids:
        key = (pid or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def what_leaves_device_lines() -> list[str]:
    """Plain-language bullets for Settings help panel."""
    tier = get_active_privacy_tier()
    lines = [
        f"Active tier: {privacy_tier_label(tier)}",
        "Page fetches (after URL discovery) go directly to the destination websites.",
    ]
    if tier == TIER_PRIVATE:
        lines.extend(
            [
                "SERP discovery: DuckDuckGo HTML (no API key) and Wikipedia API.",
                "No queries are sent to Brave, Bing, or other commercial search APIs.",
            ]
        )
    elif tier in {TIER_BALANCED, TIER_ENHANCED}:
        lines.extend(
            [
                "SERP discovery: DuckDuckGo HTML (primary) and Wikipedia API.",
                "Brave Search API is used only as fallback when configured — "
                "queries then go to Brave under their terms.",
            ]
        )
    if tier == TIER_SEARXNG:
        lines.append(
            "SearXNG: queries go to your configured instance; upstream engines "
            "depend on your server configuration."
        )
    lines.append("Qube does not proxy or log your search queries on Qube servers.")
    return lines
