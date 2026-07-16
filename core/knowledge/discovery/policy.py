"""Web discovery policy — primary provider and bot_challenge fallback chain."""

from __future__ import annotations

from core.knowledge.adapters.brave_search import brave_search_configured
from core.knowledge.discovery.searxng import SEARXNG_DISCOVERY_PROVIDER_ID

PRIMARY_DISCOVERY_PROVIDER_ID = "duckduckgo"
WIKIPEDIA_DISCOVERY_PROVIDER_ID = "wikipedia"
BRAVE_DISCOVERY_PROVIDER_ID = "brave_search"

DISCOVERY_PROVIDER_LABELS: dict[str, str] = {
    PRIMARY_DISCOVERY_PROVIDER_ID: "DuckDuckGo",
    BRAVE_DISCOVERY_PROVIDER_ID: "Brave Search API",
    WIKIPEDIA_DISCOVERY_PROVIDER_ID: "Wikipedia API",
    SEARXNG_DISCOVERY_PROVIDER_ID: "SearXNG (self-hosted)",
}


def discovery_provider_label(provider_id: str | None) -> str:
    pid = (provider_id or "").strip().lower()
    return DISCOVERY_PROVIDER_LABELS.get(pid, pid or "—")


def bot_challenge_fallback_chain(
    *,
    site_bias: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Ordered fallback providers after primary discovery failure."""
    from core.knowledge.discovery.privacy_policy import resolve_discovery_route

    return resolve_discovery_route(site_bias=site_bias).fallback_ids


def discovery_policy_summary_lines() -> list[str]:
    """Human-readable policy for Settings and Inspector."""
    from core.knowledge.discovery.backoff import (
        format_backoff_summary,
        get_provider_backoff,
    )
    from core.knowledge.discovery.health import conservative_mode_summary
    from core.knowledge.discovery.pacing import (
        discovery_pace_min_seconds,
        discovery_pacing_enabled,
        effective_discovery_pace_min_seconds,
    )
    from core.knowledge.discovery.privacy_policy import (
        get_active_privacy_tier,
        privacy_tier_label,
        resolve_discovery_route,
    )
    from core.knowledge.discovery.session_budget import (
        format_burst_budget_summary,
        format_discovery_budget_usage_lines,
        format_session_budget_summary,
    )

    tier = get_active_privacy_tier()
    route = resolve_discovery_route()
    lines = [
        f"Privacy tier: {privacy_tier_label(tier)}",
        f"Primary: {discovery_provider_label(route.primary_id)}",
    ]
    backoff_line = format_backoff_summary(
        get_provider_backoff(PRIMARY_DISCOVERY_PROVIDER_ID)
    )
    if backoff_line:
        lines.append(backoff_line)
    budget_lines = format_discovery_budget_usage_lines()
    if budget_lines:
        lines.extend(budget_lines)
    else:
        burst_line = format_burst_budget_summary()
        session_line = format_session_budget_summary()
        if burst_line:
            lines.append(burst_line)
        if session_line:
            lines.append(session_line)
    conservative_line = conservative_mode_summary()
    if conservative_line:
        lines.append(conservative_line)
    if discovery_pacing_enabled():
        effective = effective_discovery_pace_min_seconds()
        base = discovery_pace_min_seconds()
        if effective > base + 0.01:
            pacing_value = (
                f"~{effective:.0f}s minimum gap between live DDG queries "
                f"(conservative mode; base ~{base:.0f}s)."
            )
        else:
            pacing_value = (
                f"~{effective:.0f}s minimum gap between live DDG queries "
                "(reduces bot challenges)."
            )
    else:
        pacing_value = "Off (no extra delay between live DDG requests)"
    lines.append(f"Pacing: {pacing_value}")
    fallbacks = route.fallback_ids
    if fallbacks:
        labels = " → ".join(discovery_provider_label(pid) for pid in fallbacks)
        lines.append(f"On primary failure: {labels}")
    else:
        lines.append("On primary failure: no fallbacks configured")
    if (
        tier != "private"
        and not brave_search_configured()
    ):
        lines.append(
            "Brave Search API is optional — add a free API key to improve "
            "fallback coverage (including site-biased queries)."
        )
    return lines
