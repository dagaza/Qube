"""Read-only web discovery snapshot for Telemetry (R10 / Theme B free slice)."""

from __future__ import annotations

from core.knowledge.discovery.backoff import (
    format_backoff_summary,
    get_provider_backoff,
)
from core.knowledge.discovery.health import (
    challenge_count_24h,
    is_conservative_mode_active,
)
from core.knowledge.discovery.pacing import (
    discovery_pace_min_seconds,
    discovery_pacing_enabled,
    effective_discovery_pace_min_seconds,
)
from core.knowledge.discovery.policy import (
    PRIMARY_DISCOVERY_PROVIDER_ID,
    discovery_provider_label,
    discovery_policy_summary_lines,
)
from core.knowledge.discovery.privacy_policy import (
    get_active_privacy_tier,
    privacy_tier_label,
    resolve_discovery_route,
)
from core.knowledge.discovery.session_budget import (
    get_ddg_burst_budget_status,
    get_ddg_session_budget_status,
)

__all__ = [
    "discovery_policy_summary_lines",
    "discovery_telemetry_snapshot",
    "format_discovery_health_status",
]


def discovery_telemetry_snapshot() -> dict[str, object]:
    """Structured discovery state for Telemetry cards and tests."""
    tier = get_active_privacy_tier()
    route = resolve_discovery_route()
    burst = get_ddg_burst_budget_status()
    session = get_ddg_session_budget_status()
    backoff = get_provider_backoff(PRIMARY_DISCOVERY_PROVIDER_ID)
    pacing_enabled = discovery_pacing_enabled()
    effective_pace = effective_discovery_pace_min_seconds()
    base_pace = discovery_pace_min_seconds()
    conservative = is_conservative_mode_active()

    return {
        "privacy_tier": tier,
        "privacy_tier_label": privacy_tier_label(tier),
        "primary_provider_id": route.primary_id,
        "primary_provider_label": discovery_provider_label(route.primary_id),
        "fallback_provider_ids": list(route.fallback_ids),
        "burst_used": burst.used,
        "burst_limit": burst.limit,
        "burst_remaining": burst.remaining,
        "burst_exhausted": burst.exhausted if burst.limit > 0 else False,
        "session_used": session.used,
        "session_limit": session.limit,
        "session_remaining": session.remaining,
        "session_exhausted": session.exhausted if session.limit > 0 else False,
        "conservative_mode": conservative,
        "challenge_count_24h": challenge_count_24h(),
        "pacing_enabled": pacing_enabled,
        "pacing_base_seconds": base_pace,
        "pacing_effective_seconds": effective_pace,
        "backoff_active": backoff is not None,
        "backoff_summary": format_backoff_summary(backoff),
        "policy_summary_lines": discovery_policy_summary_lines(),
    }


def format_discovery_health_status(snapshot: dict[str, object] | None = None) -> str:
    """One-line health summary for the Telemetry card."""
    snap = snapshot if snapshot is not None else discovery_telemetry_snapshot()
    if snap.get("backoff_active"):
        return "⚠️ DDG backoff active"
    if snap.get("session_exhausted"):
        return "⚠️ Session budget exhausted"
    if snap.get("burst_exhausted"):
        return "⚠️ Burst budget exhausted"
    if snap.get("conservative_mode"):
        return "⚠️ Conservative pacing"
    return "🟢 Discovery stable"
