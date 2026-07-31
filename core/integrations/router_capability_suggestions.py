"""Opt-in router integration capability suggestions (Phase 4 / #62).

Read-only hints for INSPECT / routing debug — never auto-invokes capabilities (P1/P2).
Default off via ``get_router_integration_suggestions_enabled()``.
"""

from __future__ import annotations

from core.integrations.capabilities.model import CapabilityTier
from core.integrations.consent_controller import ConsentUIState
from core.integrations.search.capability_search import (
    format_capability_label,
    search_integrations_capabilities,
)

__all__ = [
    "format_router_capability_suggestions_line",
    "suggest_integration_capabilities",
]


def suggest_integration_capabilities(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, str]]:
    """Return read-tier, granted capability suggestions for a query (never invoked)."""
    q = (query or "").strip()
    if not q:
        return []

    entries = search_integrations_capabilities(q, limit=max(1, max_results) * 4)
    out: list[dict[str, str]] = []
    for entry in entries:
        if entry.ui_state is not ConsentUIState.ALLOWED:
            continue
        if entry.descriptor.tier is not CapabilityTier.READ:
            continue
        out.append(
            {
                "urn": str(entry.descriptor.urn),
                "label": format_capability_label(entry.descriptor),
                "reason": "query match (suggestion only — attach explicitly to invoke)",
            }
        )
        if len(out) >= max_results:
            break
    return out


def format_router_capability_suggestions_line(
    suggestions: list[dict[str, str]] | None,
) -> str:
    if not suggestions:
        return ""
    labels = [str(item.get("label") or item.get("urn") or "?") for item in suggestions]
    return "Integration suggestions (opt-in): " + ", ".join(labels)
