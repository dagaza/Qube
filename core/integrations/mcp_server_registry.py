"""Read-only MCP server summaries for Settings → Integrations."""

from __future__ import annotations

from dataclasses import dataclass

from core.integrations.capabilities.persistence import load_descriptor_cache
from core.integrations.consent_controller import ConsentUIState, IntegrationsConsentController
from core.knowledge.configured_sources import ConfiguredSource, list_configured_sources

__all__ = ["McpServerSummary", "list_mcp_server_summaries"]

_PROVIDER_ID = "mcp"


@dataclass(frozen=True, slots=True)
class McpServerSummary:
    source_id: str
    label: str
    namespace: str
    capability_count: int
    granted_count: int
    rereview_count: int
    health_label: str
    last_discovered_at: str | None


def _namespace_for_source(source: ConfiguredSource) -> str:
    cfg = dict(source.config or {})
    return str(cfg.get("namespace") or cfg.get("adapter_id") or source.id).strip().lower()


def list_mcp_server_summaries() -> list[McpServerSummary]:
    cache = load_descriptor_cache(_PROVIDER_ID)
    discovered_at = str(cache.get("discovered_at") or "") or None
    controller = IntegrationsConsentController(_PROVIDER_ID)
    rows_by_ns: dict[str, list] = {}
    for row in controller.list_capability_rows():
        rows_by_ns.setdefault(row.descriptor.urn.namespace.strip().lower(), []).append(row)

    summaries: list[McpServerSummary] = []
    for source in list_configured_sources():
        if source.connector_type != "mcp":
            continue
        namespace = _namespace_for_source(source)
        ns_rows = rows_by_ns.get(namespace, [])
        granted = sum(1 for row in ns_rows if row.ui_state is ConsentUIState.ALLOWED)
        rereview = sum(
            1
            for row in ns_rows
            if row.ui_state is ConsentUIState.REREVIEW_REQUIRED
        )
        cap_count = len(ns_rows)
        if cap_count == 0:
            health = "Not discovered"
        elif rereview:
            health = "Needs re-review"
        elif granted == 0:
            health = "Permissions pending"
        else:
            health = "Ready"
        summaries.append(
            McpServerSummary(
                source_id=source.id,
                label=source.label or source.id,
                namespace=namespace,
                capability_count=cap_count,
                granted_count=granted,
                rereview_count=rereview,
                health_label=health,
                last_discovered_at=discovered_at,
            )
        )
    return sorted(summaries, key=lambda item: item.label.lower())
