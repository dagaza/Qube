"""User-configurable knowledge source preferences per domain service."""

from __future__ import annotations

from core.knowledge.adapters.catalog import (
    default_enabled_adapter_ids,
    get_adapter_entry,
    implemented_adapter_ids,
)


def normalize_preferences(raw: dict | None) -> dict[str, list[str]]:
    """Normalize stored preferences to service_id → ordered adapter id list."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for service_id, adapters in raw.items():
        sid = str(service_id or "").strip().lower()
        if not sid:
            continue
        if not isinstance(adapters, (list, tuple)):
            continue
        ids: list[str] = []
        seen: set[str] = set()
        for adapter_id in adapters:
            aid = str(adapter_id or "").strip().lower()
            if not aid or aid in seen:
                continue
            entry = get_adapter_entry(aid)
            if entry is None or entry.knowledge_service != sid:
                continue
            seen.add(aid)
            ids.append(aid)
        if ids:
            out[sid] = ids
    return out


def get_effective_enabled_adapters(
    service_id: str,
    *,
    stored_preferences: dict[str, list[str]] | None = None,
) -> tuple[str, ...]:
    """Return user-enabled adapter ids for a service (implemented only)."""
    sid = (service_id or "").strip().lower()
    allowed = implemented_adapter_ids(sid)
    if not allowed:
        return ()

    prefs = normalize_preferences(stored_preferences)
    selected = prefs.get(sid)
    if selected is None:
        selected = list(default_enabled_adapter_ids(sid))

    enabled = tuple(aid for aid in selected if aid in allowed)
    if enabled:
        return enabled
    return default_enabled_adapter_ids(sid)


def is_adapter_enabled(
    service_id: str,
    adapter_id: str,
    *,
    stored_preferences: dict[str, list[str]] | None = None,
) -> bool:
    aid = (adapter_id or "").strip().lower()
    return aid in get_effective_enabled_adapters(
        service_id,
        stored_preferences=stored_preferences,
    )


def set_adapter_enabled(
    preferences: dict[str, list[str]],
    *,
    service_id: str,
    adapter_id: str,
    enabled: bool,
) -> dict[str, list[str]]:
    """Return updated preferences with one adapter toggled."""
    sid = (service_id or "").strip().lower()
    aid = (adapter_id or "").strip().lower()
    entry = get_adapter_entry(aid)
    if entry is None or entry.knowledge_service != sid:
        return normalize_preferences(preferences)

    merged = normalize_preferences(preferences)
    current = list(merged.get(sid, default_enabled_adapter_ids(sid)))
    if enabled:
        if aid not in current:
            current.append(aid)
    else:
        current = [x for x in current if x != aid]
    if current:
        merged[sid] = current
    elif sid in merged:
        del merged[sid]
    return merged


def resolve_service_adapters(
    service_id: str,
    *,
    query: str = "",
    composer_adapter_filter: tuple[str, ...] | None = None,
    stored_preferences: dict[str, list[str]] | None = None,
    medical_query: bool | None = None,
) -> tuple[str, ...]:
    """
    Resolve adapter ids for a retrieval turn.

    Priority: composer single-adapter override → user preferences → service defaults.
    Scientific service applies medical PubMed gating when no composer override.
    """
    sid = (service_id or "").strip().lower()

    if composer_adapter_filter:
        allowed = implemented_adapter_ids(sid)
        filtered = tuple(aid for aid in composer_adapter_filter if aid in allowed)
        if filtered:
            return filtered

    enabled = get_effective_enabled_adapters(
        sid,
        stored_preferences=stored_preferences,
    )

    if sid == "scientific_evidence":
        from core.knowledge.scientific_adapters import apply_scientific_adapter_policy

        return apply_scientific_adapter_policy(
            enabled,
            query=query,
            medical_query=medical_query,
        )

    return enabled
