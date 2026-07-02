"""Validate discipline packs against catalog readiness and adapter registry."""

from __future__ import annotations

import re

from core.knowledge.adapter_readiness import get_adapter_readiness_meta
from core.knowledge.adapters.catalog import get_adapter_entry
from core.knowledge.adapters.registry import SEARCH_FUNCTIONS
from core.knowledge.scientific_discipline_packs import (
    SCIENTIFIC_DISCIPLINE_PACKS,
    ScientificDisciplinePack,
)

_STUB_NOTE_RE = re.compile(r"\b\w[\w_]*\s*\(stub\)", re.IGNORECASE)


def _meta_for_adapter(adapter_id: str):
    entry = get_adapter_entry(adapter_id)
    if entry is not None:
        from core.knowledge.adapter_readiness import readiness_for_catalog_entry

        return readiness_for_catalog_entry(entry)
    return get_adapter_readiness_meta(adapter_id)


def validate_discipline_packs() -> list[str]:
    """Return human-readable consistency errors (empty when healthy)."""
    errors: list[str] = []
    live_ids = frozenset(SEARCH_FUNCTIONS)

    for pack in SCIENTIFIC_DISCIPLINE_PACKS:
        for adapter_id in pack.resolved_adapter_order():
            if adapter_id not in live_ids:
                errors.append(
                    f"{pack.id}: adapter '{adapter_id}' is not registered in SEARCH_FUNCTIONS"
                )
                continue
            meta = _meta_for_adapter(adapter_id)
            if meta.readiness == "stub":
                errors.append(
                    f"{pack.id}: adapter '{adapter_id}' is still catalog/readiness stub"
                )

        if pack.status == "active":
            for adapter_id in pack.primary_adapters:
                meta = _meta_for_adapter(adapter_id)
                if meta.readiness == "stub":
                    errors.append(
                        f"{pack.id}: primary adapter '{adapter_id}' must not be stub on active pack"
                    )

        for match in _STUB_NOTE_RE.finditer(pack.notes or ""):
            token = match.group(0).lower()
            for adapter_id in pack.resolved_adapter_order():
                if adapter_id.replace("_", " ") in token or adapter_id in token:
                    meta = _meta_for_adapter(adapter_id)
                    if meta.readiness in ("production", "beta"):
                        errors.append(
                            f"{pack.id}: notes still mark live adapter '{adapter_id}' as stub"
                        )
                    break

    return errors


def suggest_pack_notes(pack: ScientificDisciplinePack) -> str:
    """Generate a concise notes line from live adapter readiness (for tooling)."""
    primary_labels: list[str] = []
    for adapter_id in pack.primary_adapters:
        meta = _meta_for_adapter(adapter_id)
        if meta.readiness == "stub":
            primary_labels.append(f"{adapter_id} (pending)")
        else:
            primary_labels.append(adapter_id)

    fallback_live = [
        aid
        for aid in pack.fallback_adapters
        if _meta_for_adapter(aid).readiness in ("production", "beta")
    ]
    parts = [f"Primary: {', '.join(primary_labels)}."]
    if fallback_live:
        parts.append(f"Fallbacks: {', '.join(fallback_live)}.")
    return " ".join(parts)


def sync_report() -> str:
    """Text report for ``tools/sync_discipline_packs.py --check``."""
    errors = validate_discipline_packs()
    lines = [f"Discipline packs checked: {len(SCIENTIFIC_DISCIPLINE_PACKS)}"]
    if errors:
        lines.append(f"Errors: {len(errors)}")
        lines.extend(f"  - {err}" for err in errors)
    else:
        lines.append("Errors: 0 (OK)")
    lines.append("")
    lines.append("Suggested notes (informational):")
    for pack in SCIENTIFIC_DISCIPLINE_PACKS:
        lines.append(f"  {pack.id}: {suggest_pack_notes(pack)}")
    return "\n".join(lines)


def assert_discipline_packs_synced() -> None:
    """Raise AssertionError when packs drift from adapter registry/readiness."""
    errors = validate_discipline_packs()
    if errors:
        raise AssertionError("Discipline pack sync failed:\n" + "\n".join(errors))
