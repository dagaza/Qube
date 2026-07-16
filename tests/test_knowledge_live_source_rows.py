"""Live sources row registry must track every catalog row, not one per adapter id."""

from __future__ import annotations

from collections import Counter

from core.knowledge.adapters.catalog import (
    CONFIGURABLE_KNOWLEDGE_SERVICES,
    catalog_entries_for_ui_group,
    ui_groups_for_service,
)
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE


def _visible_catalog_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for service_id, _label in CONFIGURABLE_KNOWLEDGE_SERVICES:
        for group in ui_groups_for_service(service_id):
            for entry in catalog_entries_for_ui_group(service_id, group):
                rows.append((service_id, group, entry.id))
    return rows


def test_catalog_repeats_common_adapters_across_groups() -> None:
    """Regression guard: duplicate adapter ids are intentional in the catalog."""
    scientific = [
        entry_id
        for service_id, _group, entry_id in _visible_catalog_rows()
        if service_id == SERVICE_SCIENTIFIC_EVIDENCE
    ]
    counts = Counter(scientific)
    assert counts["pubmed"] > 1
    assert counts["openalex"] > 1


def test_visible_row_count_exceeds_unique_adapter_keys() -> None:
    """Theme/sync refresh must touch every on-screen row, not only the last per id."""
    rows = _visible_catalog_rows()
    unique_adapter_keys = {(service_id, entry_id) for service_id, _group, entry_id in rows}
    assert len(rows) > len(unique_adapter_keys)
