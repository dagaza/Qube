"""Capability search for composer palette and integrations discovery (P1/P5)."""

from core.integrations.search.capability_search import (
    CapabilityPaletteEntry,
    browse_integrations_capabilities,
    capability_palette_tooltip,
    format_capability_label,
    format_capability_subtitle,
    is_capability_locked,
    list_cached_provider_ids,
    search_integrations_capabilities,
)

__all__ = [
    "CapabilityPaletteEntry",
    "browse_integrations_capabilities",
    "capability_palette_tooltip",
    "format_capability_label",
    "format_capability_subtitle",
    "is_capability_locked",
    "list_cached_provider_ids",
    "search_integrations_capabilities",
]
