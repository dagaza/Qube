"""Descriptor cache helpers — merge/prune by MCP namespace (Knowledge source anchor)."""

from __future__ import annotations

from pathlib import Path

from core.integrations.capabilities.model import CapabilityDescriptor
from core.integrations.capabilities.persistence import (
    integrations_dir,
    prune_consent_for_namespaces,
    save_descriptor_cache,
)
from core.integrations.consent_controller import load_cached_descriptors
from core.integrations.mcp_configured_source import configured_mcp_namespaces

__all__ = [
    "merge_descriptor_cache_for_namespace",
    "reconcile_mcp_integration_state",
    "remove_descriptor_cache_namespace",
]

_MCP_PROVIDER_ID = "mcp"


def merge_descriptor_cache_for_namespace(
    provider_id: str,
    namespace: str,
    descriptors: list[CapabilityDescriptor],
) -> Path:
    """Replace cached capabilities for one namespace; keep other namespaces intact."""
    want = (namespace or "").strip().lower()
    kept = [
        descriptor
        for descriptor in load_cached_descriptors(provider_id)
        if descriptor.urn.namespace.strip().lower() != want
    ]
    merged = kept + list(descriptors)
    return save_descriptor_cache(provider_id, merged)


def remove_descriptor_cache_namespace(provider_id: str, namespace: str) -> bool:
    """Drop all cached capabilities for ``namespace`` (e.g. when a Knowledge source is deleted)."""
    want = (namespace or "").strip().lower()
    before = load_cached_descriptors(provider_id)
    kept = [
        descriptor
        for descriptor in before
        if descriptor.urn.namespace.strip().lower() != want
    ]
    if len(kept) == len(before):
        return False
    if kept:
        save_descriptor_cache(provider_id, kept)
        return True
    path = integrations_dir(provider_id) / "descriptors.json"
    if path.exists():
        path.unlink(missing_ok=True)
    return True


def reconcile_mcp_integration_state() -> dict[str, int]:
    """Drop MCP descriptor cache + consent residue for unconfigured Knowledge sources."""
    allowed = configured_mcp_namespaces()
    descriptors = load_cached_descriptors(_MCP_PROVIDER_ID)
    cached_namespaces = {
        descriptor.urn.namespace.strip().lower() for descriptor in descriptors
    }
    orphan_namespaces = cached_namespaces - set(allowed)

    descriptors_removed = 0
    if not allowed and descriptors:
        descriptors_removed = len(descriptors)
        path = integrations_dir(_MCP_PROVIDER_ID) / "descriptors.json"
        if path.exists():
            path.unlink(missing_ok=True)
    else:
        for namespace in sorted(orphan_namespaces):
            before = len(load_cached_descriptors(_MCP_PROVIDER_ID))
            if remove_descriptor_cache_namespace(_MCP_PROVIDER_ID, namespace):
                after = len(load_cached_descriptors(_MCP_PROVIDER_ID))
                descriptors_removed += max(0, before - after)

    grants_removed = prune_consent_for_namespaces(_MCP_PROVIDER_ID, allowed)
    return {
        "descriptors_removed": descriptors_removed,
        "grants_removed": grants_removed,
        "namespaces_pruned": len(orphan_namespaces),
    }
