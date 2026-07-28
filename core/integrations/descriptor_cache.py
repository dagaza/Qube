"""Descriptor cache helpers — merge/prune by MCP namespace (Knowledge source anchor)."""

from __future__ import annotations

from pathlib import Path

from core.integrations.capabilities.model import CapabilityDescriptor
from core.integrations.capabilities.persistence import (
    integrations_dir,
    save_descriptor_cache,
)
from core.integrations.consent_controller import load_cached_descriptors

__all__ = [
    "merge_descriptor_cache_for_namespace",
    "remove_descriptor_cache_namespace",
]


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
    kept = [
        descriptor
        for descriptor in load_cached_descriptors(provider_id)
        if descriptor.urn.namespace.strip().lower() != want
    ]
    if len(kept) == len(load_cached_descriptors(provider_id)):
        return False
    if kept:
        save_descriptor_cache(provider_id, kept)
        return True
    path = integrations_dir(provider_id) / "descriptors.json"
    if path.exists():
        path.unlink(missing_ok=True)
    return True
