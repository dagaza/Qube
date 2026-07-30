"""Namespace-level capability drift detection for MCP reconnect flows."""

from __future__ import annotations

from dataclasses import dataclass

from core.integrations.capabilities.model import CapabilityDescriptor
from core.integrations.capabilities.persistence import capability_fingerprint

__all__ = [
    "CapabilityDriftDiff",
    "descriptors_for_namespace",
    "diff_namespace_capabilities",
    "format_drift_summary",
    "has_material_drift",
]


@dataclass(frozen=True, slots=True)
class CapabilityDriftDiff:
    """Capability changes for one MCP namespace between two discovery snapshots."""

    namespace: str
    added: tuple[CapabilityDescriptor, ...]
    removed: tuple[str, ...]
    changed: tuple[CapabilityDescriptor, ...]


def descriptors_for_namespace(
    descriptors: list[CapabilityDescriptor],
    namespace: str,
) -> list[CapabilityDescriptor]:
    want = (namespace or "").strip().lower()
    return [
        descriptor
        for descriptor in descriptors
        if descriptor.urn.namespace.strip().lower() == want
    ]


def diff_namespace_capabilities(
    before: list[CapabilityDescriptor],
    after: list[CapabilityDescriptor],
    *,
    namespace: str,
) -> CapabilityDriftDiff:
    """Compare cached vs freshly discovered capabilities for one namespace."""
    ns = (namespace or "").strip().lower()
    prev = {
        str(descriptor.urn.base): descriptor
        for descriptor in descriptors_for_namespace(before, ns)
    }
    curr = {
        str(descriptor.urn.base): descriptor
        for descriptor in descriptors_for_namespace(after, ns)
    }

    added: list[CapabilityDescriptor] = []
    changed: list[CapabilityDescriptor] = []
    removed: list[str] = []

    for urn_base, descriptor in curr.items():
        prior = prev.get(urn_base)
        if prior is None:
            added.append(descriptor)
            continue
        if capability_fingerprint(prior) != capability_fingerprint(descriptor):
            changed.append(descriptor)

    for urn_base, descriptor in prev.items():
        if urn_base not in curr:
            removed.append(descriptor.action or urn_base)

    return CapabilityDriftDiff(
        namespace=ns,
        added=tuple(added),
        removed=tuple(removed),
        changed=tuple(changed),
    )


def has_material_drift(diff: CapabilityDriftDiff) -> bool:
    return bool(diff.added or diff.removed or diff.changed)


def _capability_word(count: int) -> str:
    return "capability" if count == 1 else "capabilities"


def format_drift_summary(diff: CapabilityDriftDiff) -> str:
    parts: list[str] = []
    if diff.added:
        parts.append(f"{len(diff.added)} new")
    if diff.removed:
        parts.append(f"{len(diff.removed)} removed")
    if diff.changed:
        parts.append(f"{len(diff.changed)} changed")
    if not parts:
        return "No capability changes detected."
    total = len(diff.added) + len(diff.removed) + len(diff.changed)
    return f"{', '.join(parts)} {_capability_word(total)}"
