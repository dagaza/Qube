"""Resolve why an integration capability is or is not usable (Knowledge → Integrations → Chat)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.integrations.capabilities.model import CapabilityDescriptor
from core.integrations.capabilities.persistence import ConsentStore, evaluate_access
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.consent_controller import (
    ConsentUIState,
    derive_consent_ui_state,
)
from core.integrations.capability_invoke import resolve_descriptor_for_urn
from core.integrations.mcp_configured_source import resolve_configured_mcp_binding
from core.knowledge.configured_sources import inspect_configured_mcp_namespace

__all__ = [
    "CapabilityAvailability",
    "CapabilityBlockReason",
    "resolve_capability_availability",
    "user_message_for_availability",
]


class CapabilityBlockReason(str, Enum):
    OK = "ok"
    SOURCE_MISSING = "source_missing"
    SOURCE_INVALID = "source_invalid"
    NOT_DISCOVERED = "not_discovered"
    NOT_GRANTED = "not_granted"
    NEEDS_REVIEW = "needs_review"
    REREVIEW_REQUIRED = "rereview_required"


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    urn: CapabilityURN
    available: bool
    reason: CapabilityBlockReason
    detail: str = ""
    descriptor: CapabilityDescriptor | None = None
    source_id: str = ""

    @property
    def user_message(self) -> str:
        return user_message_for_availability(self)


def user_message_for_availability(availability: CapabilityAvailability) -> str:
    urn_label = str(availability.urn).removeprefix("cap:")
    ns = availability.urn.namespace
    action = availability.urn.action
    match availability.reason:
        case CapabilityBlockReason.OK:
            return ""
        case CapabilityBlockReason.SOURCE_MISSING:
            return (
                f"Cannot use {urn_label}: no MCP source is configured for namespace "
                f"'{ns}'. Add one under Settings → Knowledge → Custom sources."
            )
        case CapabilityBlockReason.SOURCE_INVALID:
            sid = availability.source_id or ns
            detail = availability.detail.strip()
            suffix = f" ({detail})" if detail else ""
            return (
                f"Cannot use {urn_label}: the Knowledge source '{sid}' exists but "
                f"its configuration is invalid{suffix}. Fix it under Custom sources."
            )
        case CapabilityBlockReason.NOT_DISCOVERED:
            return (
                f"Cannot use {urn_label}: the capability is not registered yet. "
                f"Select the '{ns}' source under Custom sources and click Test or Save "
                f"to discover tools from the MCP server."
            )
        case CapabilityBlockReason.NOT_GRANTED:
            return (
                f"Cannot use {urn_label}: permission is not granted. Enable "
                f"'{action}' under Settings → Integrations → Capability permissions."
            )
        case CapabilityBlockReason.NEEDS_REVIEW:
            return (
                f"Cannot use {urn_label}: this capability needs manual review before "
                f"it can be granted (Settings → Integrations)."
            )
        case CapabilityBlockReason.REREVIEW_REQUIRED:
            return (
                f"Cannot use {urn_label}: the MCP server or capability changed since "
                f"you last granted it. Re-review under Settings → Integrations."
            )
    return f"Cannot use {urn_label}."


def resolve_capability_availability(urn: CapabilityURN) -> CapabilityAvailability:
    """Walk Knowledge → Integrations → consent to explain attach/invoke readiness."""
    if urn.provider != "mcp":
        descriptor = resolve_descriptor_for_urn(urn)
        if descriptor is None:
            return CapabilityAvailability(
                urn=urn,
                available=False,
                reason=CapabilityBlockReason.NOT_DISCOVERED,
            )
        return _availability_from_descriptor(urn, descriptor)

    ns_state, source_id, detail = inspect_configured_mcp_namespace(urn.namespace)
    if ns_state == "missing":
        return CapabilityAvailability(
            urn=urn,
            available=False,
            reason=CapabilityBlockReason.SOURCE_MISSING,
            detail=detail,
        )
    if ns_state == "invalid":
        return CapabilityAvailability(
            urn=urn,
            available=False,
            reason=CapabilityBlockReason.SOURCE_INVALID,
            source_id=source_id,
            detail=detail,
        )
    if resolve_configured_mcp_binding(urn.namespace) is None:
        return CapabilityAvailability(
            urn=urn,
            available=False,
            reason=CapabilityBlockReason.SOURCE_MISSING,
            source_id=source_id,
        )

    descriptor = resolve_descriptor_for_urn(urn)
    if descriptor is None:
        return CapabilityAvailability(
            urn=urn,
            available=False,
            reason=CapabilityBlockReason.NOT_DISCOVERED,
            source_id=source_id,
        )
    availability = _availability_from_descriptor(urn, descriptor)
    if availability.source_id:
        return availability
    return CapabilityAvailability(
        urn=availability.urn,
        available=availability.available,
        reason=availability.reason,
        detail=availability.detail,
        descriptor=availability.descriptor,
        source_id=source_id,
    )


def _availability_from_descriptor(
    urn: CapabilityURN,
    descriptor: CapabilityDescriptor,
) -> CapabilityAvailability:
    store = ConsentStore(urn.provider)
    grant = store.get(descriptor.urn)
    decision = evaluate_access(descriptor, grant)
    ui_state = derive_consent_ui_state(
        descriptor,
        decision,
        grant_granted=grant.granted if grant is not None else None,
    )
    if ui_state is ConsentUIState.ALLOWED:
        return CapabilityAvailability(
            urn=urn,
            available=True,
            reason=CapabilityBlockReason.OK,
            descriptor=descriptor,
        )
    if ui_state is ConsentUIState.NEEDS_REVIEW:
        reason = CapabilityBlockReason.NEEDS_REVIEW
    elif ui_state is ConsentUIState.REREVIEW_REQUIRED:
        reason = CapabilityBlockReason.REREVIEW_REQUIRED
    else:
        reason = CapabilityBlockReason.NOT_GRANTED
    return CapabilityAvailability(
        urn=urn,
        available=False,
        reason=reason,
        detail=decision.reason,
        descriptor=descriptor,
    )


def mcp_namespace_has_configured_source(namespace: str) -> bool:
    """True when a valid MCP Knowledge source exists for ``namespace``."""
    state, _, _ = inspect_configured_mcp_namespace(namespace)
    return state == "ok"
