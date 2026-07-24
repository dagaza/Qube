"""Composer send gate for write/destructive capability step approval (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.composer_attachments import ComposerAttachment, resolve_attachment_routing
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capability_invoke import (
    parse_composer_capability_urn,
    resolve_descriptor_for_urn,
)
from core.integrations.preset_capability_alias import resolve_preset_capability_urns
from core.integrations.step_approval import requires_step_approval, step_approval_store

__all__ = [
    "CapabilityStepApprovalItem",
    "capabilities_requiring_step_approval",
    "format_step_approval_message",
    "pending_step_approvals",
]


@dataclass(frozen=True, slots=True)
class CapabilityStepApprovalItem:
    urn: str
    label: str
    tier: str
    group: str


def _candidate_urns(attachments: Sequence[ComposerAttachment]) -> list[str]:
    urns: list[str] = []
    seen: set[str] = set()
    routing = resolve_attachment_routing(list(attachments))
    if routing:
        preset_id = str(routing.get("capability_preset_id") or "").strip()
        if preset_id:
            for urn in resolve_preset_capability_urns(preset_id):
                if urn not in seen:
                    seen.add(urn)
                    urns.append(urn)
        for raw in routing.get("capability_urns") or ():
            s = str(raw)
            if s and s not in seen:
                seen.add(s)
                urns.append(s)
        cap = str(routing.get("capability_urn") or "").strip()
        if cap and cap not in seen:
            seen.add(cap)
            urns.append(cap)
    for att in attachments:
        if att.kind != "capability":
            continue
        parsed = parse_composer_capability_urn(att.id)
        if parsed is None:
            continue
        canonical = str(parsed)
        if canonical not in seen:
            seen.add(canonical)
            urns.append(canonical)
    return urns


def capabilities_requiring_step_approval(
    attachments: Sequence[ComposerAttachment],
) -> list[CapabilityStepApprovalItem]:
    """Return write/destructive attached caps (for confirm dialog copy)."""
    items: list[CapabilityStepApprovalItem] = []
    for urn in _candidate_urns(attachments):
        parsed = parse_composer_capability_urn(urn)
        if parsed is None:
            continue
        descriptor = resolve_descriptor_for_urn(parsed)
        if descriptor is None or not requires_step_approval(descriptor):
            continue
        label = f"{descriptor.group} — {descriptor.action}"
        items.append(
            CapabilityStepApprovalItem(
                urn=str(parsed),
                label=label,
                tier=descriptor.tier.value,
                group=descriptor.group,
            )
        )
    return items


def pending_step_approvals(
    session_id: str,
    turn_id: str,
    attachments: Sequence[ComposerAttachment],
) -> list[CapabilityStepApprovalItem]:
    """Caps that still need per-step approval for this turn."""
    pending: list[CapabilityStepApprovalItem] = []
    for item in capabilities_requiring_step_approval(attachments):
        if step_approval_store.has_approval(session_id, turn_id, item.urn):
            continue
        pending.append(item)
    return pending


def format_step_approval_message(items: Sequence[CapabilityStepApprovalItem]) -> str:
    if not items:
        return ""
    lines = [
        "The following attached capabilities can modify external data. "
        "Approve running them for this message only:"
    ]
    for item in items:
        tier_note = item.tier
        if item.tier == CapabilityTier.DESTRUCTIVE.value:
            tier_note = f"{item.tier} (!)"
        lines.append(f"• {item.label} [{tier_note}]")
    lines.append("")
    lines.append("Read-only capabilities do not require this confirmation.")
    return "\n".join(lines)
