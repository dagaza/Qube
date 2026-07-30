"""Qt-free grant review model for first-connect and drift reconnect flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.integrations.capabilities.model import CapabilityDescriptor, CapabilityTier
from core.integrations.capabilities.persistence import ConsentStore, evaluate_access
from core.integrations.capability_drift import CapabilityDriftDiff
from core.integrations.consent_controller import (
    ConsentUIState,
    IntegrationsConsentController,
    derive_consent_ui_state,
)

__all__ = [
    "GrantReviewChange",
    "GrantReviewRow",
    "SuggestedCapabilityPreset",
    "apply_grant_review_rows",
    "build_grant_review_rows",
    "capability_in_preset",
    "preset_capability_urn_set",
    "save_suggested_capability_preset",
    "suggest_capability_presets",
]


class GrantReviewChange(str, Enum):
    UNCHANGED = "unchanged"
    NEW = "new"
    CHANGED = "changed"


@dataclass(frozen=True, slots=True)
class GrantReviewRow:
    descriptor: CapabilityDescriptor
    checked: bool
    enabled: bool
    change: GrantReviewChange
    ui_state: ConsentUIState


@dataclass(frozen=True, slots=True)
class SuggestedCapabilityPreset:
    preset_id: str
    label: str
    description: str
    capability_urns: tuple[str, ...]


def build_grant_review_rows(
    provider_id: str,
    descriptors: list[CapabilityDescriptor],
    *,
    namespace: str,
    first_connect: bool,
    drift: CapabilityDriftDiff | None = None,
) -> list[GrantReviewRow]:
    """Build dialog rows with tier-aware defaults (read on, write/destructive off)."""
    ns = (namespace or "").strip().lower()
    scoped = [
        descriptor
        for descriptor in descriptors
        if descriptor.urn.namespace.strip().lower() == ns
    ]
    scoped.sort(key=lambda descriptor: (descriptor.group.lower(), descriptor.action.lower()))

    store = ConsentStore(provider_id)
    grants = store.load()
    added_bases = {str(descriptor.urn.base) for descriptor in (drift.added if drift else ())}
    changed_bases = {str(descriptor.urn.base) for descriptor in (drift.changed if drift else ())}

    rows: list[GrantReviewRow] = []
    for descriptor in scoped:
        grant = grants.get(str(descriptor.urn.base))
        decision = evaluate_access(descriptor, grant)
        ui_state = derive_consent_ui_state(
            descriptor,
            decision,
            grant_granted=grant.granted if grant is not None else None,
        )
        enabled = not descriptor.needs_review and ui_state is not ConsentUIState.NEEDS_REVIEW

        if str(descriptor.urn.base) in added_bases:
            change = GrantReviewChange.NEW
        elif str(descriptor.urn.base) in changed_bases:
            change = GrantReviewChange.CHANGED
        else:
            change = GrantReviewChange.UNCHANGED

        if not enabled:
            checked = False
        elif first_connect or change is GrantReviewChange.NEW:
            checked = descriptor.tier is CapabilityTier.READ
        elif change is GrantReviewChange.CHANGED:
            checked = False
        elif grant is not None and grant.granted and ui_state is ConsentUIState.ALLOWED:
            checked = True
        else:
            checked = False

        rows.append(
            GrantReviewRow(
                descriptor=descriptor,
                checked=checked,
                enabled=enabled,
                change=change,
                ui_state=ui_state,
            )
        )
    return rows


def apply_grant_review_rows(provider_id: str, rows: list[GrantReviewRow]) -> None:
    controller = IntegrationsConsentController(provider_id)
    for row in rows:
        if not row.enabled:
            continue
        if row.checked:
            controller.grant_capability(row.descriptor)
        else:
            controller.deny_capability(row.descriptor)


def preset_capability_urn_set(preset: SuggestedCapabilityPreset) -> frozenset[str]:
    """Normalized URN keys for matching preset bundles to descriptors."""
    from core.integrations.capabilities.urn import CapabilityURN

    keys: set[str] = set()
    for raw in preset.capability_urns:
        text = (raw or "").strip()
        if not text:
            continue
        parsed = CapabilityURN.try_parse(text)
        if parsed is not None:
            keys.add(str(parsed))
            keys.add(str(parsed.base))
        else:
            keys.add(text)
    return frozenset(keys)


def capability_in_preset(
    descriptor: CapabilityDescriptor,
    preset: SuggestedCapabilityPreset,
) -> bool:
    keys = preset_capability_urn_set(preset)
    return str(descriptor.urn) in keys or str(descriptor.urn.base) in keys


def suggest_capability_presets(
    *,
    namespace: str,
    server_label: str,
    descriptors: list[CapabilityDescriptor],
) -> list[SuggestedCapabilityPreset]:
    """Heuristic preset bundles aligned with plan §4.5."""
    from core.knowledge.presets import MAX_PRESET_CAPABILITIES

    ns = (namespace or "").strip().lower()
    scoped = [
        descriptor
        for descriptor in descriptors
        if descriptor.urn.namespace.strip().lower() == ns and not descriptor.needs_review
    ]
    read_caps = [descriptor for descriptor in scoped if descriptor.tier is CapabilityTier.READ]
    write_caps = [descriptor for descriptor in scoped if descriptor.tier is CapabilityTier.WRITE]
    if not scoped:
        return []

    presets: list[SuggestedCapabilityPreset] = []
    if read_caps:
        presets.append(
            SuggestedCapabilityPreset(
                preset_id=f"{ns}-minimal",
                label="Minimal",
                description=f"Only {read_caps[0].action.replace('-', ' ')}",
                capability_urns=(str(read_caps[0].urn),),
            )
        )
    if len(read_caps) >= 2:
        presets.append(
            SuggestedCapabilityPreset(
                preset_id=f"{ns}-read-search",
                label="Read & search",
                description="Enable read and search capabilities",
                capability_urns=tuple(
                    str(descriptor.urn)
                    for descriptor in read_caps[:MAX_PRESET_CAPABILITIES]
                ),
            )
        )
    if read_caps and write_caps:
        combined = read_caps + write_caps
        presets.append(
            SuggestedCapabilityPreset(
                preset_id=f"{ns}-developer",
                label="Developer",
                description="Read plus limited write capabilities",
                capability_urns=tuple(
                    str(descriptor.urn)
                    for descriptor in combined[:MAX_PRESET_CAPABILITIES]
                ),
            )
        )

    label_prefix = (server_label or ns).strip()
    unique: list[SuggestedCapabilityPreset] = []
    seen: set[str] = set()
    for preset in presets:
        if preset.preset_id in seen:
            continue
        seen.add(preset.preset_id)
        unique.append(preset)
    if label_prefix:
        unique = [
            SuggestedCapabilityPreset(
                preset_id=preset.preset_id,
                label=preset.label,
                description=f"{label_prefix} — {preset.description}",
                capability_urns=preset.capability_urns,
            )
            for preset in unique
        ]
    return unique[:3]


def save_suggested_capability_preset(
    preset: SuggestedCapabilityPreset,
    *,
    server_label: str,
) -> str:
    from core.knowledge.presets import KnowledgePreset, save_preset

    label = f"{server_label} — {preset.label}".strip(" —")
    knowledge_preset = KnowledgePreset(
        id=preset.preset_id,
        label=label,
        capabilities=list(preset.capability_urns),
        composer_visible=True,
    )
    save_preset(knowledge_preset)
    return knowledge_preset.id
