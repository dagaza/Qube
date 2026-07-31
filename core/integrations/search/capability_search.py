"""Fuzzy search over cached integration capabilities for the composer palette.

Provider-agnostic (P5/P6): reads descriptor caches and consent state only; never
imports a concrete provider or branches on a specific provider id.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.composer_attachments import ComposerAttachment
from core.integrations.capabilities import persistence as persistence_mod
from core.integrations.capabilities.model import CapabilityDescriptor, CapabilityTier
from core.integrations.capabilities.persistence import ConsentStore, evaluate_access
from core.integrations.consent_controller import (
    ConsentUIState,
    derive_consent_ui_state,
    load_cached_descriptors,
)

from core.composer_discoverability import list_recent_capability_urns

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


def list_cached_provider_ids() -> list[str]:
    """Return provider ids that have a descriptor cache on disk."""
    root = persistence_mod.user_data_root() / "integrations"
    if not root.is_dir():
        return []
    ids: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and (child / "descriptors.json").is_file():
            ids.append(child.name.lower())
    return sorted(set(ids))


def _humanize_action(action: str) -> str:
    cleaned = (action or "").replace("-", " ").replace("_", " ").strip()
    return cleaned.title() if cleaned else action


def _humanize_segment(value: str) -> str:
    cleaned = (value or "").replace("-", " ").replace("_", " ").strip()
    return cleaned.title() if cleaned else value


def format_capability_label(descriptor: CapabilityDescriptor) -> str:
    """Display label: ``GitHub — Search docs`` (group/namespace + action)."""
    group = _humanize_segment(descriptor.group or descriptor.urn.namespace)
    action = _humanize_action(descriptor.action)
    return f"{group} — {action}"


def is_capability_locked(ui_state: ConsentUIState) -> bool:
    """True when the capability is not currently invokable (attach may still attach)."""
    return ui_state is not ConsentUIState.ALLOWED


def format_capability_subtitle(
    descriptor: CapabilityDescriptor,
    *,
    ui_state: ConsentUIState,
) -> str:
    """Subtitle with tier and inline trust hints (P3)."""
    tier = descriptor.tier.value
    parts: list[str] = []
    if is_capability_locked(ui_state):
        parts.append("locked")
    if descriptor.needs_review or ui_state is ConsentUIState.NEEDS_REVIEW:
        parts.append("needs review")
    elif ui_state is ConsentUIState.REREVIEW_REQUIRED:
        parts.append("re-review required")
    if descriptor.tier is not CapabilityTier.READ:
        parts.append("!")
    parts.append(tier)
    return " · ".join(parts)


def capability_palette_tooltip(entry: CapabilityPaletteEntry) -> str:
    """Plain-text tooltip for a palette row."""
    parts = [entry.label]
    if entry.descriptor.description:
        parts.append(entry.descriptor.description.strip())
    hint = format_capability_subtitle(entry.descriptor, ui_state=entry.ui_state)
    parts.append(f"({hint})")
    if entry.locked:
        from core.integrations.capability_availability import resolve_capability_availability

        availability = resolve_capability_availability(entry.descriptor.urn)
        if availability.user_message:
            parts.append(availability.user_message)
        else:
            parts.append("Grant in Settings → Integrations before invoke.")
    urn_body = str(entry.descriptor.urn)
    if urn_body.startswith("cap:"):
        urn_body = urn_body[4:]
    parts.append(f"Inserts @[cap:{urn_body}].")
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class CapabilityPaletteEntry:
    """One integrations capability row for browse/search palettes."""

    provider_id: str
    descriptor: CapabilityDescriptor
    label: str
    subtitle: str
    locked: bool
    tier: CapabilityTier
    needs_review: bool
    ui_state: ConsentUIState
    score: int = 0

    def to_attachment(self) -> ComposerAttachment:
        return ComposerAttachment(
            kind="capability",
            id=str(self.descriptor.urn),
            label=self.label,
        )


def _score_text(q: str, *, text: str, exact: bool = False) -> int:
    t = text.lower()
    if not q:
        return 0
    if exact and q == t:
        return 100
    if t.startswith(q):
        return 80
    if q in t:
        return 50
    return 0


_RECENT_SCORE_BOOST = 200


def _recent_capability_boost() -> dict[str, int]:
    return {urn: _RECENT_SCORE_BOOST for urn in list_recent_capability_urns()}


def _score_capability(
    q: str,
    descriptor: CapabilityDescriptor,
    provider_id: str,
    *,
    recent_boost: dict[str, int] | None = None,
) -> int:
    label = format_capability_label(descriptor)
    fields = (
        provider_id,
        descriptor.group,
        descriptor.urn.namespace,
        descriptor.action,
        descriptor.description,
        label,
    )
    best = 0
    for field in fields:
        best = max(best, _score_text(q, text=field))
        best = max(best, _score_text(q, text=field, exact=True))
    if recent_boost is not None:
        best += recent_boost.get(str(descriptor.urn), 0)
        best += recent_boost.get(str(descriptor.urn.base), 0)
    return best


def _palette_entry(
    provider_id: str,
    descriptor: CapabilityDescriptor,
    *,
    score: int = 0,
) -> CapabilityPaletteEntry:
    store = ConsentStore(provider_id)
    grant = store.get(descriptor.urn)
    decision = evaluate_access(descriptor, grant)
    ui_state = derive_consent_ui_state(
        descriptor,
        decision,
        grant_granted=grant.granted if grant is not None else None,
    )
    locked = is_capability_locked(ui_state)
    return CapabilityPaletteEntry(
        provider_id=provider_id,
        descriptor=descriptor,
        label=format_capability_label(descriptor),
        subtitle=format_capability_subtitle(descriptor, ui_state=ui_state),
        locked=locked,
        tier=descriptor.tier,
        needs_review=descriptor.needs_review,
        ui_state=ui_state,
        score=score,
    )


def _iter_cached_capabilities() -> list[tuple[str, CapabilityDescriptor]]:
    from core.integrations.capability_availability import mcp_namespace_has_configured_source

    rows: list[tuple[str, CapabilityDescriptor]] = []
    for provider_id in list_cached_provider_ids():
        for descriptor in load_cached_descriptors(provider_id):
            if provider_id == "mcp" and not mcp_namespace_has_configured_source(
                descriptor.urn.namespace
            ):
                continue
            rows.append((provider_id, descriptor))
    return rows


def browse_integrations_capabilities(
    query: str = "",
    *,
    limit: int = 50,
) -> list[CapabilityPaletteEntry]:
    """Browse or filter cached capabilities (empty query lists all, capped)."""
    q = (query or "").strip().lower()
    entries: list[CapabilityPaletteEntry] = []
    recent_boost = _recent_capability_boost()
    for provider_id, descriptor in _iter_cached_capabilities():
        if q:
            score = _score_capability(
                q, descriptor, provider_id, recent_boost=recent_boost
            )
        else:
            score = 1 + recent_boost.get(str(descriptor.urn), 0)
            score += recent_boost.get(str(descriptor.urn.base), 0)
        if q and score <= 0:
            continue
        entries.append(_palette_entry(provider_id, descriptor, score=score))
    entries.sort(key=lambda e: (-e.score, e.label.lower(), str(e.descriptor.urn)))
    return entries[:limit]


def search_integrations_capabilities(
    query: str,
    *,
    limit: int = 12,
) -> list[CapabilityPaletteEntry]:
    """Fuzzy search integration capabilities for the composer global search."""
    q = (query or "").strip().lower()
    if not q:
        return []
    entries: list[CapabilityPaletteEntry] = []
    recent_boost = _recent_capability_boost()
    for provider_id, descriptor in _iter_cached_capabilities():
        score = _score_capability(
            q, descriptor, provider_id, recent_boost=recent_boost
        )
        if score <= 0:
            continue
        entries.append(_palette_entry(provider_id, descriptor, score=score))
    entries.sort(key=lambda e: (-e.score, e.label.lower(), str(e.descriptor.urn)))
    return entries[:limit]
