"""Qt-free consent controller for the Integrations permission UI (P3/P7).

Groups capabilities from the descriptor cache, derives per-capability state from
:func:`evaluate_access` (not grant presence alone), and writes explicit decisions
via :class:`ConsentStore`. ``needs_review`` capabilities are never grantable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    CapabilityGroup,
    CapabilityTier,
)
from core.integrations.capabilities.persistence import (
    AccessDecision,
    ConsentStore,
    evaluate_access,
    load_descriptor_cache,
)
from core.integrations.capabilities.urn import CapabilityURN, InvalidCapabilityURN

logger = logging.getLogger("Qube.Integrations.Consent")

__all__ = [
    "ConsentUIState",
    "CapabilityConsentRow",
    "IntegrationsConsentController",
    "load_cached_descriptors",
    "derive_consent_ui_state",
]


class ConsentUIState(str, Enum):
    """UI-facing consent state derived from ``evaluate_access`` outcomes."""

    ALLOWED = "allowed"
    DENIED = "denied"
    NEEDS_REVIEW = "needs_review"
    REREVIEW_REQUIRED = "rereview_required"


@dataclass(frozen=True, slots=True)
class CapabilityConsentRow:
    """One capability row for the Integrations permission UI."""

    group: str
    descriptor: CapabilityDescriptor
    tier: CapabilityTier
    needs_review: bool
    ui_state: ConsentUIState
    decision: AccessDecision


def load_cached_descriptors(provider_id: str) -> list[CapabilityDescriptor]:
    """Rebuild :class:`CapabilityDescriptor` objects from the on-disk cache."""
    payload = load_descriptor_cache(provider_id)
    records = payload.get("capabilities") or []
    if not isinstance(records, list):
        return []
    descriptors: list[CapabilityDescriptor] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        descriptor = _descriptor_from_cache_record(record)
        if descriptor is not None:
            descriptors.append(descriptor)
    return descriptors


def _descriptor_from_cache_record(record: dict[str, Any]) -> CapabilityDescriptor | None:
    try:
        urn = CapabilityURN.parse(str(record["urn"]))
        return CapabilityDescriptor(
            urn=urn,
            group=str(record.get("group") or ""),
            action=str(record.get("action") or ""),
            tier=CapabilityTier(str(record["tier"])),
            description=str(record.get("description") or ""),
            input_schema=dict(record.get("input_schema") or {}),
            raw_ref=record.get("raw_ref"),
            needs_review=bool(record.get("needs_review")),
        )
    except (KeyError, ValueError, InvalidCapabilityURN) as exc:
        logger.warning("[integrations] skipping malformed cached descriptor %r: %s", record, exc)
        return None


def derive_consent_ui_state(
    descriptor: CapabilityDescriptor,
    decision: AccessDecision,
    *,
    grant_granted: bool | None,
) -> ConsentUIState:
    """Map ``evaluate_access`` to a UI state (P3/P7).

    Re-review is surfaced when a stored *allow* no longer matches the current
    descriptor (drift / tier escalation), not merely because a grant record exists.
    """
    if descriptor.needs_review or decision.needs_review:
        return ConsentUIState.NEEDS_REVIEW
    if decision.allowed:
        return ConsentUIState.ALLOWED
    if grant_granted is True:
        return ConsentUIState.REREVIEW_REQUIRED
    return ConsentUIState.DENIED


class IntegrationsConsentController:
    """Permission/consent orchestration for one provider (testable, Qt-free)."""

    def __init__(
        self,
        provider_id: str,
        *,
        consent_store: ConsentStore | None = None,
        descriptors: list[CapabilityDescriptor] | None = None,
    ) -> None:
        self.provider_id = (provider_id or "").strip().lower()
        self._store = consent_store or ConsentStore(self.provider_id)
        self._descriptors = (
            list(descriptors)
            if descriptors is not None
            else load_cached_descriptors(self.provider_id)
        )

    @property
    def consent_store(self) -> ConsentStore:
        return self._store

    def reload_descriptors(self, descriptors: list[CapabilityDescriptor] | None = None) -> None:
        """Refresh the in-memory descriptor list from cache or an explicit list."""
        self._descriptors = (
            list(descriptors)
            if descriptors is not None
            else load_cached_descriptors(self.provider_id)
        )

    def list_groups(self) -> list[CapabilityGroup]:
        """Return capability groups (sorted by group name)."""
        grouped: dict[str, list[CapabilityDescriptor]] = {}
        for descriptor in self._descriptors:
            grouped.setdefault(descriptor.group or "default", []).append(descriptor)
        return [
            CapabilityGroup(
                provider_id=self.provider_id,
                name=name,
                capabilities=tuple(sorted(caps, key=lambda d: d.action)),
            )
            for name, caps in sorted(grouped.items(), key=lambda item: item[0].lower())
        ]

    def list_capability_rows(self) -> list[CapabilityConsentRow]:
        """Flat rows with tier, needs_review, and state from ``evaluate_access``."""
        grants = self._store.load()
        rows: list[CapabilityConsentRow] = []
        for group in self.list_groups():
            for descriptor in group.capabilities:
                grant = grants.get(str(descriptor.urn.base))
                decision = evaluate_access(descriptor, grant)
                ui_state = derive_consent_ui_state(
                    descriptor,
                    decision,
                    grant_granted=grant.granted if grant is not None else None,
                )
                rows.append(
                    CapabilityConsentRow(
                        group=group.name,
                        descriptor=descriptor,
                        tier=descriptor.tier,
                        needs_review=descriptor.needs_review,
                        ui_state=ui_state,
                        decision=decision,
                    )
                )
        return rows

    def grant_capability(self, descriptor: CapabilityDescriptor) -> AccessDecision:
        """Persist an explicit allow for ``descriptor`` (exact fingerprint bound)."""
        if descriptor.needs_review:
            raise ValueError("capabilities flagged needs_review cannot be granted via consent")
        self._store.grant(descriptor)
        return evaluate_access(descriptor, self._store.get(descriptor.urn))

    def deny_capability(self, descriptor: CapabilityDescriptor) -> AccessDecision:
        """Persist an explicit deny for ``descriptor``."""
        self._store.deny(descriptor)
        return evaluate_access(descriptor, self._store.get(descriptor.urn))
