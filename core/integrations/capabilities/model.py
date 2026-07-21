"""Core value objects for the provider-agnostic Capability Plane.

These types are deliberately free of any provider specifics. A ``CapabilityProvider``
(e.g. the MCP client) is responsible for translating its native surface (MCP
``tools/list`` entries, Live Source adapters, ...) into :class:`CapabilityDescriptor`
objects, and for returning :class:`NormalizedHit` results from an invocation.

Nothing outside ``core/integrations/providers/<provider>/`` should need to know
which provider produced a capability (principle P6); the URN and these value
objects carry everything the rest of Qube requires.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.integrations.capabilities.urn import CapabilityURN

__all__ = [
    "CapabilityTier",
    "CapabilityDescriptor",
    "CapabilityGroup",
    "NormalizedHit",
    "HealthState",
    "HealthStatus",
    "PermissionGrant",
    "fingerprint_descriptors",
]


class CapabilityTier(str, Enum):
    """Trust/risk tier of a capability, used to drive the permission model.

    Ordering (``READ < WRITE < DESTRUCTIVE``) is meaningful: it lets the drift
    detector recognise a *privilege escalation* (e.g. a capability that was
    ``read`` becoming ``write``) and force re-consent (principle P7).
    """

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        return {"read": 0, "write": 1, "destructive": 2}[self.value]

    def escalates_over(self, other: CapabilityTier) -> bool:
        """True if ``self`` is a higher-privilege tier than ``other``."""
        return self.rank > other.rank


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """A single, user-attachable capability produced by a provider.

    This is the normalized description the registry caches and the UI renders.
    Raw provider tool ids live in ``raw_ref`` and are only surfaced in an
    Advanced view; users think in capabilities, not raw tools.
    """

    urn: CapabilityURN
    group: str
    action: str
    tier: CapabilityTier
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    raw_ref: str | None = None
    # Set when the tier was inferred with low confidence (e.g. an unrecognised
    # action verb defaulted to the most-restrictive tier). The permission UI
    # must force an explicit human decision before such a capability is granted;
    # it is never silently enabled (P3/P7).
    needs_review: bool = False

    @property
    def provider_id(self) -> str:
        return self.urn.provider

    def signature(self) -> dict[str, Any]:
        """Stable, order-independent view used for fingerprinting/drift.

        Deliberately excludes free-text ``description`` and ``tags`` (cosmetic)
        and includes only what changes the *contract* or *risk* of a capability.
        """
        return {
            "urn": str(self.urn.base),
            "tier": self.tier.value,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True, slots=True)
class CapabilityGroup:
    """A named group of capabilities (e.g. "GitHub", "Filesystem").

    Groups are the primary unit shown in the UI and the unit at which users grant
    permissions; individual capabilities within a group carry their own tier.
    """

    provider_id: str
    name: str
    capabilities: tuple[CapabilityDescriptor, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizedHit:
    """A single result from a capability invocation.

    The field names mirror the shape Live Source adapters already emit
    (``title``/``snippet``/``url``/``_adapter``/``retrieval_method``) so results
    flow through the existing ``EvidenceBundle`` / ``all_ui_sources`` pipeline
    with zero special-casing. ``source_cap`` adds provenance (principle P8).
    """

    title: str
    snippet: str
    source_cap: CapabilityURN
    url: str | None = None
    full_text: str | None = None

    def to_evidence_dict(self) -> dict[str, Any]:
        """Render into the legacy adapter hit shape consumed by the spine."""
        return {
            "title": self.title,
            "snippet": self.snippet,
            "full_text": self.full_text,
            "url": self.url,
            "_adapter": str(self.source_cap.base),
            "retrieval_method": self.source_cap.provider,
            "_capability": str(self.source_cap),
        }


class HealthState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Connection health for a provider, surfaced via the Source Status UI."""

    state: HealthState
    latency_ms: float | None = None
    last_error: str | None = None
    last_success_at: str | None = None
    last_invocation_at: str | None = None


@dataclass(frozen=True, slots=True)
class PermissionGrant:
    """A user's consent decision for one capability.

    A grant is bound to the ``fingerprint`` the capability had when consent was
    given. If the provider's fingerprint later changes in a way that escalates
    this capability's tier or alters its schema, the grant is treated as stale
    and must be re-reviewed before use (principles P3/P7).
    """

    urn: CapabilityURN  # stored as the versionless base URN
    tier: CapabilityTier
    granted: bool
    fingerprint: str
    granted_at: str | None = None


def fingerprint_descriptors(descriptors: list[CapabilityDescriptor]) -> str:
    """Deterministic fingerprint of a provider's grouped capabilities.

    Hashes the *contract-relevant* signature of each capability (URN, tier,
    input schema), order-independent. Two providers exposing the same
    capabilities with the same schemas and tiers produce the same fingerprint;
    any added/removed capability, tier change, or schema change flips it, which
    is what the drift detector keys off of.
    """
    signatures = sorted(
        (d.signature() for d in descriptors),
        key=lambda s: s["urn"],
    )
    # ``default=str`` keeps the fingerprint from crashing on a non-JSON value that
    # a provider might place in ``input_schema``; such values still contribute
    # deterministically to the hash.
    canonical = json.dumps(signatures, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
