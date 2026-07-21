"""The provider-agnostic Capability Plane.

Public API for the Capability abstraction. Providers (MCP, Live Sources, ...)
implement :class:`CapabilityProvider`; the rest of Qube depends only on the
types re-exported here. See ``docs/mcp_capability_architecture_review.md``.
"""

from __future__ import annotations

from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    CapabilityGroup,
    CapabilityTier,
    HealthState,
    HealthStatus,
    NormalizedHit,
    PermissionGrant,
    fingerprint_descriptors,
)
from core.integrations.capabilities.protocol import (
    CapabilityInvocationError,
    CapabilityProvider,
    InvokeContext,
)
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.urn import CapabilityURN, InvalidCapabilityURN

__all__ = [
    "CapabilityURN",
    "InvalidCapabilityURN",
    "CapabilityTier",
    "CapabilityDescriptor",
    "CapabilityGroup",
    "NormalizedHit",
    "HealthState",
    "HealthStatus",
    "PermissionGrant",
    "fingerprint_descriptors",
    "CapabilityProvider",
    "InvokeContext",
    "CapabilityInvocationError",
    "CapabilityMapper",
    "RawTool",
]
