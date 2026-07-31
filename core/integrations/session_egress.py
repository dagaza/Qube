"""Session integration egress ledger (Phase 3 / #61).

Records every capability invoke attempt for privacy / egress transparency
(Theme B). Distinct from :mod:`core.knowledge.egress_policy` (HTTP SSRF).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.integrations.capabilities.model import CapabilityDescriptor, CapabilityTier
from core.integrations.capabilities.urn import CapabilityURN

__all__ = [
    "IntegrationEgressRecord",
    "SessionEgressLedger",
    "build_egress_record",
    "session_egress_ledger",
]


@dataclass(frozen=True, slots=True)
class IntegrationEgressRecord:
    """One integration capability invoke attempt in a session."""

    session_id: str
    turn_id: str
    timestamp_ms: float
    provider_id: str
    server_id: str
    capability_group: str
    urn: str
    tier: str
    allowed: bool
    reason: str = ""
    raw_tool: str | None = None
    latency_ms: float = 0.0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "provider_id": self.provider_id,
            "server_id": self.server_id,
            "capability_group": self.capability_group,
            "urn": self.urn,
            "tier": self.tier,
            "allowed": self.allowed,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "dry_run": self.dry_run,
        }
        if self.raw_tool:
            payload["raw_tool"] = self.raw_tool
        return payload


def build_egress_record(
    *,
    session_id: str,
    turn_id: str,
    urn: CapabilityURN | str,
    descriptor: CapabilityDescriptor | None,
    allowed: bool,
    reason: str = "",
    latency_ms: float = 0.0,
    dry_run: bool = False,
    include_raw_tool: bool = False,
    timestamp_ms: float | None = None,
) -> IntegrationEgressRecord:
    """Build a provider-agnostic egress record from invoke metadata."""
    urn_str = str(urn)
    parsed = urn if isinstance(urn, CapabilityURN) else CapabilityURN.try_parse(urn_str)
    provider_id = descriptor.provider_id if descriptor else (parsed.provider if parsed else "")
    server_id = parsed.namespace if parsed else (descriptor.urn.namespace if descriptor else "")
    group = descriptor.group if descriptor else ""
    tier = descriptor.tier.value if descriptor else CapabilityTier.READ.value
    raw_tool = None
    if include_raw_tool and descriptor and descriptor.raw_ref:
        raw_tool = str(descriptor.raw_ref)
    return IntegrationEgressRecord(
        session_id=str(session_id),
        turn_id=str(turn_id),
        timestamp_ms=timestamp_ms if timestamp_ms is not None else time.time() * 1000.0,
        provider_id=str(provider_id or ""),
        server_id=str(server_id or ""),
        capability_group=str(group or ""),
        urn=urn_str,
        tier=str(tier),
        allowed=bool(allowed),
        reason=str(reason or ""),
        raw_tool=raw_tool,
        latency_ms=float(latency_ms),
        dry_run=bool(dry_run),
    )


class SessionEgressLedger:
    """Append-only in-memory ledger of integration egress records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, list[IntegrationEgressRecord]] = {}

    def record(self, entry: IntegrationEgressRecord) -> None:
        with self._lock:
            bucket = self._records.setdefault(entry.session_id, [])
            bucket.append(entry)

    def records_for_session(self, session_id: str) -> tuple[IntegrationEgressRecord, ...]:
        with self._lock:
            return tuple(self._records.get(str(session_id), ()))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._records.pop(str(session_id), None)

    def all_sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._records.keys())


session_egress_ledger = SessionEgressLedger()
