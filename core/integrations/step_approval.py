"""Per-invocation approval for write/destructive capabilities (Phase 3 / #61).

Settings consent grants tier access; agent mode still requires an explicit
per-step approval before WRITE/DESTRUCTIVE invokes run (architecture §5).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

from core.integrations.agent_scope import urn_base_key
from core.integrations.capabilities.model import CapabilityDescriptor, CapabilityTier
from core.integrations.capabilities.urn import CapabilityURN

__all__ = [
    "StepApprovalKey",
    "StepApprovalStore",
    "requires_step_approval",
    "step_approval_store",
]


@dataclass(frozen=True, slots=True)
class StepApprovalKey:
    session_id: str
    turn_id: str
    urn_base: str


def requires_step_approval(descriptor: CapabilityDescriptor) -> bool:
    """True when invoke-time confirmation is required beyond stored consent."""
    if descriptor.needs_review:
        return True
    return descriptor.tier is not CapabilityTier.READ


class StepApprovalStore:
    """One-shot approvals keyed by session + turn + capability base."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._approved: set[StepApprovalKey] = set()

    def _key(
        self,
        session_id: str,
        turn_id: str,
        urn: CapabilityURN | str,
    ) -> StepApprovalKey:
        return StepApprovalKey(
            session_id=str(session_id),
            turn_id=str(turn_id),
            urn_base=urn_base_key(urn),
        )

    def grant(
        self,
        session_id: str,
        turn_id: str,
        urn: CapabilityURN | str,
    ) -> None:
        with self._lock:
            self._approved.add(self._key(session_id, turn_id, urn))

    def grant_many(
        self,
        session_id: str,
        turn_id: str,
        urns: Iterable[str],
    ) -> None:
        for urn in urns:
            self.grant(session_id, turn_id, urn)

    def has_approval(
        self,
        session_id: str | None,
        turn_id: str | None,
        urn: CapabilityURN | str,
    ) -> bool:
        if not session_id or not turn_id:
            return False
        with self._lock:
            return self._key(session_id, turn_id, urn) in self._approved

    def clear_session(self, session_id: str) -> None:
        sid = str(session_id)
        with self._lock:
            self._approved = {
                key for key in self._approved if key.session_id != sid
            }


step_approval_store = StepApprovalStore()
