"""Agent scope boundaries for capability invocation (Phase 3 / #61).

Tracks which ``cap:`` URNs the user explicitly attached for a session so the
runtime can enforce P1 — the model cannot invoke a capability that was not
attached (or bundled via a preset alias on this turn).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from core.composer_attachments import ComposerAttachment, resolve_attachment_routing
from core.integrations.capabilities.urn import CapabilityURN

__all__ = [
    "AgentScope",
    "AgentScopeStore",
    "agent_scope_store",
    "build_agent_scope_from_attachments",
    "parse_scope_capability_urn",
    "urn_base_key",
]


def parse_scope_capability_urn(token_id: str) -> CapabilityURN | None:
    raw = (token_id or "").strip()
    if not raw:
        return None
    if not raw.startswith("cap:"):
        raw = f"cap:{raw}"
    return CapabilityURN.try_parse(raw)


def urn_base_key(urn: CapabilityURN | str) -> str:
    """Stable base identity for scope membership (ignores version suffix)."""
    if isinstance(urn, str):
        parsed = parse_scope_capability_urn(urn)
        if parsed is None:
            return (urn or "").strip().lower()
        urn = parsed
    return str(urn.base).lower()


@dataclass(frozen=True, slots=True)
class AgentScope:
    """Session-scoped set of user-attached capability identities."""

    session_id: str
    allowed_urn_bases: frozenset[str] = field(default_factory=frozenset)
    preset_id: str | None = None

    def allows(self, urn: CapabilityURN | str) -> bool:
        if not self.allowed_urn_bases:
            return True
        return urn_base_key(urn) in self.allowed_urn_bases

    def check(self, urn: CapabilityURN | str) -> tuple[bool, str]:
        if self.allows(urn):
            return True, "ok"
        return False, "capability not in agent scope for this turn"


def _urns_from_attachments(attachments: Sequence[ComposerAttachment]) -> frozenset[str]:
    from core.integrations.preset_capability_alias import resolve_preset_capability_urns

    bases: set[str] = set()
    routing = resolve_attachment_routing(list(attachments))
    if routing:
        preset_id = str(routing.get("capability_preset_id") or "").strip()
        if preset_id:
            for urn in resolve_preset_capability_urns(preset_id):
                bases.add(urn_base_key(urn))
        for raw in routing.get("capability_urns") or ():
            bases.add(urn_base_key(str(raw)))
        cap_urn = str(routing.get("capability_urn") or "").strip()
        if cap_urn:
            bases.add(urn_base_key(cap_urn))
    for att in attachments:
        if att.kind != "capability":
            continue
        parsed = parse_scope_capability_urn(att.id)
        if parsed is not None:
            bases.add(urn_base_key(parsed))
    return frozenset(bases)


def build_agent_scope_from_attachments(
    session_id: str,
    attachments: Sequence[ComposerAttachment],
) -> AgentScope:
    """Derive scope from composer attachments for one turn."""
    routing = resolve_attachment_routing(list(attachments))
    preset_id = None
    if routing:
        preset_id = str(routing.get("capability_preset_id") or "").strip() or None
    return AgentScope(
        session_id=str(session_id),
        allowed_urn_bases=_urns_from_attachments(attachments),
        preset_id=preset_id,
    )


class AgentScopeStore:
    """In-memory scope registry keyed by session id (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scopes: dict[str, AgentScope] = {}

    def set_scope(self, scope: AgentScope) -> None:
        with self._lock:
            self._scopes[scope.session_id] = scope

    def get_scope(self, session_id: str) -> AgentScope | None:
        with self._lock:
            return self._scopes.get(str(session_id))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._scopes.pop(str(session_id), None)

    def allowed_bases(self, session_id: str) -> frozenset[str]:
        scope = self.get_scope(session_id)
        if scope is None:
            return frozenset()
        return scope.allowed_urn_bases


agent_scope_store = AgentScopeStore()
