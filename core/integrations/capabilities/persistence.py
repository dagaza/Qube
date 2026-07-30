"""Durable state for the Capability Plane: descriptor cache + consent store.

Two *separate* concerns, two separate files under
``<user_data>/integrations/<provider_id>/`` (P3/P7):

* ``descriptors.json`` — a cache of what a provider last discovered (its
  capabilities + fingerprint). Written on every successful discovery. It is
  purely informational; it never grants anything.
* ``consent.json`` — the user's explicit grant decisions. Written *only* when a
  user reviews and decides. Discovery must never touch it, so re-discovering a
  server can never silently widen privilege.

This module is provider-agnostic (no MCP specifics, no ``provider ==`` branch):
it works for any :class:`CapabilityProvider`. Access evaluation is **default-deny**
and drift-aware — a grant is bound to the fingerprint the capability had when it
was given, and any tier escalation or contract change invalidates it until the
user re-reviews.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    CapabilityTier,
    PermissionGrant,
    fingerprint_descriptors,
)
from core.integrations.capabilities.urn import CapabilityURN, InvalidCapabilityURN
from core.paths import user_data_root

logger = logging.getLogger("Qube.Integrations.Persistence")

__all__ = [
    "integrations_dir",
    "capability_fingerprint",
    "save_descriptor_cache",
    "load_descriptor_cache",
    "ConsentStore",
    "AccessDecision",
    "evaluate_access",
    "prune_consent_for_namespaces",
]

_CONSENT_SCHEMA_VERSION = 1


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def integrations_dir(provider_id: str) -> Path:
    """Return (creating) ``<user_data>/integrations/<provider_id>/``.

    Uses the canonical :func:`core.paths.user_data_root` so the location is
    correct and consistent across platforms (``%LOCALAPPDATA%\\Qube`` on Windows,
    ``~/.qube`` elsewhere) instead of a hard-coded home path.
    """
    segment = (provider_id or "").strip().lower() or "unknown"
    path = user_data_root() / "integrations" / segment
    path.mkdir(parents=True, exist_ok=True)
    return path


def capability_fingerprint(descriptor: CapabilityDescriptor) -> str:
    """Stable per-capability fingerprint (tier + schema + base URN).

    A grant binds to this value; if the capability's contract or risk changes,
    the fingerprint changes and the grant is treated as stale (P3/P7). Reuses
    :func:`fingerprint_descriptors` so a single capability and a provider's full
    set are hashed the same way.
    """
    return fingerprint_descriptors([descriptor])


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + replace so a crash cannot corrupt the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# -- descriptor cache -----------------------------------------------------


def save_descriptor_cache(
    provider_id: str,
    descriptors: list[CapabilityDescriptor],
) -> Path:
    """Persist a provider's discovered capabilities (informational cache)."""
    path = integrations_dir(provider_id) / "descriptors.json"
    payload = {
        "schema_version": _CONSENT_SCHEMA_VERSION,
        "provider_id": provider_id,
        "discovered_at": _iso_now(),
        "fingerprint": fingerprint_descriptors(descriptors),
        "capabilities": [
            {
                "urn": str(d.urn.base),
                "group": d.group,
                "action": d.action,
                "tier": d.tier.value,
                "description": d.description,
                "input_schema": d.input_schema,
                "raw_ref": d.raw_ref,
                "needs_review": d.needs_review,
                "fingerprint": capability_fingerprint(d),
            }
            for d in descriptors
        ],
    }
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return path


def load_descriptor_cache(provider_id: str) -> dict[str, Any]:
    """Load the cached descriptor payload, or ``{}`` if none exists/parseable."""
    path = integrations_dir(provider_id) / "descriptors.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt cache is non-fatal
        logger.warning("[integrations] unreadable descriptor cache %s: %s", path, exc)
        return {}


# -- consent store --------------------------------------------------------


class ConsentStore:
    """Read/write the user's explicit grant decisions for one provider.

    Grants are keyed by the versionless base URN. The store never auto-creates a
    grant; only an explicit :meth:`grant`/:meth:`deny` (driven by the permission
    UI in a later phase) writes to disk.
    """

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self._path = integrations_dir(provider_id) / "consent.json"

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, PermissionGrant]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[integrations] unreadable consent file %s: %s", self._path, exc)
            return {}
        grants: dict[str, PermissionGrant] = {}
        for record in payload.get("grants", []):
            grant = _grant_from_dict(record)
            if grant is not None:
                grants[str(grant.urn.base)] = grant
        return grants

    def save(self, grants: dict[str, PermissionGrant]) -> Path:
        payload = {
            "schema_version": _CONSENT_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "grants": [_grant_to_dict(g) for g in grants.values()],
        }
        _atomic_write(
            self._path, json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
        return self._path

    def get(self, urn: CapabilityURN) -> PermissionGrant | None:
        return self.load().get(str(urn.base))

    def grant(self, descriptor: CapabilityDescriptor) -> PermissionGrant:
        """Record an explicit *allow* bound to the capability's fingerprint."""
        return self._record(descriptor, granted=True)

    def deny(self, descriptor: CapabilityDescriptor) -> PermissionGrant:
        """Record an explicit *deny* (distinct from 'no decision yet')."""
        return self._record(descriptor, granted=False)

    def _record(self, descriptor: CapabilityDescriptor, *, granted: bool) -> PermissionGrant:
        grants = self.load()
        grant = PermissionGrant(
            urn=descriptor.urn.base,
            tier=descriptor.tier,
            granted=granted,
            fingerprint=capability_fingerprint(descriptor),
            granted_at=_iso_now(),
        )
        grants[str(descriptor.urn.base)] = grant
        self.save(grants)
        return grant


def _grant_to_dict(grant: PermissionGrant) -> dict[str, Any]:
    return {
        "urn": str(grant.urn.base),
        "tier": grant.tier.value,
        "granted": grant.granted,
        "fingerprint": grant.fingerprint,
        "granted_at": grant.granted_at,
    }


def _grant_from_dict(record: dict[str, Any]) -> PermissionGrant | None:
    try:
        urn = CapabilityURN.parse(str(record["urn"])).base
        return PermissionGrant(
            urn=urn,
            tier=CapabilityTier(str(record["tier"])),
            granted=bool(record["granted"]),
            fingerprint=str(record["fingerprint"]),
            granted_at=record.get("granted_at"),
        )
    except (KeyError, ValueError, InvalidCapabilityURN) as exc:
        logger.warning("[integrations] skipping malformed grant %r: %s", record, exc)
        return None


def prune_consent_for_namespaces(
    provider_id: str,
    allowed_namespaces: frozenset[str] | set[str],
) -> int:
    """Remove consent grants whose namespace is not in ``allowed_namespaces``."""
    store = ConsentStore(provider_id)
    grants = store.load()
    if not grants:
        return 0
    allowed = {(ns or "").strip().lower() for ns in allowed_namespaces}
    kept: dict[str, PermissionGrant] = {}
    removed = 0
    for urn_base, grant in grants.items():
        try:
            ns = CapabilityURN.parse(str(urn_base)).namespace.strip().lower()
        except (ValueError, InvalidCapabilityURN):
            removed += 1
            continue
        if ns in allowed:
            kept[urn_base] = grant
        else:
            removed += 1
    if not removed:
        return 0
    if kept:
        store.save(kept)
    elif store.path.exists():
        store.path.unlink(missing_ok=True)
    return removed


# -- access evaluation (default-deny, drift-aware) ------------------------


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """The outcome of evaluating a capability against stored consent."""

    allowed: bool
    reason: str
    needs_review: bool = False


def evaluate_access(
    descriptor: CapabilityDescriptor,
    grant: PermissionGrant | None,
) -> AccessDecision:
    """Decide whether ``descriptor`` may be invoked, given the stored ``grant``.

    Default-deny and drift-aware (P7):

    * no grant, or an explicit deny -> **deny**;
    * capability flagged ``needs_review`` -> **deny** (force explicit review);
    * grant fingerprint != current fingerprint (contract/tier drift) -> **deny**;
    * current tier escalates over the granted tier -> **deny**;
    * otherwise -> **allow**.
    """
    if descriptor.needs_review:
        return AccessDecision(False, "capability needs explicit review", needs_review=True)
    if grant is None:
        return AccessDecision(False, "no grant on record (default-deny)")
    if not grant.granted:
        return AccessDecision(False, "capability was explicitly denied")
    if grant.fingerprint != capability_fingerprint(descriptor):
        return AccessDecision(
            False, "capability changed since consent — re-review required"
        )
    if descriptor.tier.escalates_over(grant.tier):
        return AccessDecision(
            False,
            f"tier escalated {grant.tier.value} -> {descriptor.tier.value} — re-review required",
        )
    return AccessDecision(True, "granted")
