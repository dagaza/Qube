"""`.qube-license` document schema and validation (Phase 1.4)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from core.capabilities import ALL_CAPABILITY_IDS, EditionTier
from core.licensing.schema import SIGNING_FIELD, PackSigning, extract_signing_block

LICENSE_SCHEMA_VERSION = 1
LICENSE_FILE_EXTENSION = ".qube-license"


class LicenseError(ValueError):
    """Raised when a license document is invalid or rejected."""


@dataclass(frozen=True)
class LicenseDocument:
    """Verified offline license payload."""

    tier: EditionTier
    org_id: str | None
    seats: int
    entitlements: tuple[str, ...]
    issued: datetime
    expires: datetime | None
    license_schema: int = LICENSE_SCHEMA_VERSION

    @property
    def is_expired(self) -> bool:
        if self.expires is None:
            return False
        return datetime.now(timezone.utc) >= self.expires

    def entitlement_overrides(self) -> dict[str, bool]:
        """Extra capability grants declared explicitly in the license."""
        return {cap_id: True for cap_id in self.entitlements}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "license_schema": self.license_schema,
            "tier": self.tier.value,
            "seats": self.seats,
            "entitlements": list(self.entitlements),
            "issued": _format_datetime(self.issued),
        }
        if self.org_id:
            payload["org_id"] = self.org_id
        if self.expires is not None:
            payload["expires"] = _format_datetime(self.expires)
        return payload


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(raw: Any, *, field_name: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise LicenseError(f"License field {field_name!r} is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LicenseError(f"License field {field_name!r} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_tier(raw: Any) -> EditionTier:
    text = str(raw or "").strip().lower()
    if not text:
        raise LicenseError("License tier is required")
    try:
        return EditionTier(text)
    except ValueError as exc:
        allowed = ", ".join(t.value for t in EditionTier)
        raise LicenseError(f"Unsupported license tier {text!r} (expected one of: {allowed})") from exc


def _parse_entitlements(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise LicenseError("License entitlements must be an array")
    entitlements: list[str] = []
    seen: set[str] = set()
    for item in raw:
        cap_id = str(item or "").strip()
        if not cap_id:
            continue
        if cap_id not in ALL_CAPABILITY_IDS:
            raise LicenseError(f"Unknown entitlement capability id: {cap_id!r}")
        if cap_id in seen:
            continue
        seen.add(cap_id)
        entitlements.append(cap_id)
    return tuple(sorted(entitlements))


def license_signing_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical license body used for Ed25519 signing (signing block excluded)."""
    payload = deepcopy(dict(document))
    payload.pop(SIGNING_FIELD, None)
    return payload


def parse_license_document(raw: Mapping[str, Any]) -> LicenseDocument:
    """Parse and validate a license JSON object (signature not checked here)."""
    if not isinstance(raw, Mapping):
        raise LicenseError("License must be a JSON object")

    schema = int(raw.get("license_schema") or LICENSE_SCHEMA_VERSION)
    if schema != LICENSE_SCHEMA_VERSION:
        raise LicenseError(
            f"Unsupported license schema: {schema!r} (expected {LICENSE_SCHEMA_VERSION})"
        )

    tier = _parse_tier(raw.get("tier"))
    org_raw = raw.get("org_id")
    org_id = str(org_raw).strip() if org_raw not in (None, "") else None

    try:
        seats = int(raw.get("seats") or 0)
    except (TypeError, ValueError) as exc:
        raise LicenseError("License seats must be a positive integer") from exc
    if seats < 1:
        raise LicenseError("License seats must be at least 1")

    entitlements = _parse_entitlements(raw.get("entitlements"))
    issued = _parse_datetime(raw.get("issued"), field_name="issued")
    expires_raw = raw.get("expires")
    expires = (
        _parse_datetime(expires_raw, field_name="expires")
        if expires_raw not in (None, "")
        else None
    )
    if expires is not None and expires <= issued:
        raise LicenseError("License expires must be after issued")

    signing = extract_signing_block(raw)
    if signing is None:
        raise LicenseError("License requires a signing block")

    if tier in (EditionTier.TEAM, EditionTier.ENTERPRISE) and not org_id:
        raise LicenseError("Team and Enterprise licenses require org_id")

    return LicenseDocument(
        license_schema=schema,
        tier=tier,
        org_id=org_id,
        seats=seats,
        entitlements=entitlements,
        issued=issued,
        expires=expires,
    )


def license_document_from_dict(raw: Mapping[str, Any]) -> LicenseDocument:
    """Alias for parse_license_document."""
    return parse_license_document(raw)
