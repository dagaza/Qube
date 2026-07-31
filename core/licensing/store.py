"""License load, cache, and import (Phase 1.6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.capabilities import invalidate_capabilities_cache
from core.licensing.license_schema import (
    LICENSE_SCHEMA_VERSION,
    LicenseDocument,
    LicenseError,
    parse_license_document,
)
from core.licensing.schema import PackSignatureError
from core.licensing.verify import verify_license_document
from core.paths import user_data_root

LICENSE_CACHE_SCHEMA = 1
DEFAULT_LICENSE_CACHE_NAME = "license.json"

_license_cache_path: Path | None = None


@dataclass(frozen=True)
class LicenseImportResult:
    ok: bool
    document: LicenseDocument | None = None
    error: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class LicenseCacheRecord:
    """On-disk cache entry written after a successful import."""

    cache_schema: int
    imported_at: str
    source_file: str | None
    document: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LicenseCacheRecord:
        schema = int(raw.get("cache_schema") or LICENSE_CACHE_SCHEMA)
        if schema != LICENSE_CACHE_SCHEMA:
            raise LicenseError(f"Unsupported license cache schema: {schema!r}")
        document = raw.get("document")
        if not isinstance(document, dict):
            raise LicenseError("License cache is missing document object")
        return cls(
            cache_schema=schema,
            imported_at=str(raw.get("imported_at") or ""),
            source_file=str(raw.get("source_file") or "") or None,
            document=document,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cache_schema": self.cache_schema,
            "imported_at": self.imported_at,
            "document": self.document,
        }
        if self.source_file:
            payload["source_file"] = self.source_file
        return payload


def default_license_cache_path() -> Path:
    return user_data_root() / DEFAULT_LICENSE_CACHE_NAME


def license_cache_path() -> Path:
    return _license_cache_path or default_license_cache_path()


def set_license_cache_path(path: Path | None) -> None:
    """Override the license cache location (tests and future Settings hook)."""
    global _license_cache_path
    _license_cache_path = Path(path).expanduser() if path is not None else None
    invalidate_capabilities_cache()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LicenseError(f"Unable to read license file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LicenseError(f"License file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LicenseError("License file must contain a JSON object")
    return payload


def _load_cache_record(path: Path) -> LicenseCacheRecord | None:
    if not path.is_file():
        return None
    raw = _read_json(path)
    if "document" in raw:
        return LicenseCacheRecord.from_dict(raw)
    # Allow a raw `.qube-license` document cached directly for convenience.
    return LicenseCacheRecord(
        cache_schema=LICENSE_CACHE_SCHEMA,
        imported_at="",
        source_file=None,
        document=raw,
    )


def _validate_active_document(raw_document: dict[str, Any]) -> LicenseDocument:
    document = parse_license_document(raw_document)
    verify_license_document(raw_document)
    if document.is_expired:
        raise LicenseError("License has expired")
    return document


def get_active_license(*, verify: bool = True) -> LicenseDocument | None:
    """Return the cached license when present, valid, and not expired."""
    path = license_cache_path()
    try:
        record = _load_cache_record(path)
    except LicenseError:
        return None
    if record is None:
        return None
    try:
        if verify:
            return _validate_active_document(record.document)
        return parse_license_document(record.document)
    except (LicenseError, PackSignatureError):
        return None


def load_license_cache_metadata() -> LicenseCacheRecord | None:
    """Return raw cache metadata without rejecting expired licenses."""
    path = license_cache_path()
    try:
        return _load_cache_record(path)
    except LicenseError:
        return None


def import_license_from_path(path: Path) -> LicenseImportResult:
    """Validate, verify, and cache a `.qube-license` file."""
    source = Path(path).expanduser()
    try:
        raw = _read_json(source)
        document = _validate_active_document(raw)
        record = LicenseCacheRecord(
            cache_schema=LICENSE_CACHE_SCHEMA,
            imported_at=datetime.now(timezone.utc).isoformat(),
            source_file=str(source),
            document=raw,
        )
        cache_path = license_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        invalidate_capabilities_cache()
        return LicenseImportResult(
            ok=True,
            document=document,
            source_path=str(source),
        )
    except LicenseError as exc:
        return LicenseImportResult(ok=False, error=str(exc), source_path=str(source))
    except OSError as exc:
        return LicenseImportResult(
            ok=False,
            error=f"Unable to write license cache: {exc}",
            source_path=str(source),
        )


def remove_license() -> bool:
    """Delete the cached license file, if any."""
    path = license_cache_path()
    removed = False
    if path.is_file():
        path.unlink()
        removed = True
    invalidate_capabilities_cache()
    return removed


def license_summary() -> dict[str, Any]:
    """Lightweight license status for Settings UI (Phase 1.8)."""
    record = load_license_cache_metadata()
    if record is None:
        return {"active": False, "cached": False}
    try:
        document = parse_license_document(record.document)
        verify_license_document(record.document)
        expired = document.is_expired
    except (LicenseError, PackSignatureError) as exc:
        return {
            "active": False,
            "cached": True,
            "error": str(exc),
            "source_file": record.source_file,
        }
    return {
        "active": not expired,
        "cached": True,
        "tier": document.tier.value,
        "org_id": document.org_id,
        "seats": document.seats,
        "entitlements": list(document.entitlements),
        "issued": document.issued.isoformat(),
        "expires": document.expires.isoformat() if document.expires else None,
        "source_file": record.source_file,
        "imported_at": record.imported_at or None,
        "license_schema": LICENSE_SCHEMA_VERSION,
    }


def format_license_status_text(summary: Mapping[str, Any]) -> str:
    """Human-readable license status for Settings → License."""
    from core.licensing.display import format_license_status_text as _format

    return _format(summary)
