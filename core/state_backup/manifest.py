"""Manifest schema and verification for Qube state backups."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.__version__ import __version__
from core.state_backup.paths import BACKUP_TIER_ESSENTIAL, is_safe_archive_path

BACKUP_VERSION = 1
BACKUP_MANIFEST_NAME = "manifest.json"
BACKUP_EXTENSION = ".qube-backup.zip"


@dataclass(frozen=True)
class ManifestFileEntry:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class BackupManifest:
    backup_version: int
    created_at: str
    qube_version: str
    tier: str
    includes_wallpapers: bool
    files: tuple[ManifestFileEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_version": self.backup_version,
            "created_at": self.created_at,
            "qube_version": self.qube_version,
            "tier": self.tier,
            "includes_wallpapers": self.includes_wallpapers,
            "files": [
                {"path": entry.path, "sha256": entry.sha256, "bytes": entry.bytes}
                for entry in self.files
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackupManifest:
        version = int(payload.get("backup_version") or 0)
        if version != BACKUP_VERSION:
            raise ValueError(
                f"Unsupported backup version: {version!r} (expected {BACKUP_VERSION})"
            )
        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("Backup manifest requires a non-empty files list")
        files: list[ManifestFileEntry] = []
        for item in raw_files:
            if not isinstance(item, dict):
                raise ValueError("Invalid manifest file entry")
            path = str(item.get("path") or "").strip()
            sha256 = str(item.get("sha256") or "").strip().lower()
            size = int(item.get("bytes") or 0)
            if not path or not sha256 or size < 0:
                raise ValueError(f"Invalid manifest entry for {path!r}")
            if not is_safe_archive_path(path):
                raise ValueError(f"Unsafe manifest path: {path!r}")
            files.append(ManifestFileEntry(path=path, sha256=sha256, bytes=size))
        return cls(
            backup_version=version,
            created_at=str(payload.get("created_at") or ""),
            qube_version=str(payload.get("qube_version") or ""),
            tier=str(payload.get("tier") or BACKUP_TIER_ESSENTIAL),
            includes_wallpapers=bool(payload.get("includes_wallpapers")),
            files=tuple(files),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    staged_files: dict[str, Path],
    *,
    includes_wallpapers: bool = False,
    created_at: datetime | None = None,
) -> BackupManifest:
    when = created_at or datetime.now(timezone.utc)
    entries: list[ManifestFileEntry] = []
    for arcname in sorted(staged_files):
        path = staged_files[arcname]
        entries.append(
            ManifestFileEntry(
                path=arcname,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
        )
    return BackupManifest(
        backup_version=BACKUP_VERSION,
        created_at=when.isoformat(),
        qube_version=__version__,
        tier=BACKUP_TIER_ESSENTIAL,
        includes_wallpapers=includes_wallpapers,
        files=tuple(entries),
    )


def write_manifest(path: Path, manifest: BackupManifest) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest_from_zip(archive: zipfile.ZipFile) -> BackupManifest:
    try:
        raw = archive.read(BACKUP_MANIFEST_NAME)
    except KeyError as exc:
        raise ValueError(f"Missing {BACKUP_MANIFEST_NAME} in backup archive") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Backup manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Backup manifest must be a JSON object")
    return BackupManifest.from_dict(payload)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    manifest: BackupManifest | None = None
    error: str | None = None


def verify_backup_archive(path: Path) -> VerifyResult:
    archive_path = Path(path)
    if not archive_path.is_file():
        return VerifyResult(ok=False, error="Backup file not found")
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest = read_manifest_from_zip(archive)
            names = set(archive.namelist())
            for entry in manifest.files:
                if entry.path not in names:
                    return VerifyResult(
                        ok=False,
                        manifest=manifest,
                        error=f"Missing file in archive: {entry.path}",
                    )
                payload = archive.read(entry.path)
                if len(payload) != entry.bytes:
                    return VerifyResult(
                        ok=False,
                        manifest=manifest,
                        error=f"Size mismatch for {entry.path}",
                    )
                if sha256_bytes(payload) != entry.sha256:
                    return VerifyResult(
                        ok=False,
                        manifest=manifest,
                        error=f"Checksum mismatch for {entry.path}",
                    )
            if BACKUP_MANIFEST_NAME not in names:
                return VerifyResult(ok=False, error=f"Missing {BACKUP_MANIFEST_NAME}")
    except zipfile.BadZipFile:
        return VerifyResult(ok=False, error="Backup file is not a valid zip archive")
    except ValueError as exc:
        return VerifyResult(ok=False, error=str(exc))
    except OSError as exc:
        return VerifyResult(ok=False, error=str(exc))
    return VerifyResult(ok=True, manifest=manifest)


def default_backup_filename(*, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"qube-state-{stamp}{BACKUP_EXTENSION}"
