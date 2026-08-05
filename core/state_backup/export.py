"""Build a local Qube state backup archive."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.state_backup.manifest import (
    BACKUP_MANIFEST_NAME,
    build_manifest,
    default_backup_filename,
    write_manifest,
)
from core.state_backup.paths import (
    ESSENTIAL_ROOT_FILES,
    iter_backup_entries,
    resolve_db_path,
    resolve_user_data_root,
)

logger = logging.getLogger("Qube.StateBackup")


@dataclass(frozen=True)
class ExportResult:
    ok: bool
    destination: Path | None = None
    file_count: int = 0
    total_bytes: int = 0
    error: str | None = None


def _sqlite_backup(source_db: Path, destination_db: Path) -> None:
    destination_db.parent.mkdir(parents=True, exist_ok=True)
    if destination_db.exists():
        destination_db.unlink()
    source = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(str(destination_db))
        try:
            source.backup(dest)
            dest.commit()
        finally:
            dest.close()
    finally:
        source.close()


def _stage_backup_tree(
    staging: Path,
    *,
    root: Path,
    db_path: Path,
    include_wallpapers: bool,
) -> dict[str, Path]:
    staged: dict[str, Path] = {}

    if db_path.is_file():
        dest_db = staging / "qube_data.db"
        _sqlite_backup(db_path, dest_db)
        staged["qube_data.db"] = dest_db

    for source, arcname in iter_backup_entries(
        root=root,
        include_wallpapers=include_wallpapers,
    ):
        if arcname == "qube_data.db":
            continue
        dest = staging / arcname
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        staged[arcname] = dest

    return staged


def export_state_backup(
    destination: Path,
    *,
    user_data_root: Path | None = None,
    db_path: Path | None = None,
    include_wallpapers: bool = False,
) -> ExportResult:
    """Write a verified zip backup of essential Qube state."""
    root = resolve_user_data_root(user_data_root)
    db = resolve_db_path(root=root, db_path=db_path)
    dest = Path(destination).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    has_db = db.is_file()
    has_any_root = any((root / rel).is_file() for rel in ESSENTIAL_ROOT_FILES)
    has_any_dir = any((root / rel).is_dir() for rel in (
        "data/lancedb",
        "knowledge",
        "integrations",
        "themes",
        "system_data",
        "companion",
    ))
    if not has_db and not has_any_root and not has_any_dir:
        return ExportResult(ok=False, error="No Qube state found to back up")

    tmp_parent = root / "tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="qube-backup-", dir=tmp_parent))
    try:
        staged_files = _stage_backup_tree(
            staging,
            root=root,
            db_path=db,
            include_wallpapers=include_wallpapers,
        )
        if not staged_files:
            return ExportResult(ok=False, error="No files were staged for backup")

        manifest = build_manifest(
            staged_files,
            includes_wallpapers=include_wallpapers,
            created_at=datetime.now(timezone.utc),
        )
        manifest_path = staging / BACKUP_MANIFEST_NAME
        write_manifest(manifest_path, manifest)

        tmp_zip = dest.with_suffix(dest.suffix + ".partial")
        if tmp_zip.exists():
            tmp_zip.unlink()
        with zipfile.ZipFile(tmp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, arcname=BACKUP_MANIFEST_NAME)
            for arcname in sorted(staged_files):
                archive.write(staged_files[arcname], arcname=arcname)

        tmp_zip.replace(dest)
        total_bytes = sum(entry.bytes for entry in manifest.files)
        logger.info(
            "State backup exported to %s (%d files, %d bytes)",
            dest,
            len(manifest.files),
            total_bytes,
        )
        return ExportResult(
            ok=True,
            destination=dest,
            file_count=len(manifest.files),
            total_bytes=total_bytes,
        )
    except OSError as exc:
        logger.error("State backup export failed: %s", exc)
        return ExportResult(ok=False, error=str(exc))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def export_state_backup_to_default_path(
    directory: Path,
    *,
    user_data_root: Path | None = None,
    db_path: Path | None = None,
    include_wallpapers: bool = False,
    filename: str | None = None,
) -> ExportResult:
    name = filename or default_backup_filename()
    return export_state_backup(
        Path(directory) / name,
        user_data_root=user_data_root,
        db_path=db_path,
        include_wallpapers=include_wallpapers,
    )
