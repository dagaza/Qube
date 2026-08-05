"""Restore Qube state from a local backup archive."""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.state_backup.export import export_state_backup
from core.state_backup.manifest import BACKUP_MANIFEST_NAME, read_manifest_from_zip, verify_backup_archive
from core.state_backup.paths import default_backups_dir, is_safe_archive_path, resolve_user_data_root

logger = logging.getLogger("Qube.StateBackup")


@dataclass(frozen=True)
class RestoreResult:
    ok: bool
    files_restored: int = 0
    pre_restore_backup: Path | None = None
    error: str | None = None
    requires_restart: bool = True


def _extract_backup(
    archive_path: Path,
    *,
    destination_root: Path,
) -> int:
    restored = 0
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        manifest = read_manifest_from_zip(archive)
        for entry in manifest.files:
            if not is_safe_archive_path(entry.path):
                raise ValueError(f"Unsafe archive path: {entry.path}")
            payload = archive.read(entry.path)
            target = destination_root / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            restored += 1
    return restored


def restore_state_backup(
    source: Path,
    *,
    user_data_root: Path | None = None,
    create_pre_restore_snapshot: bool = True,
    pre_restore_dir: Path | None = None,
) -> RestoreResult:
    """Restore essential Qube state from a verified backup archive."""
    root = resolve_user_data_root(user_data_root)
    archive_path = Path(source).expanduser()

    verification = verify_backup_archive(archive_path)
    if not verification.ok:
        return RestoreResult(ok=False, error=verification.error or "Backup verification failed")

    pre_restore_path: Path | None = None
    if create_pre_restore_snapshot:
        snapshot_dir = pre_restore_dir or default_backups_dir(root)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        pre_restore_path = snapshot_dir / f"pre-restore-{stamp}.qube-backup.zip"
        snapshot = export_state_backup(
            pre_restore_path,
            user_data_root=root,
            include_wallpapers=False,
        )
        if not snapshot.ok:
            return RestoreResult(
                ok=False,
                error=(
                    "Could not create a pre-restore safety snapshot. "
                    f"{snapshot.error or 'Export failed.'}"
                ),
            )

    try:
        restored = _extract_backup(archive_path, destination_root=root)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logger.error("State backup restore failed: %s", exc)
        return RestoreResult(
            ok=False,
            pre_restore_backup=pre_restore_path,
            error=str(exc),
        )

    logger.info(
        "State backup restored from %s (%d files). Restart required.",
        archive_path,
        restored,
    )
    return RestoreResult(
        ok=True,
        files_restored=restored,
        pre_restore_backup=pre_restore_path,
        requires_restart=True,
    )
