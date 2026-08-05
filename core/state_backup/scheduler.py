"""Automatic local backup scheduling and retention."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import app_settings
from core.state_backup.export import export_state_backup
from core.state_backup.manifest import BACKUP_EXTENSION, default_backup_filename
from core.state_backup.paths import auto_backups_dir, resolve_user_data_root

logger = logging.getLogger("Qube.StateBackup")

STARTUP_AUTO_BACKUP_DELAY_MS = 45_000


@dataclass(frozen=True)
class AutoBackupResult:
    ok: bool
    ran: bool = False
    destination: Path | None = None
    file_count: int = 0
    total_bytes: int = 0
    pruned_count: int = 0
    error: str | None = None


def _parse_last_run_at(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_auto_backup_due(*, now: datetime | None = None) -> bool:
    if not app_settings.get_backup_auto_enabled():
        return False
    last = _parse_last_run_at(app_settings.get_backup_last_run_at())
    if last is None:
        return True
    interval = app_settings.get_backup_interval_days()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return current - last >= timedelta(days=interval)


def prune_auto_backups(*, root: Path | None = None) -> int:
    directory = auto_backups_dir(root)
    keep = app_settings.get_backup_retention_count()
    archives = sorted(
        (
            path
            for path in directory.glob(f"*{BACKUP_EXTENSION}")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for path in archives[keep:]:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Failed to prune auto backup %s: %s", path, exc)
    return removed


def run_auto_backup_if_due(*, root: Path | None = None) -> AutoBackupResult:
    """Export an automatic backup when enabled and the interval has elapsed."""
    if not is_auto_backup_due():
        return AutoBackupResult(ok=True, ran=False)

    data_root = resolve_user_data_root(root)
    destination = auto_backups_dir(data_root) / default_backup_filename()
    include_wallpapers = app_settings.get_backup_include_wallpapers()
    export_result = export_state_backup(
        destination,
        user_data_root=data_root,
        include_wallpapers=include_wallpapers,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    app_settings.set_backup_last_run_at(now_iso)

    if not export_result.ok:
        app_settings.set_backup_last_run_status("failed")
        app_settings.set_backup_last_run_path("")
        logger.error("Automatic state backup failed: %s", export_result.error)
        return AutoBackupResult(
            ok=False,
            ran=True,
            error=export_result.error or "Automatic backup failed.",
        )

    app_settings.set_backup_last_run_status("success")
    app_settings.set_backup_last_run_path(str(export_result.destination or destination))
    pruned = prune_auto_backups(root=data_root)
    logger.info(
        "Automatic state backup saved to %s (%d files, pruned %d older archive(s))",
        export_result.destination,
        export_result.file_count,
        pruned,
    )
    return AutoBackupResult(
        ok=True,
        ran=True,
        destination=export_result.destination,
        file_count=export_result.file_count,
        total_bytes=export_result.total_bytes,
        pruned_count=pruned,
    )
