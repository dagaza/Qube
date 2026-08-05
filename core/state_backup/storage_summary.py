"""Human-readable backup storage summaries for Settings UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.state_backup.manifest import BACKUP_EXTENSION
from core.state_backup.paths import (
    auto_backups_dir,
    default_backups_dir,
    iter_backup_entries,
    resolve_db_path,
    resolve_user_data_root,
)


def format_byte_size(num_bytes: int) -> str:
    value = max(int(num_bytes), 0)
    if value < 1024:
        return f"{value} B"
    kb = value / 1024
    if kb < 1024:
        return f"{kb:.0f} KB" if kb >= 10 else f"{kb:.1f} KB"
    mb = kb / 1024
    if mb < 1024:
        return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB"
    gb = mb / 1024
    return f"{gb:.1f} GB" if gb < 10 else f"{gb:.0f} GB"


def estimate_backup_uncompressed_bytes(
    *,
    root: Path | None = None,
    include_wallpapers: bool = False,
) -> int:
    """Sum on-disk sizes of files included in a state backup (pre-zip estimate)."""
    base = resolve_user_data_root(root)
    total = 0

    db_path = resolve_db_path(root=base)
    if db_path.is_file():
        total += db_path.stat().st_size

    seen: set[Path] = set()
    for source, arcname in iter_backup_entries(
        root=base,
        include_wallpapers=include_wallpapers,
    ):
        if arcname == "qube_data.db" or not source.is_file():
            continue
        resolved = source.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        total += source.stat().st_size

    return total


@dataclass(frozen=True)
class LatestBackupInfo:
    path: Path
    size_bytes: int
    modified_at: datetime


def find_latest_backup_archive(*, root: Path | None = None) -> LatestBackupInfo | None:
    """Return the newest `.qube-backup.zip` under manual or automatic backup folders."""
    base = resolve_user_data_root(root)
    backups_dir = default_backups_dir(base)
    auto_dir = auto_backups_dir(base)

    best: tuple[float, Path, int] | None = None
    for directory in (backups_dir, auto_dir):
        if not directory.is_dir():
            continue
        for path in directory.glob(f"*{BACKUP_EXTENSION}"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            candidate = (stat.st_mtime, path, stat.st_size)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None

    _mtime, path, size_bytes = best
    return LatestBackupInfo(
        path=path,
        size_bytes=size_bytes,
        modified_at=datetime.fromtimestamp(_mtime, tz=timezone.utc),
    )


def format_storage_summary_text(
    *,
    root: Path | None = None,
    include_wallpapers: bool = False,
) -> str:
    """One-line summary for Settings → Backup & restore."""
    latest = find_latest_backup_archive(root=root)
    models_note = "models not included"
    if latest is not None:
        return f"Last backup {format_byte_size(latest.size_bytes)}; {models_note}."
    estimated = estimate_backup_uncompressed_bytes(
        root=root,
        include_wallpapers=include_wallpapers,
    )
    return f"Estimated backup ~{format_byte_size(estimated)}; {models_note}."
