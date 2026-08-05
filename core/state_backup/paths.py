"""Paths included in and excluded from essential state backups."""

from __future__ import annotations

from pathlib import Path

from core.licensing.store import DEFAULT_LICENSE_CACHE_NAME
from core.paths import default_db_path, user_data_root

BACKUP_TIER_ESSENTIAL = "essential"

# Top-level directories under ~/.qube that must never be backed up automatically.
EXCLUDED_TOP_LEVEL_DIRS: frozenset[str] = frozenset(
    {
        "models",
        "logs",
        "exports",
        "evidence_cache",
        "avatar_cache",
        "tmp",
        "backups",
    }
)

# Single files at the user data root (relative to user_data_root()).
ESSENTIAL_ROOT_FILES: tuple[str, ...] = (
    "settings.json",
    "model_overrides.json",
    "prompt_layout_overrides.json",
    "user_profile.json",
    "composer_recent_tokens.json",
    "memory_negatives.json",
    "notification_history.json",
    "help_corpus_state.json",
    DEFAULT_LICENSE_CACHE_NAME,
)

# Directories copied recursively (relative to user_data_root()).
ESSENTIAL_DIRECTORIES: tuple[str, ...] = (
    "data/lancedb",
    "knowledge",
    "integrations",
    "themes",
    "system_data",
    "companion",
)

OPTIONAL_WALLPAPERS_DIR = "wallpapers"


def resolve_user_data_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else user_data_root()


def resolve_db_path(*, root: Path | None = None, db_path: Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    base = resolve_user_data_root(root)
    default = default_db_path()
    if default.parent == base or default.parent.resolve() == base.resolve():
        return default
    return base / "qube_data.db"


def default_backups_dir(root: Path | None = None) -> Path:
    path = resolve_user_data_root(root) / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def auto_backups_dir(root: Path | None = None) -> Path:
    path = default_backups_dir(root) / "auto"
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_backup_entries(
    *,
    root: Path | None = None,
    include_wallpapers: bool = False,
) -> list[tuple[Path, str]]:
    """Return (absolute source path, archive-relative path) pairs to include."""
    base = resolve_user_data_root(root)
    entries: list[tuple[Path, str]] = []

    db_path = resolve_db_path(root=base)
    if db_path.is_file():
        entries.append((db_path, "qube_data.db"))

    for rel in ESSENTIAL_ROOT_FILES:
        source = base / rel
        if source.is_file():
            entries.append((source, rel.replace("\\", "/")))

    for rel in ESSENTIAL_DIRECTORIES:
        source = base / rel
        if source.is_dir():
            rel_posix = rel.replace("\\", "/").rstrip("/")
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                arcname = f"{rel_posix}/{path.relative_to(source).as_posix()}"
                entries.append((path, arcname))

    if include_wallpapers:
        wallpapers = base / OPTIONAL_WALLPAPERS_DIR
        if wallpapers.is_dir():
            for path in sorted(wallpapers.rglob("*")):
                if not path.is_file():
                    continue
                arcname = f"{OPTIONAL_WALLPAPERS_DIR}/{path.relative_to(wallpapers).as_posix()}"
                entries.append((path, arcname))

    return entries


def is_safe_archive_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return False
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    top = parts[0]
    if top in EXCLUDED_TOP_LEVEL_DIRS:
        return False
    return True
