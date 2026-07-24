"""Import user wallpaper images into ~/.qube/wallpapers/."""

from __future__ import annotations

import re
from pathlib import Path

from core.surface_fill.compositor_cache import clear_image_composite_cache
from core.surface_fill.image_import import WallpaperImportResult, prepare_wallpaper_image
from core.surface_fill.storage import wallpapers_directory

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
USER_WALLPAPER_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def user_wallpaper_storage_name(source: str) -> str:
    """Normalize a stored wallpaper image reference to a user-dir filename."""
    raw = str(source or "").strip()
    if not raw:
        return ""
    return Path(raw).name


def list_user_wallpaper_filenames() -> list[str]:
    """Return filenames in ``~/.qube/wallpapers/`` (newest first)."""
    directory = wallpapers_directory()
    files: list[tuple[float, str]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in USER_WALLPAPER_EXTENSIONS:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        files.append((mtime, path.name))
    files.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in files]


def _sanitize_filename(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    cleaned = _SAFE_NAME_RE.sub("-", stem).strip("-._")
    if not cleaned:
        cleaned = "wallpaper"
    return f"{cleaned}{suffix}"


def import_wallpaper_image(source_path: Path) -> WallpaperImportResult:
    """Import ``source_path`` into the user wallpapers dir (downscale when needed)."""
    src = Path(source_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    dest_dir = wallpapers_directory()
    filename = _sanitize_filename(src.name)
    dest = dest_dir / filename
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        counter = 2
        while dest.exists():
            dest = dest_dir / f"{stem}-{counter}{suffix}"
            counter += 1
    result = prepare_wallpaper_image(src, dest)
    clear_image_composite_cache()
    return result
