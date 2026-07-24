"""Resolve wallpaper image paths from bundled, preset, or user storage."""

from __future__ import annotations

from pathlib import Path

from core.paths import resource_path, user_data_root
from core.surface_fill.presets import preset_asset_path


def resolve_wallpaper_image_path(source: str) -> Path | None:
    """Return a readable image path when ``source`` is allowed, else ``None``."""
    raw = str(source or "").strip()
    if not raw:
        return None
    if ".." in raw.replace("\\", "/").split("/"):
        return None

    path = Path(raw)
    if path.is_absolute():
        allowed_roots = (
            user_data_root().resolve(),
            (user_data_root() / "wallpapers").resolve(),
            resource_path().resolve(),
        )
        try:
            resolved = path.resolve()
        except OSError:
            return None
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                return resolved if resolved.is_file() else resolved
            except ValueError:
                continue
        return None

    if raw.startswith("assets/"):
        return resource_path(*raw.split("/"))
    preset_path = preset_asset_path(raw)
    if preset_path is not None:
        return preset_path
    under_user = (user_data_root() / "wallpapers" / raw).resolve()
    if under_user.is_file():
        return under_user
    return resource_path("assets", "wallpapers", raw)
