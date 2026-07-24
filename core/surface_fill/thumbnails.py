"""Render preset thumbnails for the wallpaper picker."""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QPixmap

from core.surface_fill.compositor import SurfaceFillCompositor
from core.surface_fill.import_wallpaper import list_user_wallpaper_filenames
from core.surface_fill.models import WallpaperImage
from core.surface_fill.presets import get_preset, list_preset_ids, preset_wallpaper
from core.surface_fill.storage import wallpapers_directory
from core.theme.accessors import theme_for
from core.theme.color_utils import theme_qcolor

_THUMBNAIL_CACHE: dict[tuple, QPixmap] = {}


def clear_wallpaper_thumbnail_cache() -> None:
    _THUMBNAIL_CACHE.clear()


def _cached_thumbnail(key: tuple, builder) -> QPixmap:
    cached = _THUMBNAIL_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    pixmap = builder()
    _THUMBNAIL_CACHE[key] = pixmap
    return pixmap


def preset_thumbnail_pixmap(
    preset_id: str,
    size: int = 72,
    *,
    is_dark: bool = True,
) -> QPixmap:
    """Build a square thumbnail for a bundled preset."""
    key = ("preset", preset_id, size, is_dark)

    def _build() -> QPixmap:
        theme = theme_for(is_dark=is_dark)
        compositor = SurfaceFillCompositor()
        try:
            wallpaper = preset_wallpaper(preset_id)
        except KeyError:
            placeholder = QPixmap(size, size)
            placeholder.fill(
                Qt.GlobalColor.darkGray if is_dark else Qt.GlobalColor.lightGray
            )
            return placeholder

        composed = compositor.compose_wallpaper(
            wallpaper,
            QRect(0, 0, size, size),
            theme=theme,
        )
        pixmap = QPixmap(size, size)
        if composed.pixmap is not None and not composed.pixmap.isNull():
            if composed.pixmap.size() == pixmap.size():
                return composed.pixmap
            pixmap = composed.pixmap.scaled(
                size,
                size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        elif composed.fill_color:
            pixmap.fill(theme_qcolor(composed.fill_color))
        else:
            pixmap.fill(theme_qcolor(theme.background))
        return pixmap

    return _cached_thumbnail(key, _build)


def list_picker_preset_ids() -> list[str]:
    """Presets shown in the Settings grid (excludes theme-default-only entries)."""
    return [pid for pid in list_preset_ids() if get_preset(pid) is not None]


def user_wallpaper_thumbnail_pixmap(
    filename: str,
    *,
    width: int = 96,
    height: int = 56,
    is_dark: bool = True,
) -> QPixmap:
    """Build a thumbnail for a user-imported wallpaper image."""
    key = ("user", filename, width, height, is_dark)

    def _build() -> QPixmap:
        theme = theme_for(is_dark=is_dark)
        path = wallpapers_directory() / filename
        placeholder = QPixmap(width, height)
        placeholder.fill(
            Qt.GlobalColor.darkGray if is_dark else Qt.GlobalColor.lightGray
        )
        if not path.is_file():
            return placeholder

        compositor = SurfaceFillCompositor()
        composed = compositor.compose_wallpaper(
            WallpaperImage(source=filename),
            QRect(0, 0, width, height),
            theme=theme,
        )
        if composed.pixmap is not None and not composed.pixmap.isNull():
            if composed.pixmap.size().width() == width and composed.pixmap.size().height() == height:
                return composed.pixmap
            return composed.pixmap.scaled(
                width,
                height,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        if composed.fill_color:
            placeholder.fill(theme_qcolor(composed.fill_color))
        return placeholder

    return _cached_thumbnail(key, _build)


def list_picker_user_image_filenames() -> list[str]:
    """User images shown in the Settings Images grid."""
    return list_user_wallpaper_filenames()
