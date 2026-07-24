"""Performance-related surface fill behavior."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QImage, QPixmap

from core.surface_fill.compositor import (
    SurfaceFillCompositor,
    _load_scaled_image_pixmap,
)
from core.surface_fill.models import WallpaperImage
from core.surface_fill.thumbnails import (
    clear_wallpaper_thumbnail_cache,
    preset_thumbnail_pixmap,
    user_wallpaper_thumbnail_pixmap,
)


def test_load_scaled_image_pixmap_decodes_near_target_size(_qube_app, tmp_path):
    path = tmp_path / "large.png"
    image = QImage(2400, 1350, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    assert image.save(str(path))

    rect = QRect(0, 0, 96, 56)
    pixmap = _load_scaled_image_pixmap(path, rect)
    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.width() <= 2400
    assert pixmap.height() <= 1350
    assert pixmap.width() >= rect.width()
    assert pixmap.height() >= rect.height()


def test_preset_thumbnail_cache_reuses_pixmap(_qube_app):
    clear_wallpaper_thumbnail_cache()
    first = preset_thumbnail_pixmap("builtin.mist", size=48, is_dark=True)
    second = preset_thumbnail_pixmap("builtin.mist", size=48, is_dark=True)
    assert first.cacheKey() == second.cacheKey()


def test_user_thumbnail_cache_reuses_pixmap(_qube_app, monkeypatch, tmp_path):
    from core.surface_fill import thumbnails

    clear_wallpaper_thumbnail_cache()
    image_path = tmp_path / "user-wall.jpg"
    image = QImage(1800, 1000, QImage.Format.Format_RGB32)
    image.fill(0xAA5522)
    assert image.save(str(image_path))

    monkeypatch.setattr(
        thumbnails,
        "wallpapers_directory",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        thumbnails,
        "list_picker_user_image_filenames",
        lambda: ["user-wall.jpg"],
    )

    first = user_wallpaper_thumbnail_pixmap(
        "user-wall.jpg",
        width=96,
        height=56,
        is_dark=True,
    )
    second = user_wallpaper_thumbnail_pixmap(
        "user-wall.jpg",
        width=96,
        height=56,
        is_dark=True,
    )
    assert first.cacheKey() == second.cacheKey()


def test_saturation_scale_is_cached(_qube_app):
    compositor = SurfaceFillCompositor()
    source = QPixmap(32, 32)
    source.fill(0xFF0000)
    first = compositor.apply_saturation_scale(source, 0.7)
    second = compositor.apply_saturation_scale(source, 0.7)
    assert first.cacheKey() == second.cacheKey()


def test_compose_image_wallpaper_uses_cache(_qube_app, tmp_path, monkeypatch):
    path = tmp_path / "wall.jpg"
    image = QImage(800, 600, QImage.Format.Format_RGB32)
    image.fill(0x445566)
    assert image.save(str(path))

    monkeypatch.setattr(
        "core.surface_fill.compositor.resolve_wallpaper_image_path",
        lambda _source: path,
    )
    comp = SurfaceFillCompositor()
    theme = __import__("core.theme.accessors", fromlist=["theme_for"]).theme_for(
        is_dark=True
    )
    rect = QRect(0, 0, 96, 56)
    wallpaper = WallpaperImage(source="ignored.jpg")
    first = comp.compose_wallpaper(wallpaper, rect, theme=theme)
    second = comp.compose_wallpaper(wallpaper, rect, theme=theme)
    assert first.pixmap is not None and second.pixmap is not None
    assert first.pixmap.cacheKey() == second.pixmap.cacheKey()


def test_transcript_wallpaper_host_reuses_background_cache(_qube_app):
    from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
    from core.surface_fill.models import SurfaceProfile, WallpaperSolid
    from PyQt6.QtWidgets import QLabel

    from ui.surface_fill.transcript_host import TranscriptWallpaperHost

    host = TranscriptWallpaperHost(
        SURFACE_CHAT_TRANSCRIPT,
        QLabel("hello"),
    )
    host.set_preview_profile(
        SurfaceProfile(wallpaper=WallpaperSolid(color="#336699")),
        theme=__import__("core.theme.accessors", fromlist=["theme_for"]).theme_for(
            is_dark=True
        ),
    )
    host.resize(120, 80)
    host.show()
    _qube_app.processEvents()

    first = host._background_pixmap()
    second = host._background_pixmap()
    assert first is not None and second is not None
    assert first.cacheKey() == second.cacheKey()
