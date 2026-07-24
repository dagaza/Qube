"""Phase 3 tests — import downscale and compositor cache."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QRect

from core.surface_fill.compositor import SurfaceFillCompositor
from core.surface_fill.compositor_cache import (
    clear_image_composite_cache,
    get_cached_image,
)
from core.surface_fill.image_import import MAX_STORED_DIMENSION, prepare_wallpaper_image
from core.surface_fill.import_wallpaper import import_wallpaper_image
from core.surface_fill.models import WallpaperImage
from core.theme.accessors import theme_for


def _write_test_jpeg(path: Path, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), color=(30, 64, 120))
    image.save(path, format="JPEG", quality=95)


def test_prepare_wallpaper_downscales_large_image(tmp_path):
    source = tmp_path / "large.jpg"
    dest = tmp_path / "stored.jpg"
    _write_test_jpeg(source, 4000, 3000)

    result = prepare_wallpaper_image(source, dest)

    assert result.downscaled is True
    assert result.stored_dimensions is not None
    assert max(result.stored_dimensions) <= MAX_STORED_DIMENSION
    assert dest.is_file()
    with Image.open(dest) as stored:
        assert max(stored.size) <= MAX_STORED_DIMENSION


def test_import_wallpaper_returns_result_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: tmp_path,
    )
    source = tmp_path / "hero.jpg"
    _write_test_jpeg(source, 3200, 1800)

    result = import_wallpaper_image(source)

    assert result.filename.endswith(".jpg")
    assert (tmp_path / result.filename).is_file()
    assert result.downscaled is True


def test_compositor_image_cache_hit(_qube_app, tmp_path, monkeypatch):
    clear_image_composite_cache()
    image_path = tmp_path / "cache.jpg"
    _write_test_jpeg(image_path, 800, 600)
    monkeypatch.setattr(
        "core.surface_fill.compositor.resolve_wallpaper_image_path",
        lambda _source: image_path,
    )

    compositor = SurfaceFillCompositor()
    theme = theme_for(is_dark=True)
    wallpaper = WallpaperImage(source=str(image_path))
    rect = QRect(0, 0, 640, 480)

    first = compositor.compose_wallpaper(wallpaper, rect, theme=theme)
    second = compositor.compose_wallpaper(wallpaper, rect, theme=theme)

    assert first.pixmap is not None and not first.pixmap.isNull()
    assert second.pixmap is not None and not second.pixmap.isNull()
    from core.surface_fill.compositor_cache import cache_key_for_path

    key = cache_key_for_path(image_path, rect.width(), rect.height())
    assert key is not None
    assert get_cached_image(key) is not None
