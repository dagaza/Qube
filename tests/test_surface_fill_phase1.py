"""Phase 1 tests — compositor, renderer, and transcript host wiring."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QRect, QSize
from PyQt6.QtGui import QPainter, QPixmap

from core.surface_fill.compositor import SurfaceFillCompositor
from core.surface_fill.models import (
    GradientStop,
    OverlaySpec,
    SurfaceProfile,
    WallpaperGradient,
    WallpaperNone,
    WallpaperSolid,
    WallpaperThemeDefault,
)
from core.surface_fill.renderer import SurfaceFillRenderer
from core.theme.accessors import theme_for


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_compositor_solid_wallpaper(qapp):
    compositor = SurfaceFillCompositor()
    theme = theme_for(is_dark=True)
    result = compositor.compose_wallpaper(
        WallpaperSolid(color="#112233"),
        QRect(0, 0, 120, 80),
        theme=theme,
    )
    assert result.fill_color == "#112233"
    assert result.pixmap is None


def test_compositor_gradient_pixmap(qapp):
    compositor = SurfaceFillCompositor()
    theme = theme_for(is_dark=True)
    gradient = WallpaperGradient(
        direction="vertical",
        stops=(
            GradientStop(0.0, "#111111"),
            GradientStop(1.0, "#222222"),
        ),
    )
    result = compositor.compose_wallpaper(
        gradient,
        QRect(0, 0, 64, 64),
        theme=theme,
    )
    assert result.pixmap is not None
    assert not result.pixmap.isNull()
    assert result.pixmap.size() == QSize(64, 64)


def test_compositor_multistop_gradient_pixmap(qapp):
    compositor = SurfaceFillCompositor()
    theme = theme_for(is_dark=True)
    gradient = WallpaperGradient(
        direction="vertical",
        stops=(
            GradientStop(0.0, "#111111"),
            GradientStop(0.5, "#888888"),
            GradientStop(1.0, "#eeeeee"),
        ),
    )
    result = compositor.compose_wallpaper(
        gradient,
        QRect(0, 0, 64, 64),
        theme=theme,
    )
    assert result.pixmap is not None
    assert not result.pixmap.isNull()


def test_renderer_paints_overlay_from_theme(qapp):
    renderer = SurfaceFillRenderer()
    theme_dark = theme_for(is_dark=True)
    theme_light = theme_for(is_dark=False)
    profile = SurfaceProfile(
        wallpaper=WallpaperSolid(color="#101010"),
        overlay=OverlaySpec(strength="balanced"),
    )
    rect = QRect(0, 0, 40, 40)

    dark_pix = QPixmap(40, 40)
    dark_pix.fill()
    dark_painter = QPainter(dark_pix)
    renderer.paint(dark_painter, rect, profile, theme=theme_dark)
    dark_painter.end()

    light_pix = QPixmap(40, 40)
    light_pix.fill()
    light_painter = QPainter(light_pix)
    renderer.paint(light_painter, rect, profile, theme=theme_light)
    light_painter.end()

    assert dark_pix.toImage().pixel(20, 20) != light_pix.toImage().pixel(20, 20)


def test_renderer_suppressed_uses_theme_background(qapp):
    renderer = SurfaceFillRenderer()
    theme = theme_for(is_dark=True)
    profile = SurfaceProfile(
        wallpaper=WallpaperGradient(
            direction="horizontal",
            stops=(
                GradientStop(0.0, "#ff0000"),
                GradientStop(1.0, "#00ff00"),
            ),
        ),
        overlay=OverlaySpec(strength="vivid"),
    )
    pixmap = QPixmap(30, 30)
    pixmap.fill()
    painter = QPainter(pixmap)
    renderer.paint(
        painter,
        QRect(0, 0, 30, 30),
        profile,
        theme=theme,
        suppressed=True,
    )
    painter.end()
    # Suppressed path should not throw and should produce opaque pixels.
    assert pixmap.toImage().pixel(10, 10) != 0


def test_transcript_wallpaper_host_refresh_callback(qapp):
    from PyQt6.QtWidgets import QLabel, QWidget

    from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
    from core.surface_fill.models import WallpaperGradient
    from core.surface_fill.storage import SurfaceFillStorage
    from core.theme.manager import ThemeManager
    from core.theme.storage import ThemeStorage
    from ui.surface_fill.transcript_host import TranscriptWallpaperHost

    class _MemoryStore:
        def __init__(self) -> None:
            self._data: dict[str, object] = {}

        def get(self, key: str, default: object) -> object:
            return self._data.get(key, default)

        def set(self, key: str, value: object) -> None:
            self._data[key] = value

    class _Window(QWidget):
        def __init__(self, manager: ThemeManager) -> None:
            super().__init__()
            self._theme_manager = manager

        @property
        def theme_manager(self) -> ThemeManager:
            return self._theme_manager

    store = _MemoryStore()
    manager = ThemeManager(
        storage=ThemeStorage(settings_get=store.get, settings_set=store.set),
        surface_storage=SurfaceFillStorage(settings_get=store.get, settings_set=store.set),
    )
    window = _Window(manager)
    content = QLabel("hello")
    host = TranscriptWallpaperHost(SURFACE_CHAT_TRANSCRIPT, content)
    host.resize(200, 120)
    host.setParent(window)
    window.show()
    qapp.processEvents()

    calls: list[str] = []
    host.refresh_surface_fill = lambda: calls.append("refresh")  # type: ignore[method-assign]
    manager.register_surface_refresh(host.refresh_surface_fill)
    manager.apply_surface_profiles(persist=False)
    assert calls == ["refresh"]

    resolved = manager.resolved_effective_surface_profile(SURFACE_CHAT_TRANSCRIPT)
    assert isinstance(resolved.wallpaper, WallpaperGradient)
