"""Paint surface profiles onto transcript wallpaper hosts."""

from __future__ import annotations

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QPainter

from core.surface_fill.compositor import SurfaceFillCompositor
from core.surface_fill.models import OverlaySpec, SurfaceProfile, WallpaperNone
from core.surface_fill.overlay import overlay_render_params, overlay_scrim_rgba
from core.theme.color_utils import theme_qcolor
from core.theme.tokens import ResolvedTheme


class SurfaceFillRenderer:
    """Backend-agnostic painter for wallpaper + overlay stacks."""

    def __init__(self, *, compositor: SurfaceFillCompositor | None = None) -> None:
        self._compositor = compositor or SurfaceFillCompositor()

    def paint(
        self,
        painter: QPainter,
        rect: QRect,
        profile: SurfaceProfile,
        *,
        theme: ResolvedTheme,
        overlay_boost: int = 0,
        suppressed: bool = False,
        resolved_wallpaper=None,
    ) -> None:
        from core.surface_fill.models import Wallpaper

        wallpaper: Wallpaper = resolved_wallpaper or profile.wallpaper

        if suppressed:
            self._compositor.fill_rect_color(painter, rect, theme.background)
            return

        overlay_params = overlay_render_params(
            profile.overlay,
            theme,
            boost=overlay_boost,
        )

        composed = self._compositor.compose_wallpaper(wallpaper, rect, theme=theme)
        if composed.fill_color is not None:
            fill_color = self._compositor.tint_fill_color(
                composed.fill_color,
                saturation_scale=overlay_params.saturation_scale,
            )
            self._compositor.fill_rect_color(painter, rect, fill_color)
        if composed.pixmap is not None and not composed.pixmap.isNull():
            pixmap = self._compositor.apply_saturation_scale(
                composed.pixmap,
                overlay_params.saturation_scale,
            )
            painter.drawPixmap(rect.topLeft(), pixmap)

        if isinstance(wallpaper, WallpaperNone) and composed.fill_color is None:
            self._compositor.fill_rect_color(painter, rect, theme.background)
            return

        if isinstance(wallpaper, WallpaperNone):
            return

        self._paint_overlay(
            painter,
            rect,
            profile.overlay,
            theme=theme,
            overlay_boost=overlay_boost,
        )

    def _paint_overlay(
        self,
        painter: QPainter,
        rect: QRect,
        overlay: OverlaySpec,
        *,
        theme: ResolvedTheme,
        overlay_boost: int,
    ) -> None:
        scrim = overlay_scrim_rgba(overlay, theme, boost=overlay_boost)
        color = theme_qcolor(scrim)
        if color.alpha() <= 0:
            return
        painter.fillRect(rect, color)

    def clear(self, painter: QPainter, rect: QRect, *, theme: ResolvedTheme) -> None:
        self._compositor.fill_rect_color(painter, rect, theme.background)
