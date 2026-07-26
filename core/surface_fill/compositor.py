"""Compose wallpaper layers into paint-ready output."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRect, QSize, Qt
from PyQt6.QtGui import QImage, QImageReader, QLinearGradient, QPainter, QPixmap

from core.surface_fill.compositor_cache import (
    cache_key_for_path,
    get_cached_image,
    store_cached_image,
)
from core.surface_fill.image_paths import resolve_wallpaper_image_path
from core.surface_fill.models import (
    Wallpaper,
    WallpaperGradient,
    WallpaperImage,
    WallpaperNone,
    WallpaperSolid,
)
from core.theme.color_utils import adjust_saturation, theme_qcolor
from core.theme.tokens import ResolvedTheme

_SATURATION_CACHE: OrderedDict[tuple[int, float], QPixmap] = OrderedDict()
_SATURATION_CACHE_MAX = 48


def _load_scaled_image_pixmap(path: Path, rect: QRect) -> QPixmap | None:
    """Decode only as many pixels as needed for *rect* (KeepAspectRatioByExpanding)."""
    target_w = max(1, rect.width())
    target_h = max(1, rect.height())

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid() and source_size.width() > 0 and source_size.height() > 0:
        scale = max(
            target_w / source_size.width(),
            target_h / source_size.height(),
        )
        decode_w = max(1, int(round(source_size.width() * scale)))
        decode_h = max(1, int(round(source_size.height() * scale)))
        reader.setScaledSize(QSize(decode_w, decode_h))
        image = reader.read()
        if not image.isNull():
            return QPixmap.fromImage(image)

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return None
    return pixmap.scaled(
        rect.size(),
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )


def _center_crop_pixmap(source: QPixmap, rect: QRect) -> QPixmap:
    if source.width() < rect.width() or source.height() < rect.height():
        return source
    x = max(0, (source.width() - rect.width()) // 2)
    y = max(0, (source.height() - rect.height()) // 2)
    cropped = source.copy(x, y, rect.width(), rect.height())
    return cropped if not cropped.isNull() else source


@dataclass(frozen=True)
class ComposedWallpaper:
    """Wallpaper layer only (no overlay scrim)."""

    pixmap: QPixmap | None = None
    fill_color: str | None = None


class SurfaceFillCompositor:
    """Build wallpaper imagery for a target rectangle."""

    def compose_wallpaper(
        self,
        wallpaper: Wallpaper,
        rect: QRect,
        *,
        theme: ResolvedTheme,
    ) -> ComposedWallpaper:
        if rect.width() <= 0 or rect.height() <= 0:
            return ComposedWallpaper(fill_color=theme.background)

        if isinstance(wallpaper, WallpaperNone):
            return ComposedWallpaper(fill_color=theme.background)

        if isinstance(wallpaper, WallpaperSolid):
            return ComposedWallpaper(fill_color=wallpaper.color)

        if isinstance(wallpaper, WallpaperGradient):
            pixmap = self._compose_gradient(wallpaper, rect)
            return ComposedWallpaper(pixmap=pixmap)

        if isinstance(wallpaper, WallpaperImage):
            pixmap = self._compose_image(wallpaper, rect)
            if pixmap is None:
                return ComposedWallpaper(fill_color=theme.background)
            return ComposedWallpaper(pixmap=pixmap)

        return ComposedWallpaper(fill_color=theme.background)

    def _compose_gradient(self, wallpaper: WallpaperGradient, rect: QRect) -> QPixmap:
        pixmap = QPixmap(rect.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            gradient = QLinearGradient()
            self._configure_gradient_endpoints(gradient, wallpaper.direction, rect)
            for stop in wallpaper.stops:
                gradient.setColorAt(
                    float(stop.position),
                    theme_qcolor(stop.color),
                )
            painter.fillRect(pixmap.rect(), gradient)
        finally:
            painter.end()
        return pixmap

    def _configure_gradient_endpoints(
        self,
        gradient: QLinearGradient,
        direction: str,
        rect: QRect,
    ) -> None:
        if direction == "vertical":
            gradient.setStart(QPointF(rect.left(), rect.top()))
            gradient.setFinalStop(QPointF(rect.left(), rect.bottom()))
            return
        if direction == "horizontal":
            gradient.setStart(QPointF(rect.left(), rect.top()))
            gradient.setFinalStop(QPointF(rect.right(), rect.top()))
            return
        if direction == "diagonal_down":
            gradient.setStart(QPointF(rect.left(), rect.top()))
            gradient.setFinalStop(QPointF(rect.right(), rect.bottom()))
            return
        if direction == "diagonal_up":
            gradient.setStart(QPointF(rect.left(), rect.bottom()))
            gradient.setFinalStop(QPointF(rect.right(), rect.top()))
            return
        gradient.setStart(QPointF(rect.left(), rect.top()))
        gradient.setFinalStop(QPointF(rect.left(), rect.bottom()))

    def _compose_image(self, wallpaper: WallpaperImage, rect: QRect) -> QPixmap | None:
        path = resolve_wallpaper_image_path(wallpaper.source)
        if path is None or not path.is_file():
            return None
        cache_key = cache_key_for_path(path, rect.width(), rect.height())
        if cache_key is not None:
            cached = get_cached_image(cache_key)
            if cached is not None:
                return cached
        source = _load_scaled_image_pixmap(path, rect)
        if source is None or source.isNull():
            return None
        composed = _center_crop_pixmap(source, rect)
        if cache_key is not None and not composed.isNull():
            store_cached_image(cache_key, composed)
        return composed

    @staticmethod
    def fill_rect_color(painter: QPainter, rect: QRect, color: str) -> None:
        painter.fillRect(rect, theme_qcolor(color))

    @staticmethod
    def apply_saturation_scale(pixmap: QPixmap, scale: float) -> QPixmap:
        """Desaturate a composed wallpaper pixmap (1.0 = unchanged)."""
        factor = max(0.0, min(1.0, float(scale)))
        if factor >= 0.999 or pixmap.isNull():
            return pixmap

        cache_key = (int(pixmap.cacheKey()), round(factor, 3))
        cached = _SATURATION_CACHE.get(cache_key)
        if cached is not None and not cached.isNull():
            _SATURATION_CACHE.move_to_end(cache_key)
            return cached

        image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        if image.isNull():
            return pixmap
        # PyQt6 does not expose QImage.isDetached(); copy ensures an exclusive buffer
        # before in-place edits via bits().
        image = image.copy()

        SurfaceFillCompositor._desaturate_argb32_image(image, factor)
        result = QPixmap.fromImage(image)
        _SATURATION_CACHE[cache_key] = result
        _SATURATION_CACHE.move_to_end(cache_key)
        while len(_SATURATION_CACHE) > _SATURATION_CACHE_MAX:
            _SATURATION_CACHE.popitem(last=False)
        return result

    @staticmethod
    def _desaturate_argb32_image(image: QImage, factor: float) -> None:
        """In-place BGRA saturation scale using raw buffer access (much faster than QColor loops)."""
        width = image.width()
        height = image.height()
        if width <= 0 or height <= 0:
            return

        bytes_per_line = image.bytesPerLine()
        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        row = memoryview(ptr)

        f = float(factor)
        for y in range(height):
            base = y * bytes_per_line
            for x in range(width):
                off = base + x * 4
                b = row[off]
                g = row[off + 1]
                r = row[off + 2]
                a = row[off + 3]
                if a == 0:
                    continue
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                row[off] = int(round(gray + (b - gray) * f))
                row[off + 1] = int(round(gray + (g - gray) * f))
                row[off + 2] = int(round(gray + (r - gray) * f))

    @staticmethod
    def tint_fill_color(color: str, *, saturation_scale: float) -> str:
        return adjust_saturation(color, saturation_scale)
