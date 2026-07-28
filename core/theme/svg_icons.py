"""Theme-aware icon tinting: SVG assets and Font Awesome (qtawesome) glyphs."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QColor, QPainter, QPixmap

from core.theme.color_utils import parse_color, qtawesome_color


def themed_fa_pixmap(icon_name: str, color: str, size: int = 16) -> QPixmap:
    """Return a pixmap tinted to ``color`` (supports rgba theme tokens)."""
    return qta.icon(icon_name, color=qtawesome_color(color)).pixmap(QSize(size, size))


def themed_fa_icon(
    icon_name: str,
    color: str,
    size: int = 16,
    *,
    disabled_color: str | None = None,
) -> QIcon:
    """Bake a Font Awesome icon so QSS/platform styles cannot retint it to black."""
    icon = QIcon()
    normal = themed_fa_pixmap(icon_name, color, size=size)
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
        icon.addPixmap(normal, mode, QIcon.State.Off)
    if disabled_color is not None:
        disabled = themed_fa_pixmap(icon_name, disabled_color, size=size)
        icon.addPixmap(disabled, QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def tinted_svg_pixmap(svg_path: str, color_hex: str, size: int = 18) -> QPixmap:
    """Render an SVG and recolor it to ``color_hex``."""
    from PyQt6.QtSvg import QSvgRenderer

    target_size = QSize(size, size)
    renderer = QSvgRenderer(str(svg_path))
    if renderer.isValid():
        pixmap = QPixmap(target_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
    else:
        pixmap = QPixmap(str(svg_path))
        if pixmap.isNull():
            return QPixmap(target_size)
        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    tinted = QPixmap(pixmap.size())
    tinted.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    rgba = parse_color(color_hex)
    painter.fillRect(tinted.rect(), QColor(rgba.r, rgba.g, rgba.b, rgba.a))
    painter.end()
    return tinted


def tinted_svg_icon(svg_path: str, color_hex: str, size: int = 18) -> QIcon:
    """Return a baked ``QIcon`` tinted to ``color_hex`` (never an untinted SVG fallback)."""
    pixmap = tinted_svg_pixmap(svg_path, color_hex, size=size)
    if pixmap.isNull():
        return QIcon(str(svg_path))
    icon = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
        icon.addPixmap(pixmap, mode, QIcon.State.Off)
    return icon
