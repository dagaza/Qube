"""Derived overlay/scrim colors for modal and tour dim layers."""

from __future__ import annotations

from PyQt6.QtGui import QColor

from core.theme.accessors import theme_for
from core.theme.color_utils import parse_color
from core.theme.tokens import ResolvedTheme


def overlay_scrim_qcolor(
    theme: ResolvedTheme | None = None,
    *,
    is_dark: bool = True,
) -> QColor:
    """Return a dim scrim color for fullscreen modal/tour overlays."""
    resolved = theme if theme is not None else theme_for(is_dark=is_dark)
    if resolved.is_dark:
        return QColor(0, 0, 0, 175)
    rgba = parse_color(resolved.text_primary)
    return QColor(rgba.r, rgba.g, rgba.b, 110)
