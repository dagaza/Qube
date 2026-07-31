"""Theme resolution and widget polish helpers for Settings primitives."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from core.theme.accessors import theme_for
from core.theme.view_theme import view_resolved_theme


def repolish_widget(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(False)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def coalesce_settings_is_dark(host, *, is_dark: bool | None = None) -> bool:
    """Return the active settings theme, preferring MainWindow over stale cache."""
    window = host.window() if hasattr(host, "window") else None
    if window is not None and hasattr(window, "_is_dark_theme"):
        resolved = bool(window._is_dark_theme)
    elif is_dark is not None:
        resolved = bool(is_dark)
    else:
        resolved = bool(getattr(host, "_settings_ui_is_dark", True))
    host._settings_ui_is_dark = resolved
    return resolved


def resolve_settings_is_dark(host) -> bool:
    """Backward-compatible alias — always syncs from the window when possible."""
    return coalesce_settings_is_dark(host)


def settings_theme(*, is_dark: bool, host=None):
    if host is not None:
        return view_resolved_theme(host, is_dark=is_dark)
    return theme_for(is_dark=is_dark)
