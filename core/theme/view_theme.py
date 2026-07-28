"""Resolve the active ``ResolvedTheme`` from a widget hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.theme.accessors import theme_for

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

    from core.theme.tokens import ResolvedTheme


def view_resolved_theme(
    widget: QWidget | None,
    *,
    is_dark: bool | None = None,
) -> ResolvedTheme:
    """Prefer ``window().theme_manager.current``; fall back to built-in scheme."""
    win = widget.window() if widget is not None else None
    if win is not None and hasattr(win, "theme_manager"):
        return win.theme_manager.current
    if is_dark is None:
        is_dark = getattr(win, "_is_dark_theme", True) if win is not None else True
    return theme_for(is_dark=is_dark)
