"""Shared ghost icon buttons (sidebar headers, utility toolbars)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import GHOST_ICON_BUTTON


def ghost_icon_button_stylesheet(
    theme: ResolvedTheme,
    *,
    hide_menu_indicator: bool = False,
    padding: str = "6px",
) -> str:
    qss = theme.style(GHOST_ICON_BUTTON, padding=padding)
    if hide_menu_indicator:
        qss += """
            QPushButton::menu-indicator { image: none; width: 0px; }
        """
    return qss


def apply_ghost_icon_button_style(
    button: QPushButton | None,
    theme: ResolvedTheme,
    *,
    hide_menu_indicator: bool = False,
    fixed_size: int | None = 28,
    padding: str = "6px",
) -> None:
    """Apply theme-aware hover/press chrome; required on Linux for QSS hover fills."""
    if button is None:
        return
    button.setProperty("class", "IconButton")
    button.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    if fixed_size is not None and fixed_size > 0:
        button.setFixedSize(fixed_size, fixed_size)
    button.setStyleSheet(
        ghost_icon_button_stylesheet(
            theme,
            hide_menu_indicator=hide_menu_indicator,
            padding=padding,
        )
    )
    style = button.style()
    if style is not None:
        style.unpolish(button)
        style.polish(button)
    button.update()
