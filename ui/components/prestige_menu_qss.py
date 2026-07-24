"""Shared Prestige styling for sidebar kebab / context QMenus."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QMenu

from core.theme.accessors import theme_for
from core.theme.widget_styles import settings_prestige_menu_palette

# Item padding: left reserves icon column; right stays tight vs legacy 25px.
_KEBAB_MENU_ITEM_PADDING = "8px 12px 8px 12px"
_KEBAB_MENU_ITEM_MARGIN = "0px 2px"
# Nudge icons inward so they sit inside the rounded hover pill, not on the menu edge.
_KEBAB_MENU_ICON_LEFT = "8px"
_KEBAB_MENU_CONTAINER_PADDING = "6px"


def apply_prestige_kebab_menu_theme(menu: QMenu, is_dark: bool) -> None:
    """Apply rounded Prestige colors with compact icon+label spacing."""
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    theme = theme_for(is_dark=is_dark)
    colors = settings_prestige_menu_palette(theme)
    bg = colors["bg"]
    fg = colors["fg"]
    sel_bg = colors["sel_bg"]
    sel_fg = colors["sel_fg"]
    border = colors["border"]
    hover = colors["hover"]

    palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
        palette.setColor(role, QColor(bg))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(fg))
    palette.setColor(QPalette.ColorRole.Text, QColor(fg))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(sel_bg))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(sel_fg))
    menu.setPalette(palette)

    menu.setStyleSheet(
        f"""
            QMenu {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: {_KEBAB_MENU_CONTAINER_PADDING};
            }}
            QMenu::item {{
                background-color: transparent;
                padding: {_KEBAB_MENU_ITEM_PADDING};
                margin: {_KEBAB_MENU_ITEM_MARGIN};
                border-radius: 8px;
            }}
            QMenu::icon {{
                left: {_KEBAB_MENU_ICON_LEFT};
            }}
            QMenu::item:selected {{
                background-color: {hover};
                color: {sel_fg};
            }}
            QMenu::right-arrow {{
                width: 12px;
                height: 12px;
                margin-right: 4px;
            }}
        """
    )
