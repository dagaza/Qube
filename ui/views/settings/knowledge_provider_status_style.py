"""Widget-level Prestige styling for the Source status table."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import (
    MUTED_STATUS,
    PROVIDER_STATUS_TABLE,
    SUCCESS_STATUS,
    WARNING_STATUS,
)

_TABLE_OBJECT_NAME = "KnowledgeProviderStatusTable"

_STATUS_ROLES = {
    "Connected": SUCCESS_STATUS,
    "Not configured": "error",
    "Env override": "link",
    "Not available": MUTED_STATUS,
    "Anonymous": MUTED_STATUS,
}

_HEALTH_ROLES = {
    "Good": SUCCESS_STATUS,
    "Degraded": WARNING_STATUS,
    "Unknown": MUTED_STATUS,
}


def _theme(*, is_dark: bool):
    return theme_for(is_dark=is_dark)


def provider_status_table_stylesheet(*, is_dark: bool) -> str:
    """Return a widget-level stylesheet for the provider status table."""
    return _theme(is_dark=is_dark).style(
        PROVIDER_STATUS_TABLE, object_name=_TABLE_OBJECT_NAME
    )


def _row_background_color(row_idx: int, *, is_dark: bool, theme) -> QColor:
    if row_idx % 2 == 0:
        return QColor(0, 0, 0, 0)
    if is_dark:
        return theme.qcolor(with_alpha(theme.surface_elevated, 0.28))
    return theme.qcolor(with_alpha(theme.border, 0.65))


def _default_text_color(*, theme) -> QColor:
    return theme.qcolor(theme.text_primary)


def _status_text_color(status: str, *, theme) -> QColor:
    role = _STATUS_ROLES.get(status, MUTED_STATUS)
    if role == "error":
        return theme.qcolor(theme.error)
    if role == "link":
        return theme.qcolor(theme.link)
    return theme.qcolor_role(role)


def _health_text_color(health: str, *, theme) -> QColor:
    role = _HEALTH_ROLES.get(health, MUTED_STATUS)
    return theme.qcolor_role(role)


def apply_provider_status_table_theme(table: QTableWidget, *, is_dark: bool) -> None:
    """Apply Prestige table chrome (header, scrollbar, base typography)."""
    table.setObjectName(_TABLE_OBJECT_NAME)
    table.setStyleSheet(provider_status_table_stylesheet(is_dark=is_dark))


def apply_provider_status_row_style(
    *,
    row_idx: int,
    provider_item: QTableWidgetItem,
    status_item: QTableWidgetItem,
    quota_item: QTableWidgetItem,
    health_item: QTableWidgetItem,
    status_text: str,
    health_text: str,
    is_dark: bool,
) -> None:
    """Paint row backgrounds and per-column foreground colors."""
    theme = _theme(is_dark=is_dark)
    row_bg = QBrush(_row_background_color(row_idx, is_dark=is_dark, theme=theme))
    default_fg = _default_text_color(theme=theme)

    for item in (provider_item, status_item, quota_item, health_item):
        item.setBackground(row_bg)

    provider_item.setForeground(default_fg)
    quota_item.setForeground(default_fg)
    status_item.setForeground(_status_text_color(status_text, theme=theme))
    health_item.setForeground(_health_text_color(health_text, theme=theme))
