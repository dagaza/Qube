"""Widget-level Prestige styling for the Source status table."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

_TABLE_OBJECT_NAME = "KnowledgeProviderStatusTable"

_STATUS_FOREGROUND: dict[str, dict[str, str]] = {
    "Connected": {"dark": "#a6e3a1", "light": "#15803d"},
    "Not configured": {"dark": "#f38ba8", "light": "#be123c"},
    "Env override": {"dark": "#89b4fa", "light": "#1d4ed8"},
    "Not available": {"dark": "#6c7086", "light": "#94a3b8"},
    "Anonymous": {"dark": "#a6adc8", "light": "#64748b"},
}

_HEALTH_FOREGROUND: dict[str, dict[str, str]] = {
    "Good": {"dark": "#a6e3a1", "light": "#15803d"},
    "Degraded": {"dark": "#f9e2af", "light": "#b45309"},
    "Unknown": {"dark": "#a6adc8", "light": "#64748b"},
}


def provider_status_table_stylesheet(*, is_dark: bool) -> str:
    """Return a widget-level stylesheet for the provider status table."""
    if is_dark:
        return f"""
        QTableWidget#{_TABLE_OBJECT_NAME} {{
            background-color: transparent;
            border: none;
            gridline-color: transparent;
            outline: none;
            color: #cdd6f4;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QAbstractScrollArea::viewport {{
            background-color: transparent;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME}::item {{
            color: #cdd6f4;
            border: none;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 400;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QHeaderView::section {{
            background-color: rgba(49, 50, 68, 0.45);
            color: #a6adc8;
            padding: 6px 10px;
            border: none;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 11px;
            font-weight: 600;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QTableCornerButton::section {{
            background-color: rgba(49, 50, 68, 0.45);
            border: none;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 4px 2px 4px 0;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar::handle:vertical {{
            background: rgba(166, 173, 200, 0.35);
            border-radius: 4px;
            min-height: 24px;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar::add-line:vertical,
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """

    return f"""
        QTableWidget#{_TABLE_OBJECT_NAME} {{
            background-color: transparent;
            border: none;
            gridline-color: transparent;
            outline: none;
            color: #334155;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QAbstractScrollArea::viewport {{
            background-color: transparent;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME}::item {{
            color: #334155;
            border: none;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 400;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QHeaderView::section {{
            background-color: rgba(248, 250, 252, 0.95);
            color: #64748b;
            padding: 6px 10px;
            border: none;
            border-bottom: 1px solid #e2e8f0;
            font-size: 11px;
            font-weight: 600;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QTableCornerButton::section {{
            background-color: rgba(248, 250, 252, 0.95);
            border: none;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 4px 2px 4px 0;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar::handle:vertical {{
            background: rgba(100, 116, 139, 0.35);
            border-radius: 4px;
            min-height: 24px;
        }}
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar::add-line:vertical,
        QTableWidget#{_TABLE_OBJECT_NAME} QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """


def _row_background_color(row_idx: int, *, is_dark: bool) -> QColor:
    if row_idx % 2 == 0:
        return QColor(0, 0, 0, 0)
    if is_dark:
        return QColor(69, 71, 90, 71)
    return QColor(226, 232, 240, 166)


def _default_text_color(*, is_dark: bool) -> QColor:
    return QColor("#cdd6f4" if is_dark else "#334155")


def _status_text_color(status: str, *, is_dark: bool) -> QColor:
    theme_key = "dark" if is_dark else "light"
    hex_color = _STATUS_FOREGROUND.get(status, {}).get(theme_key)
    if hex_color is None:
        return _default_text_color(is_dark=is_dark)
    return QColor(hex_color)


def _health_text_color(health: str, *, is_dark: bool) -> QColor:
    theme_key = "dark" if is_dark else "light"
    hex_color = _HEALTH_FOREGROUND.get(health)
    if hex_color is None:
        return _default_text_color(is_dark=is_dark)
    return QColor(hex_color[theme_key])


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
    row_bg = QBrush(_row_background_color(row_idx, is_dark=is_dark))
    default_fg = _default_text_color(is_dark=is_dark)

    for item in (provider_item, status_item, quota_item, health_item):
        item.setBackground(row_bg)

    provider_item.setForeground(default_fg)
    quota_item.setForeground(default_fg)
    status_item.setForeground(_status_text_color(status_text, is_dark=is_dark))
    health_item.setForeground(_health_text_color(health_text, is_dark=is_dark))
