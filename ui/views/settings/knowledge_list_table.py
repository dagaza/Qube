"""Bordered list tables for Knowledge → Custom sources / My knowledge."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHeaderView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
)

from core.theme.accessors import theme_for
from core.theme.widget_styles import SETTINGS_BORDERED_TABLE

_ROW_HEIGHT_PX = 34
_HEADER_HEIGHT_PX = 32
_MIN_PLACEHOLDER_BODY_HEIGHT_PX = 72
_FRAME_INSET_PX = 4


def configure_borderless_list_table(
    table: QTableWidget,
    *,
    columns: tuple[str, ...],
    object_name: str,
) -> None:
    table.setObjectName(object_name)
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    header = table.horizontalHeader()
    for col in range(len(columns)):
        header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
    header.setStretchLastSection(True)
    table.horizontalHeader().setDefaultAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    table.horizontalHeader().setFixedHeight(_HEADER_HEIGHT_PX)
    table.horizontalHeader().setHighlightSections(False)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(_ROW_HEIGHT_PX)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    table.setShowGrid(False)
    table.setAlternatingRowColors(False)
    table.setWordWrap(True)
    table.setFrameShape(QFrame.Shape.NoFrame)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setMinimumWidth(0)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    table.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def apply_borderless_list_table_theme(table: QTableWidget, *, is_dark: bool) -> None:
    """Apply bordered panel chrome so empty lists read as table containers."""
    theme = theme_for(is_dark=is_dark)
    table.setStyleSheet(
        theme.style(SETTINGS_BORDERED_TABLE, object_name=table.objectName())
    )


def set_table_placeholder_row(
    table: QTableWidget,
    *,
    text: str,
    is_dark: bool,
) -> None:
    theme = theme_for(is_dark=is_dark)
    table.setRowCount(1)
    table.clearSpans()
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.NoItemFlags)
    item.setForeground(theme.qcolor(theme.text_muted))
    item.setTextAlignment(
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
    )
    table.setItem(0, 0, item)
    if table.columnCount() > 1:
        table.setSpan(0, 0, 1, table.columnCount())
    _sync_table_height(table, placeholder=True)


def _sync_table_height(table: QTableWidget, *, placeholder: bool = False) -> None:
    row_count = max(1, table.rowCount())
    table.resizeRowsToContents()
    content_height = sum(table.rowHeight(row) for row in range(row_count))
    if placeholder:
        content_height = max(content_height, _MIN_PLACEHOLDER_BODY_HEIGHT_PX)
    table.setMinimumHeight(_HEADER_HEIGHT_PX + content_height + _FRAME_INSET_PX)


def populate_table_rows(
    table: QTableWidget,
    *,
    rows: list[tuple[str, ...]],
    placeholder: str,
    is_dark: bool,
) -> None:
    apply_borderless_list_table_theme(table, is_dark=is_dark)
    table.clearSpans()
    if not rows:
        set_table_placeholder_row(table, text=placeholder, is_dark=is_dark)
        return

    table.setRowCount(len(rows))
    for row_idx, cells in enumerate(rows):
        for col_idx, value in enumerate(cells):
            table.setItem(row_idx, col_idx, QTableWidgetItem(value))
    _sync_table_height(table, placeholder=False)


def selected_data_row(table: QTableWidget) -> int | None:
    """Return selected row index, or None if selection is empty or a placeholder row."""
    row = table.currentRow()
    if row < 0:
        return None
    first = table.item(row, 0)
    if first is None or not (first.flags() & Qt.ItemFlag.ItemIsSelectable):
        return None
    return row
