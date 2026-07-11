"""Borderless list tables for Knowledge → Custom sources / My knowledge."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHeaderView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
)

_PLACEHOLDER_FG_DARK = QColor("#a6adc8")
_PLACEHOLDER_FG_LIGHT = QColor("#64748b")
_ROW_HEIGHT_PX = 34
_HEADER_HEIGHT_PX = 32


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


def apply_borderless_list_table_theme(table: QTableWidget, *, is_dark: bool) -> None:
    if is_dark:
        table.setStyleSheet(
            f"""
            QTableWidget#{table.objectName()} {{
                background: transparent;
                border: none;
            }}
            QTableWidget#{table.objectName()}::item {{
                padding: 6px 4px;
                border: none;
            }}
            QTableWidget#{table.objectName()} QHeaderView::section {{
                background: transparent;
                border: none;
                padding: 4px;
                font-weight: 600;
            }}
            """
        )
    else:
        table.setStyleSheet(
            f"""
            QTableWidget#{table.objectName()} {{
                background: transparent;
                border: none;
            }}
            QTableWidget#{table.objectName()}::item {{
                padding: 6px 4px;
                border: none;
            }}
            QTableWidget#{table.objectName()} QHeaderView::section {{
                background: transparent;
                border: none;
                padding: 4px;
                font-weight: 600;
            }}
            """
        )


def set_table_placeholder_row(
    table: QTableWidget,
    *,
    text: str,
    is_dark: bool,
) -> None:
    table.setRowCount(1)
    table.clearSpans()
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.NoItemFlags)
    item.setForeground(_PLACEHOLDER_FG_DARK if is_dark else _PLACEHOLDER_FG_LIGHT)
    table.setItem(0, 0, item)
    if table.columnCount() > 1:
        table.setSpan(0, 0, 1, table.columnCount())
    _sync_table_height(table)


def _sync_table_height(table: QTableWidget) -> None:
    row_count = max(1, table.rowCount())
    table.resizeRowsToContents()
    content_height = sum(table.rowHeight(row) for row in range(row_count))
    table.setMinimumHeight(_HEADER_HEIGHT_PX + content_height + 4)


def populate_table_rows(
    table: QTableWidget,
    *,
    rows: list[tuple[str, ...]],
    placeholder: str,
    is_dark: bool,
) -> None:
    table.clearSpans()
    if not rows:
        set_table_placeholder_row(table, text=placeholder, is_dark=is_dark)
        return

    table.setRowCount(len(rows))
    for row_idx, cells in enumerate(rows):
        for col_idx, value in enumerate(cells):
            table.setItem(row_idx, col_idx, QTableWidgetItem(value))
    _sync_table_height(table)


def selected_data_row(table: QTableWidget) -> int | None:
    """Return selected row index, or None if selection is empty or a placeholder row."""
    row = table.currentRow()
    if row < 0:
        return None
    first = table.item(row, 0)
    if first is None or not (first.flags() & Qt.ItemFlag.ItemIsSelectable):
        return None
    return row
