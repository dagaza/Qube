"""Sidebar list rows that use QListWidget.setItemWidget.

Qt stylesheets often do **not** match ``#List::item … QLabel`` for widgets installed via
``setItemWidget`` (the label is not a style child of ``::item`` in practice). Row title
colors are applied here so dark/light + selection stay correct; keep typography in sync
with ``assets/styles/base.qss`` / ``light.qss`` for non-color properties.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QListWidget

_ROW_TITLE_STYLE = (
    "background: transparent; border: none; "
    "font-size: 13px; font-weight: 500; color: {color};"
)


def apply_sidebar_row_title_colors(
    list_widget: QListWidget | None,
    *,
    is_dark: bool,
    label_object_name: str = "HistoryRowTitle",
) -> None:
    """Set HistoryRowTitle label color from selection + theme (reliable with setItemWidget)."""
    if list_widget is None:
        return
    if is_dark:
        normal = "#cdd6f4"
        selected = "#ffffff"
    else:
        normal = "#1e293b"
        selected = "#1e293b"

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        row = list_widget.itemWidget(item)
        if row is None:
            continue
        lbl = row.findChild(QLabel, label_object_name)
        if lbl is None:
            continue
        color = selected if item.isSelected() else normal
        lbl.setStyleSheet(_ROW_TITLE_STYLE.format(color=color))
