"""Sidebar list rows that use QListWidget.setItemWidget.

Qt stylesheets often do **not** match ``#List::item … QLabel`` for widgets installed via
``setItemWidget`` (the label is not a style child of ``::item`` in practice). Row title
colors are applied here so dark/light + selection stay correct; keep typography in sync
with ``assets/styles/base.qss`` / ``light.qss`` for non-color properties.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QListWidget, QWidget

from ui.components.sidebar_folder_list import ROW_KIND_FOLDER, row_kind

_ROW_TITLE_STYLE = (
    "background: transparent; border: none; "
    "font-size: 13px; font-weight: 500; color: {color};"
)

_FOLDER_TITLE_STYLE = (
    "background: transparent; border: none; "
    "font-size: 13px; font-weight: 700; color: {color};"
)


def _nav_list_sidebar_bg(is_dark: bool) -> QColor:
    return QColor("#232337" if is_dark else "#E9EFF5")


def _paint_widget_window_bg(widget: QWidget, bg: QColor) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, bg)
    widget.setPalette(palette)


def apply_nav_list_sidebar_surface(
    *,
    is_dark: bool,
    sidebar_frame: QWidget | None = None,
    list_widget: QListWidget | None = None,
) -> None:
    """Paint hub sidebar frame + list viewport only (Model Manager / Library pattern)."""
    bg = _nav_list_sidebar_bg(is_dark)
    if sidebar_frame is not None:
        _paint_widget_window_bg(sidebar_frame, bg)
    if list_widget is None:
        return
    _paint_widget_window_bg(list_widget, bg)
    pal = list_widget.palette()
    pal.setColor(QPalette.ColorRole.Base, bg)
    list_widget.setPalette(pal)
    viewport = list_widget.viewport()
    if viewport is not None:
        viewport.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        viewport.setAutoFillBackground(True)
        vpal = viewport.palette()
        vpal.setColor(QPalette.ColorRole.Window, bg)
        vpal.setColor(QPalette.ColorRole.Base, bg)
        viewport.setPalette(vpal)


def apply_sidebar_row_title_colors(
    list_widget: QListWidget | None,
    *,
    is_dark: bool,
    label_object_name: str = "HistoryRowTitle",
    active_folder_id: str | None = None,
) -> None:
    """Set row label colors from selection + theme (reliable with setItemWidget).

    When ``active_folder_id`` is set (Library upload target), that folder row uses
    the selected title color even if a document row carries QListWidget selection.
    """
    if list_widget is None:
        return
    if is_dark:
        normal = "#cdd6f4"
        selected = "#ffffff"
    else:
        normal = "#1e293b"
        selected = "#1e293b"

    active_folder_key = str(active_folder_id) if active_folder_id else None

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        row = list_widget.itemWidget(item)
        if row is None:
            continue
        is_active_upload_folder = (
            active_folder_key is not None
            and row_kind(item) == ROW_KIND_FOLDER
            and str(item.data(Qt.ItemDataRole.UserRole) or "") == active_folder_key
        )
        for obj_name, template in (
            (label_object_name, _ROW_TITLE_STYLE),
            ("HistoryFolderTitle", _FOLDER_TITLE_STYLE),
        ):
            lbl = row.findChild(QLabel, obj_name)
            if lbl is None:
                continue
            if obj_name == "HistoryFolderTitle":
                lbl_color = (
                    selected
                    if (is_active_upload_folder or item.isSelected())
                    else normal
                )
            else:
                lbl_color = selected if item.isSelected() else normal
            lbl.setStyleSheet(template.format(color=lbl_color))
