"""Sidebar list rows that use QListWidget.setItemWidget.

Qt stylesheets often do **not** match ``#List::item … QLabel`` for widgets installed via
``setItemWidget`` (the label is not a style child of ``::item`` in practice). Row title
**colors** and row action **icons** (chevron, ellipsis) are applied exclusively here
(see ``assets/styles/*.qss`` for typography only).
Call ``apply_sidebar_row_title_colors`` and ``apply_sidebar_row_action_icons`` from
``_update_row_colors()`` and after theme toggles.
"""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QListWidget, QPushButton, QWidget

from core.theme.accessors import theme_for
from core.theme.tokens import ResolvedTheme
from ui.components.sidebar_folder_list import ROW_KIND_FOLDER, row_kind
from ui.shell_theme import sidebar_row_action_icon_color

_ROW_TITLE_STYLE = (
    "background: transparent; border: none; "
    "font-size: 13px; font-weight: 500; color: {color};"
)

_FOLDER_TITLE_STYLE = (
    "background: transparent; border: none; "
    "font-size: 13px; font-weight: 700; color: {color};"
)

_CHEVRON_ICON_PROPERTY = "sidebar_chevron_icon"


def _sidebar_row_icon_highlighted(
    item,
    *,
    active_folder_key: str | None,
) -> bool:
    if item.isSelected():
        return True
    if (
        active_folder_key is not None
        and row_kind(item) == ROW_KIND_FOLDER
        and str(item.data(Qt.ItemDataRole.UserRole) or "") == active_folder_key
    ):
        return True
    return False


def apply_sidebar_row_action_icons(
    list_widget: QListWidget | None,
    *,
    is_dark: bool | None = None,
    theme: ResolvedTheme | None = None,
    active_folder_id: str | None = None,
) -> None:
    """Refresh chevron and ellipsis qtawesome icons (QSS cannot retint them)."""
    if list_widget is None:
        return
    resolved = theme_for(is_dark=is_dark if is_dark is not None else True, resolved=theme)
    active_folder_key = str(active_folder_id) if active_folder_id else None

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        row = list_widget.itemWidget(item)
        if row is None:
            continue
        highlighted = _sidebar_row_icon_highlighted(
            item,
            active_folder_key=active_folder_key,
        )
        icon_color = sidebar_row_action_icon_color(resolved, highlighted=highlighted)

        chevron_btn = row.findChild(QPushButton, "HistoryFolderChevronBtn")
        if chevron_btn is not None:
            icon_name = chevron_btn.property(_CHEVRON_ICON_PROPERTY) or "fa5s.chevron-down"
            chevron_btn.setIcon(qta.icon(str(icon_name), color=icon_color))
            chevron_btn.setIconSize(QSize(12, 12))

        opts_btn = row.findChild(QPushButton, "HistoryOptionsBtn")
        if opts_btn is not None:
            opts_btn.setIcon(qta.icon("fa5s.ellipsis-v", color=icon_color))
            opts_btn.setIconSize(QSize(16, 16))


def apply_sidebar_row_theme(
    list_widget: QListWidget | None,
    *,
    is_dark: bool | None = None,
    theme: ResolvedTheme | None = None,
    label_object_name: str = "HistoryRowTitle",
    active_folder_id: str | None = None,
) -> None:
    """Refresh row titles and action icons together."""
    apply_sidebar_row_title_colors(
        list_widget,
        is_dark=is_dark,
        theme=theme,
        label_object_name=label_object_name,
        active_folder_id=active_folder_id,
    )
    apply_sidebar_row_action_icons(
        list_widget,
        is_dark=is_dark,
        theme=theme,
        active_folder_id=active_folder_id,
    )


def _nav_list_sidebar_bg(theme: ResolvedTheme) -> QColor:
    return theme.qcolor(theme.sidebar_surface)


def _paint_widget_window_bg(widget: QWidget, bg: QColor) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, bg)
    widget.setPalette(palette)


def apply_nav_list_sidebar_surface(
    *,
    is_dark: bool | None = None,
    theme: ResolvedTheme | None = None,
    sidebar_frame: QWidget | None = None,
    list_widget: QListWidget | None = None,
) -> None:
    """Paint hub sidebar frame + list viewport only (Model Manager / Library pattern)."""
    resolved = theme_for(is_dark=is_dark if is_dark is not None else True, resolved=theme)
    bg = _nav_list_sidebar_bg(resolved)
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
    is_dark: bool | None = None,
    theme: ResolvedTheme | None = None,
    label_object_name: str = "HistoryRowTitle",
    active_folder_id: str | None = None,
) -> None:
    """Set row label colors from selection + theme (reliable with setItemWidget)."""
    if list_widget is None:
        return
    resolved = theme_for(is_dark=is_dark if is_dark is not None else True, resolved=theme)
    normal = resolved.text_primary
    selected = resolved.list_row_title_selected

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
