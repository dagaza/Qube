"""Sidebar list rows that use QListWidget.setItemWidget.

Qt stylesheets often do **not** match ``#List::item … QLabel`` for widgets installed via
``setItemWidget`` (the label is not a style child of ``::item`` in practice). Row title
**colors** and row action **icons** (chevron, ellipsis) are applied exclusively here
(see ``assets/styles/*.qss`` for typography only).
Call ``apply_sidebar_row_title_colors`` and ``apply_sidebar_row_action_icons`` from
``_update_row_colors()`` and after theme toggles.
"""

from __future__ import annotations


from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QListWidget, QPushButton, QWidget

from core.theme.accessors import theme_for
from core.theme.tokens import ResolvedTheme
from core.theme.view_theme import view_resolved_theme
from core.theme.svg_icons import themed_fa_icon, themed_fa_pixmap
from ui.components.sidebar_entry_actions import (
    sidebar_list_hovered_item,
    sidebar_options_menu,
)
from ui.components.sidebar_folder_list import (
    ROW_KIND_DOCUMENT,
    ROW_KIND_FOLDER,
    ROW_KIND_SESSION,
    SIDEBAR_ROW_PAYLOAD_ROLE,
    row_kind,
)
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


def _resolve_sidebar_theme(
    widget: QWidget | None,
    *,
    is_dark: bool | None = None,
    theme: ResolvedTheme | None = None,
) -> ResolvedTheme:
    if theme is not None:
        return theme
    if widget is not None:
        return view_resolved_theme(widget, is_dark=is_dark)
    return theme_for(is_dark=is_dark if is_dark is not None else True)


def _sidebar_row_icon_highlighted(
    item,
    *,
    active_folder_key: str | None,
    list_widget: QListWidget | None = None,
) -> bool:
    if item.isSelected():
        return True
    hovered = sidebar_list_hovered_item(list_widget)
    if hovered is item:
        return True
    if (
        active_folder_key is not None
        and row_kind(item) == ROW_KIND_FOLDER
        and str(item.data(Qt.ItemDataRole.UserRole) or "") == active_folder_key
    ):
        return True
    return False


def _sidebar_entry_is_pinned(item) -> bool:
    if row_kind(item) not in (ROW_KIND_SESSION, ROW_KIND_DOCUMENT):
        return False
    payload = item.data(SIDEBAR_ROW_PAYLOAD_ROLE)
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("is_pinned"))


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
    resolved = _resolve_sidebar_theme(list_widget, is_dark=is_dark, theme=theme)
    active_folder_key = str(active_folder_id) if active_folder_id else None

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        row = list_widget.itemWidget(item)
        if row is None:
            continue
        highlighted = _sidebar_row_icon_highlighted(
            item,
            active_folder_key=active_folder_key,
            list_widget=list_widget,
        )
        icon_color = sidebar_row_action_icon_color(resolved, highlighted=highlighted)

        chevron_btn = row.findChild(QPushButton, "HistoryFolderChevronBtn")
        if chevron_btn is not None:
            icon_name = chevron_btn.property(_CHEVRON_ICON_PROPERTY) or "fa5s.chevron-down"
            chevron_btn.setIcon(themed_fa_icon(str(icon_name), icon_color, 12))
            chevron_btn.setIconSize(QSize(12, 12))

        opts_btn = row.findChild(QPushButton, "HistoryOptionsBtn")
        if opts_btn is not None:
            pinned = _sidebar_entry_is_pinned(item)
            show_menu = (not pinned) or highlighted
            menu = sidebar_options_menu(opts_btn)
            pin_lbl = row.findChild(QLabel, "HistoryPinIndicator")

            if pin_lbl is not None:
                if pinned and not show_menu:
                    pin_lbl.setPixmap(
                        themed_fa_pixmap("fa5s.thumbtack", icon_color, 14)
                    )
                    pin_lbl.setVisible(True)
                    opts_btn.setVisible(False)
                else:
                    pin_lbl.setVisible(False)
                    opts_btn.setVisible(True)

            if show_menu:
                if menu is not None and opts_btn.menu() is None:
                    opts_btn.setMenu(menu)
                opts_btn.setIcon(themed_fa_icon("fa5s.ellipsis-v", icon_color, 16))
                opts_btn.setEnabled(True)
            elif pin_lbl is None:
                if opts_btn.menu() is not None:
                    opts_btn.setMenu(None)
                opts_btn.setIcon(themed_fa_icon("fa5s.thumbtack", icon_color, 14))
                opts_btn.setEnabled(False)
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
    host = list_widget or sidebar_frame
    resolved = _resolve_sidebar_theme(host, is_dark=is_dark, theme=theme)
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
    resolved = _resolve_sidebar_theme(list_widget, is_dark=is_dark, theme=theme)
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
