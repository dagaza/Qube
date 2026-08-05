"""Pin / ellipsis action affordances for sidebar entry rows."""

from __future__ import annotations

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QWidget,
)


def _qt_alive(obj: QObject | None) -> bool:
    """True when the Python wrapper still refers to a live C++ QObject."""
    return obj is not None and not sip.isdeleted(obj)


_SIDEBAR_OPTIONS_MENU_PROPERTY = "sidebar_options_menu"


def store_sidebar_options_menu(button: QPushButton, menu: QMenu) -> None:
    """Keep menu reference so pin-indicator mode can detach and restore it."""
    button.setProperty(_SIDEBAR_OPTIONS_MENU_PROPERTY, menu)


def sidebar_options_menu(button: QPushButton) -> QMenu | None:
    menu = button.property(_SIDEBAR_OPTIONS_MENU_PROPERTY)
    return menu if isinstance(menu, QMenu) else None


def sidebar_list_hovered_item(list_widget: QListWidget | None) -> QListWidgetItem | None:
    if not _qt_alive(list_widget):
        return None
    hovered = getattr(list_widget, "_sidebar_hovered_item", None)
    return hovered if isinstance(hovered, QListWidgetItem) else None


def create_sidebar_pin_indicator(parent: QWidget) -> QLabel:
    """Non-interactive pin badge shown on pinned rows at rest."""
    lbl = QLabel(parent)
    lbl.setObjectName("HistoryPinIndicator")
    lbl.setFixedSize(28, 28)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("background: transparent; border: none;")
    lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    lbl.hide()
    return lbl


class _SidebarListViewportFilter(QObject):
    """Clear hover when the pointer leaves the list viewport."""

    def __init__(self, list_widget: QListWidget, on_change) -> None:
        super().__init__(list_widget)
        self._list_widget = list_widget
        self._on_change = on_change

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        # Viewport Leave can still be delivered while the list is being destroyed
        # on app quit; touching a deleted QListWidget raises and can segfault.
        if not _qt_alive(self._list_widget):
            return False
        viewport = self._list_widget.viewport()
        if not _qt_alive(viewport):
            return False
        if (
            obj is viewport
            and event.type() == QEvent.Type.Leave
            and sidebar_list_hovered_item(self._list_widget) is not None
        ):
            self._list_widget._sidebar_hovered_item = None
            self._on_change()
        return super().eventFilter(obj, event)


class _SidebarRowInteractionFilter(QObject):
    """Track hover/press on ``setItemWidget`` rows where list signals miss child widgets."""

    def __init__(
        self,
        list_widget: QListWidget,
        item: QListWidgetItem,
        row_widget: QWidget,
        on_change,
    ) -> None:
        super().__init__(row_widget)
        self._list_widget = list_widget
        self._item = item
        self._row_widget = row_widget
        self._on_change = on_change

    def _options_button(self) -> QPushButton | None:
        btn = self._row_widget.findChild(QPushButton, "HistoryOptionsBtn")
        return btn if isinstance(btn, QPushButton) else None

    def _should_defer_to_options_button(self, obj: QObject) -> bool:
        opts_btn = self._options_button()
        return (
            opts_btn is not None
            and obj is opts_btn
            and opts_btn.isVisible()
            and opts_btn.isEnabled()
        )

    def _activate_row(self, *, select: bool = False) -> None:
        if not _qt_alive(self._list_widget) or not _qt_alive(self._row_widget):
            return
        if select:
            self._list_widget.setCurrentItem(self._item)
        hovered = sidebar_list_hovered_item(self._list_widget)
        if hovered is not self._item or select:
            self._list_widget._sidebar_hovered_item = self._item
            self._on_change()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if not _qt_alive(self._list_widget) or not _qt_alive(self._row_widget):
            return False
        et = event.type()
        if et == QEvent.Type.MouseMove:
            self._activate_row()
        elif (
            et == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and not self._should_defer_to_options_button(obj)
        ):
            self._activate_row(select=True)
        return False


def register_sidebar_entry_row(
    list_widget: QListWidget,
    item: QListWidgetItem,
    row_widget: QWidget,
    on_change,
) -> None:
    """Install row-level tracking so pin ↔ ellipsis swaps work over child widgets."""
    row_filter = _SidebarRowInteractionFilter(
        list_widget, item, row_widget, on_change
    )
    tracked = [row_widget, *row_widget.findChildren(QWidget)]
    for widget in tracked:
        widget.installEventFilter(row_filter)
        widget.setMouseTracking(True)
        if widget.objectName() in ("HistoryRowTitle", "HistoryPinIndicator", "ProGemBadge"):
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    row_widget._sidebar_row_interaction_filter = row_filter


def install_sidebar_list_hover_tracking(
    list_widget: QListWidget,
    on_hover_changed,
) -> None:
    """Track hovered/pressed rows reliably for pin ↔ ellipsis swapping."""
    list_widget._sidebar_hovered_item = None
    list_widget.setMouseTracking(True)
    list_widget.viewport().setMouseTracking(True)

    viewport_filter = _SidebarListViewportFilter(list_widget, on_hover_changed)
    list_widget.viewport().installEventFilter(viewport_filter)
    list_widget._sidebar_viewport_filter = viewport_filter
