"""Searchable theme picker for Settings → Themes (§14 Phase 4)."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.theme.catalog import ThemePickerEntry
from ui.components.selector_button import SelectorButton

_SWATCH_SIZE = 14
_SEARCH_DEBOUNCE_MS = 150


def _swatch_icon(color_hex: str) -> QIcon:
    pixmap = QPixmap(_SWATCH_SIZE, _SWATCH_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    fill = QColor(color_hex)
    if not fill.isValid():
        fill = QColor("#64748b")
    painter.setBrush(fill)
    painter.setPen(QColor(0, 0, 0, 40))
    painter.drawRoundedRect(0, 0, _SWATCH_SIZE - 1, _SWATCH_SIZE - 1, 3, 3)
    painter.end()
    return QIcon(pixmap)


class ThemePickerPopup(QFrame):
    """Popup list with search filter for choosing a theme preset."""

    schemeSelected = pyqtSignal(str)

    _POPUP_WIDTH = 360
    _POPUP_MAX_HEIGHT = 320

    def __init__(
        self,
        parent: QWidget | None,
        *,
        entries: tuple[ThemePickerEntry, ...],
        current_scheme_id: str,
        is_dark: bool,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("ThemePickerPopup")
        self._entries = entries
        self._current_scheme_id = current_scheme_id
        self._is_dark = is_dark
        self._pending_query = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._search = QLineEdit()
        self._search.setObjectName("ThemePickerSearch")
        self._search.setPlaceholderText("Search themes…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search_changed)
        self._search.installEventFilter(self)
        root.addWidget(self._search)

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(_SEARCH_DEBOUNCE_MS)
        self._filter_timer.timeout.connect(self._apply_pending_filter)

        self._list = QListWidget()
        self._list.setObjectName("ThemePickerList")
        self._list.itemActivated.connect(self._on_item_activated)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.installEventFilter(self)
        root.addWidget(self._list, stretch=1)

        self.setFixedWidth(self._POPUP_WIDTH)
        self._apply_filter("")

    def apply_theme(self, *, is_dark: bool) -> None:
        self._is_dark = is_dark
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#ThemePickerPopup {
                    background: #232337;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 10px;
                }
                QLineEdit#ThemePickerSearch {
                    background: #1e1e2e;
                    color: #cdd6f4;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 8px;
                    padding: 8px 10px;
                }
                QListWidget#ThemePickerList {
                    background: transparent;
                    color: #cdd6f4;
                    border: none;
                    outline: none;
                }
                QListWidget#ThemePickerList::item {
                    padding: 8px 10px;
                    border-radius: 6px;
                }
                QListWidget#ThemePickerList::item:selected {
                    background: rgba(139,92,246,0.25);
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#ThemePickerPopup {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 10px;
                }
                QLineEdit#ThemePickerSearch {
                    background: #f8fafc;
                    color: #1e293b;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 8px 10px;
                }
                QListWidget#ThemePickerList {
                    background: transparent;
                    color: #1e293b;
                    border: none;
                    outline: none;
                }
                QListWidget#ThemePickerList::item {
                    padding: 8px 10px;
                    border-radius: 6px;
                }
                QListWidget#ThemePickerList::item:selected {
                    background: rgba(139,92,246,0.15);
                }
                """
            )

    def show_below(self, anchor: QWidget) -> None:
        self.adjustSize()
        height = min(self._POPUP_MAX_HEIGHT, self.sizeHint().height())
        self.setFixedHeight(max(height, 160))
        global_pos = anchor.mapToGlobal(anchor.rect().bottomLeft())
        self.move(global_pos)
        self._search.setFocus()
        self._select_current_item()
        self.show()

    def eventFilter(self, watched, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(watched, event)

        key_event = event
        if not isinstance(key_event, QKeyEvent):
            return super().eventFilter(watched, event)

        if watched is self._search:
            if key_event.key() == Qt.Key.Key_Down:
                self._focus_list(select_first=True)
                return True
            if key_event.key() == Qt.Key.Key_Up and self._list.count() > 0:
                self._focus_list(select_first=False)
                return True

        if watched is self._list:
            if key_event.key() == Qt.Key.Key_Up and self._list.currentRow() <= 0:
                self._search.setFocus()
                return True
            if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                item = self._list.currentItem()
                if item is not None:
                    self._emit_selection(item)
                return True

        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            item = self._list.currentItem()
            if item is not None:
                self._emit_selection(item)
            return
        super().keyPressEvent(event)

    def _on_search_changed(self, query: str) -> None:
        self._pending_query = query
        self._filter_timer.start()

    def _apply_pending_filter(self) -> None:
        self._apply_filter(self._pending_query)

    def _focus_list(self, *, select_first: bool) -> None:
        if self._list.count() == 0:
            return
        self._list.setFocus()
        if select_first:
            self._list.setCurrentRow(0)
        elif self._list.currentRow() < 0:
            self._list.setCurrentRow(0)

    def _apply_filter(self, query: str) -> None:
        needle = query.strip().lower()
        self._list.clear()
        current_row = 0
        row = 0

        for entry in self._entries:
            if needle and needle not in entry.search_text:
                continue

            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.ItemDataRole.UserRole, entry.scheme_id)
            item.setIcon(_swatch_icon(entry.swatch_color))
            if entry.variant_display_name:
                item.setToolTip(f"{entry.display_name} · {entry.variant_display_name}")
            self._list.addItem(item)
            if entry.scheme_id == self._current_scheme_id:
                current_row = row
            row += 1

        if self._list.count() == 0:
            empty = QListWidgetItem("No matching themes")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(empty)
        else:
            self._list.setCurrentRow(current_row)

    def _select_current_item(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            scheme_id = item.data(Qt.ItemDataRole.UserRole)
            if scheme_id == self._current_scheme_id:
                self._list.setCurrentRow(row)
                return

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._emit_selection(item)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        self._emit_selection(item)

    def _emit_selection(self, item: QListWidgetItem) -> None:
        scheme_id = item.data(Qt.ItemDataRole.UserRole)
        if not scheme_id:
            return
        self.schemeSelected.emit(str(scheme_id))
        self.close()


class ThemePickerButton(SelectorButton):
    """Dropdown control that opens a searchable theme list."""

    schemeSelected = pyqtSignal(str)

    def __init__(self, text: str = "Theme", parent=None, is_dark: bool = True) -> None:
        super().__init__(text, parent=parent, is_dark=is_dark)
        self.setObjectName("ThemesThemePicker")
        self._entries: tuple[ThemePickerEntry, ...] = ()
        self._current_scheme_id = ""
        self._popup: ThemePickerPopup | None = None
        self.clicked.connect(self._open_picker)

    def set_picker_model(
        self,
        *,
        entries: tuple[ThemePickerEntry, ...],
        current_scheme_id: str,
        display_name: str,
    ) -> None:
        self._entries = entries
        self._current_scheme_id = current_scheme_id
        self.setText(display_name)

    def _open_picker(self) -> None:
        if not self._entries:
            return
        if self._popup is not None:
            self._popup.close()
        self._popup = ThemePickerPopup(
            self.window(),
            entries=self._entries,
            current_scheme_id=self._current_scheme_id,
            is_dark=self._is_dark,
        )
        self._popup.apply_theme(is_dark=self._is_dark)
        self._popup.schemeSelected.connect(self._on_scheme_selected)
        self._popup.show_below(self)

    def _on_scheme_selected(self, scheme_id: str) -> None:
        self.schemeSelected.emit(scheme_id)
