"""Searchable system-font picker for Settings → Themes reading font."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from core.reading_fonts import system_reading_font_families
from core.theme.widget_styles import (
    PRESTIGE_DIALOG_CANCEL,
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_DIALOG_CONTAINER,
    PRESTIGE_DIALOG_INPUT,
    PRESTIGE_DIALOG_LIST,
    PRESTIGE_DIALOG_MESSAGE,
    PRESTIGE_DIALOG_TITLE,
    prestige_accent_colors,
)
from ui.components.prestige_dialog import (
    _center_dialog_on_host,
    _dialog_theme,
    _PRESTIGE_BTN_BASE,
)


class ReadingFontPickerDialog(QDialog):
    """Pick a font family installed on this system."""

    def __init__(
        self,
        parent,
        *,
        initial_family: str | None = None,
        is_dark: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(460)
        self.setMinimumHeight(420)

        self._selected_family: str | None = None
        self._all_families = system_reading_font_families()

        if is_dark is None:
            is_dark = getattr(parent.window() if parent else None, "_is_dark_theme", True)
        theme = _dialog_theme(parent, is_dark)
        accent, confirm_fg = prestige_accent_colors(theme, tone="default", title="Font")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("DialogContainer")
        container.setStyleSheet(
            theme.style(PRESTIGE_DIALOG_CONTAINER, accent=accent, object_name="DialogContainer")
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)

        title = QLabel("CHOOSE A SYSTEM FONT")
        title.setStyleSheet(theme.style(PRESTIGE_DIALOG_TITLE, accent=accent))
        message = QLabel(
            "Fonts installed on this computer. Qube uses the family name only; "
            "the font is not bundled with the app."
        )
        message.setWordWrap(True)
        message.setStyleSheet(theme.style(PRESTIGE_DIALOG_MESSAGE))

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search fonts…")
        self._search.setClearButtonEnabled(True)
        self._search.setStyleSheet(theme.style(PRESTIGE_DIALOG_INPUT))
        self._search.textChanged.connect(self._apply_filter)

        self._list = QListWidget()
        self._list.setStyleSheet(theme.style(PRESTIGE_DIALOG_LIST))
        self._list.itemDoubleClicked.connect(self._accept_current_item)
        self._populate_list(self._all_families)

        self._status = QLabel("")
        self._status.setObjectName("PrestigeDialogMuted")
        self._status.setStyleSheet(theme.style(PRESTIGE_DIALOG_MESSAGE))
        self._update_status(len(self._all_families))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setStyleSheet(
            theme.style(PRESTIGE_DIALOG_CANCEL, btn_base=_PRESTIGE_BTN_BASE)
        )
        cancel_btn.clicked.connect(self.reject)
        select_btn = QPushButton("SELECT")
        select_btn.setStyleSheet(
            theme.style(
                PRESTIGE_DIALOG_CONFIRM,
                btn_base=_PRESTIGE_BTN_BASE,
                accent=accent,
                confirm_fg=confirm_fg,
            )
        )
        select_btn.clicked.connect(self._accept_selection)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(select_btn)

        layout.addWidget(title)
        layout.addWidget(message)
        layout.addWidget(self._search)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._status)
        layout.addLayout(btn_row)
        outer.addWidget(container)

        if initial_family:
            self._select_family(initial_family)

    def selected_family(self) -> str | None:
        return self._selected_family

    def showEvent(self, event) -> None:
        super().showEvent(event)
        _center_dialog_on_host(self)
        self._search.setFocus(Qt.FocusReason.OtherFocusReason)

    def _populate_list(self, families: tuple[str, ...]) -> None:
        self._list.clear()
        for family in families:
            item = QListWidgetItem(family)
            font = QFont(family, 12)
            item.setFont(font)
            item.setData(Qt.ItemDataRole.UserRole, family)
            self._list.addItem(item)

    def _apply_filter(self, query: str) -> None:
        needle = query.strip().casefold()
        if not needle:
            filtered = self._all_families
        else:
            filtered = tuple(
                family for family in self._all_families if needle in family.casefold()
            )
        self._populate_list(filtered)
        self._update_status(len(filtered))

    def _update_status(self, visible_count: int) -> None:
        total = len(self._all_families)
        if visible_count == total:
            self._status.setText(f"{total} fonts on this system")
        else:
            self._status.setText(f"Showing {visible_count} of {total} fonts")

    def _select_family(self, family: str) -> None:
        target = family.casefold()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            value = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if value.casefold() == target:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)
                return
        self._search.setText(family)
        self._apply_filter(family)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            value = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if value.casefold() == target:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)
                return

    def _accept_current_item(self, item: QListWidgetItem) -> None:
        family = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not family:
            return
        self._selected_family = family
        self.accept()

    def _accept_selection(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        family = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not family:
            return
        self._selected_family = family
        self.accept()
