"""Quick-insert row for recent or suggested @ mentions above the chat composer."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from core.composer_discoverability import RecentMention
from core.composer_attachments import composer_tool_tooltip, composer_tool_by_id
from core.theme.color_utils import theme_qcolor
from core.theme.widget_styles import PLACEHOLDER_MUTED
from ui.views.settings.settings_theme import resolve_settings_theme


class ComposerRecentMentionsRow(QWidget):
    """Horizontal chip row: default suggestions until the user picks real @ mentions."""

    mention_clicked = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerRecentMentionsRow")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._label = QLabel("")
        self._label.setObjectName("ComposerRecentMentionsLabel")
        self._layout.addWidget(self._label, 0, Qt.AlignmentFlag.AlignLeft)
        self._buttons: list[QPushButton] = []
        self._layout.addStretch(1)

    def set_entries(self, entries: list[RecentMention], *, using_defaults: bool) -> None:
        for button in self._buttons:
            self._layout.removeWidget(button)
            button.deleteLater()
        self._buttons.clear()

        self._label.setText("Try:" if using_defaults else "Recent:")
        self._label.setVisible(bool(entries))

        for mention in entries:
            button = QPushButton(f"@{mention.label}")
            button.setObjectName("ComposerRecentMentionChip")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setToolTip(self._tooltip_for(mention))
            button.clicked.connect(
                lambda _checked=False, item=mention: self.mention_clicked.emit(item)
            )
            insert_at = self._layout.count() - 1
            self._layout.insertWidget(insert_at, button, 0, Qt.AlignmentFlag.AlignLeft)
            self._buttons.append(button)

        self.setVisible(bool(entries))

    def apply_theme(self, is_dark: bool) -> None:
        theme = resolve_settings_theme(self, is_dark=is_dark)
        muted = theme_qcolor(theme.color(PLACEHOLDER_MUTED))
        self._label.setStyleSheet(f"color: {muted.name()}; font-size: 11px;")
        chip_style = (
            "QPushButton#ComposerRecentMentionChip {"
            f"color: {muted.name()};"
            "background: transparent;"
            "border: 1px solid rgba(128,128,128,0.35);"
            "border-radius: 10px;"
            "padding: 2px 10px;"
            "font-size: 11px;"
            "}"
            "QPushButton#ComposerRecentMentionChip:hover {"
            "background: rgba(128,128,128,0.12);"
            "}"
        )
        for button in self._buttons:
            button.setStyleSheet(chip_style)

    @staticmethod
    def _tooltip_for(mention: RecentMention) -> str:
        if mention.kind == "tool":
            tool = composer_tool_by_id(mention.id)
            if tool is not None:
                return composer_tool_tooltip(tool)
        return f"Insert @{mention.label}"
