"""Color swatch button for theme token pickers."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QColorDialog, QHBoxLayout, QLabel, QPushButton, QWidget

from core.theme.color_utils import contrasting_label_color, parse_color
from core.theme.constants import UNRESOLVED_TOKEN_COLOR


class ThemeColorSwatch(QWidget):
    """Label + swatch button that opens ``QColorDialog``."""

    colorChanged = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        color: str,
        *,
        parent=None,
        token_key: str = "",
    ) -> None:
        super().__init__(parent)
        self._token_key = token_key
        self._color = self._normalize_color(color)
        self.setMinimumHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._label = QLabel(label)
        self._label.setObjectName("SettingsSubsectionLabel")
        self._button = QPushButton()
        self._button.setObjectName("ThemeColorSwatchButton")
        self._button.setFixedSize(120, 32)
        self._button.setToolTip(f"Choose {label.lower()} color")
        self._button.clicked.connect(self._pick_color)
        layout.addWidget(self._label)
        layout.addWidget(self._button)
        layout.addStretch()
        self._apply_button_style()

    @property
    def token_key(self) -> str:
        return self._token_key

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        normalized = self._normalize_color(color)
        if normalized == self._color:
            return
        self._color = normalized
        self._apply_button_style()

    def _normalize_color(self, color: str) -> str:
        try:
            return parse_color(color).to_hex()
        except ValueError:
            return UNRESOLVED_TOKEN_COLOR

    def _apply_button_style(self) -> None:
        text_color = contrasting_label_color(self._color)
        self._button.setText(self._color)
        self._button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self._color};
                color: {text_color};
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 6px;
                padding: 4px 8px;
                font-family: monospace;
            }}
            """
        )

    def _pick_color(self) -> None:
        initial = QColor(self._color)
        chosen = QColorDialog.getColor(
            initial,
            self.window(),
            f"Choose {self._label.text()}",
        )
        if not chosen.isValid():
            return
        self.set_color(chosen.name())
        self.colorChanged.emit(self._color)
