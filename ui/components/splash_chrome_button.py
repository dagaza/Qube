"""Minimize / close controls on the frameless startup splash."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, QSize, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QPushButton, QWidget

from ui.branded_theme import (
    SPLASH_CHROME_BUTTON_BG_HOVER_OPAQUE,
    SPLASH_CHROME_BUTTON_BG_OPAQUE,
    SPLASH_CHROME_BUTTON_BORDER_HOVER_OPAQUE,
    SPLASH_CHROME_BUTTON_BORDER_OPAQUE,
    SPLASH_CHROME_ICON,
)

_CORNER_RADIUS_PX = 6.0


class SplashChromeButton(QPushButton):
    """Icon-only splash window control with manual paint (no native focus frame)."""

    def __init__(
        self,
        *,
        object_name: str,
        icon_name: str,
        tooltip: str,
        parent: QWidget,
        on_click: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self.setObjectName(object_name)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAutoDefault(False)
        self.setDefault(False)
        self.setFlat(True)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setStyleSheet("background: transparent; border: none; margin: 0; padding: 0;")
        self._set_icon()
        self.clicked.connect(on_click)

    def _set_icon(self) -> None:
        import qtawesome as qta

        self.setIcon(qta.icon(self._icon_name, color=SPLASH_CHROME_ICON))
        self.setIconSize(QSize(12, 12))

    def paintEvent(self, _event) -> None:  # noqa: N802
        hovered = self.underMouse() and self.isEnabled()
        pressed = self.isDown() and self.isEnabled()
        if pressed or hovered:
            bg = SPLASH_CHROME_BUTTON_BG_HOVER_OPAQUE
            border = SPLASH_CHROME_BUTTON_BORDER_HOVER_OPAQUE
        else:
            bg = SPLASH_CHROME_BUTTON_BG_OPAQUE
            border = SPLASH_CHROME_BUTTON_BORDER_OPAQUE

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(border), 1.0))
        painter.setBrush(QColor(bg))
        painter.drawRoundedRect(rect, _CORNER_RADIUS_PX, _CORNER_RADIUS_PX)

        icon = self.icon()
        if not icon.isNull():
            pixmap = icon.pixmap(self.iconSize())
            x = (self.width() - pixmap.width()) // 2
            y = (self.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)
        painter.end()