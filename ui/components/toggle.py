from PyQt6.QtWidgets import QAbstractButton
from PyQt6.QtCore import QPropertyAnimation, QPoint, Qt, pyqtProperty, QRect, QSize
from PyQt6.QtGui import QPainter, QColor

from core.theme.accessors import theme_for
from core.theme.color_utils import theme_qcolor
from core.theme.tokens import ResolvedTheme


class PrestigeToggle(QAbstractButton):
    def __init__(self, parent=None, *, is_dark: bool = True):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self._active_color = QColor("#10b981")
        self._bg_color = QColor("#45475a")
        self._circle_color = QColor("#ffffff")

        self._circle_position = 3
        self.setFixedSize(38, 22)

        self.animation = QPropertyAnimation(self, b"circle_position", self)
        self.animation.setDuration(200)

        self.apply_theme(is_dark=is_dark)

    def apply_theme(self, *, is_dark: bool, theme: ResolvedTheme | None = None) -> None:
        """Sync off-track colors with app light/dark mode."""
        resolved = theme_for(is_dark=is_dark, resolved=theme)
        self._bg_color = theme_qcolor(
            resolved.surface_pressed if resolved.is_dark else resolved.border
        )
        self._active_color = theme_qcolor(resolved.success)
        self._circle_color = theme_qcolor(resolved.text_on_accent)
        self.update()

    @pyqtProperty(int)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        self.animation.stop()
        self._circle_position = 19 if checked else 3
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        color = self._active_color if self.isChecked() else self._bg_color
        p.setBrush(color)
        p.drawRoundedRect(0, 0, self.width(), self.height(), 11, 11)

        p.setBrush(self._circle_color)
        p.drawEllipse(self._circle_position, 3, 16, 16)

    def nextCheckState(self):
        super().nextCheckState()
        end_value = 19 if self.isChecked() else 3
        self.animation.setEndValue(end_value)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)
