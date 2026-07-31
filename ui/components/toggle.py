from PyQt6.QtCore import QPropertyAnimation, QPoint, Qt, pyqtProperty, QRectF
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton

from core.theme.accessors import theme_for
from core.theme.color_utils import theme_qcolor
from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import prestige_toggle_palette


class PrestigeToggle(QAbstractButton):
    _TRACK_RADIUS = 11.0
    _KNOB_DIAMETER = 16
    _KNOB_Y = 3
    _KNOB_OFF_X = 3
    _KNOB_ON_X = 19
    _BORDER_WIDTH = 1.0
    _FOCUS_BORDER_WIDTH = 2.0

    def __init__(self, parent=None, *, is_dark: bool = True):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: transparent; border: none; outline: none;")

        self._track_fill = theme_qcolor("#45475a")
        self._track_border = theme_qcolor("#89b4fa")
        self._knob_color = theme_qcolor("#ffffff")

        self._circle_position = self._KNOB_OFF_X
        self.setFixedSize(38, 22)

        self.animation = QPropertyAnimation(self, b"circle_position", self)
        self.animation.setDuration(200)

        self.apply_theme(is_dark=is_dark)

    def apply_theme(self, *, is_dark: bool, theme: ResolvedTheme | None = None) -> None:
        """Sync track, border, and knob colors with the active theme."""
        resolved = theme_for(is_dark=is_dark, resolved=theme)
        palette = prestige_toggle_palette(resolved)
        self._palette = palette
        self._track_fill = theme_qcolor(palette["track_unchecked_fill"])
        self._track_checked_fill = theme_qcolor(palette["track_checked_fill"])
        self._track_checked_border = theme_qcolor(palette["track_checked_border"])
        self._track_checked_border_focused = theme_qcolor(
            palette["track_checked_border_focused"]
        )
        self._track_unchecked_border = theme_qcolor(palette["track_unchecked_border"])
        self._track_unchecked_border_focused = theme_qcolor(
            palette["track_unchecked_border_focused"]
        )
        self._track_unchecked_border_hover = theme_qcolor(
            palette["track_unchecked_border_hover"]
        )
        self._track_disabled_fill = theme_qcolor(palette["track_disabled_fill"])
        self._track_disabled_border = theme_qcolor(palette["track_disabled_border"])
        self._knob_color = theme_qcolor(palette["knob"])
        self._knob_disabled = theme_qcolor(palette["knob_disabled"])
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
        self._circle_position = self._KNOB_ON_X if checked else self._KNOB_OFF_X
        self.update()

    def focusInEvent(self, event) -> None:  # noqa: N802
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        self.update()

    def _track_colors(self) -> tuple:
        if not self.isEnabled():
            return self._track_disabled_fill, self._track_disabled_border, self._BORDER_WIDTH

        if self.isChecked():
            border = self._track_checked_border
            if self.hasFocus():
                border = self._track_checked_border_focused
            return self._track_checked_fill, border, self._BORDER_WIDTH

        border = self._track_unchecked_border
        width = self._BORDER_WIDTH
        if self.hasFocus():
            border = self._track_unchecked_border_focused
            width = self._FOCUS_BORDER_WIDTH
        elif self.underMouse():
            border = self._track_unchecked_border_hover
        return self._track_fill, border, width

    def paintEvent(self, e):
        del e
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        fill, border, border_width = self._track_colors()
        inset = border_width / 2.0
        track_rect = QRectF(
            inset,
            inset,
            self.width() - border_width,
            self.height() - border_width,
        )
        radius = self._TRACK_RADIUS - inset

        p.setPen(QPen(border, border_width))
        p.setBrush(fill)
        p.drawRoundedRect(track_rect, radius, radius)

        knob = self._knob_disabled if not self.isEnabled() else self._knob_color
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(knob)
        p.drawEllipse(
            self._circle_position,
            self._KNOB_Y,
            self._KNOB_DIAMETER,
            self._KNOB_DIAMETER,
        )

    def nextCheckState(self):
        super().nextCheckState()
        end_value = self._KNOB_ON_X if self.isChecked() else self._KNOB_OFF_X
        self.animation.setEndValue(end_value)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)
