"""Full-screen compass snap hints shown while dragging the desktop companion."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QFont, QPainter, QColor, QScreen
from PyQt6.QtWidgets import QWidget

from core.companion_placement import (
    COMPANION_SNAP_ZONE_LABELS,
    COMPASS_SNAP_ZONES,
    CompanionSnapZone,
    snap_zone_label_box,
    workspace_for_screen,
)
from core.platform.frameless_window import apply_translucent_window_chrome

_LABEL_FONT_PT = 44
_CENTER_FONT_PT = 34
_IDLE_ALPHA = 55
_ACTIVE_ALPHA = 210
_ACTIVE_GLOW_ALPHA = 90


class CompanionSnapOverlay(QWidget):
    """Transparent always-on-top layer with compass labels; ignores mouse input."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setObjectName("CompanionSnapOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        apply_translucent_window_chrome(self)

        self._work_area = QRect()
        self._local_area = QRect()
        self._active_zone = CompanionSnapZone.NONE

    def show_for_screen(self, screen: QScreen | None) -> None:
        geo = workspace_for_screen(screen)
        if geo is None:
            return
        self._work_area = geo
        self._local_area = QRect(0, 0, geo.width(), geo.height())
        self._active_zone = CompanionSnapZone.NONE
        self.setGeometry(geo)
        self.show()
        self.raise_()
        self.update()

    def set_highlight(self, zone: CompanionSnapZone | str | None) -> None:
        if isinstance(zone, str):
            try:
                zone = CompanionSnapZone(zone)
            except ValueError:
                zone = CompanionSnapZone.NONE
        active = zone if isinstance(zone, CompanionSnapZone) else CompanionSnapZone.NONE
        if active == self._active_zone:
            return
        self._active_zone = active
        self.update()

    def hide_overlay(self) -> None:
        self._active_zone = CompanionSnapZone.NONE
        self.hide()

    def paintEvent(self, _event) -> None:
        if self._local_area.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        for zone in COMPASS_SNAP_ZONES:
            label = COMPANION_SNAP_ZONE_LABELS.get(zone, zone.value.upper())
            active = zone == self._active_zone
            font_pt = _CENTER_FONT_PT if zone == CompanionSnapZone.CENTER else _LABEL_FONT_PT
            font = QFont("Inter", font_pt)
            font.setWeight(QFont.Weight.Bold)
            painter.setFont(font)

            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label)
            text_h = metrics.height()
            x, y, w, h = snap_zone_label_box(
                zone,
                self._local_area,
                text_width=text_w,
                text_height=text_h,
                local=True,
            )
            rect = QRect(x, y, w, h)

            if active:
                glow = QColor(129, 140, 248, _ACTIVE_GLOW_ALPHA)
                for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                    painter.setPen(glow)
                    painter.drawText(rect.translated(dx, dy), int(Qt.AlignmentFlag.AlignCenter), label)

            alpha = _ACTIVE_ALPHA if active else _IDLE_ALPHA
            painter.setPen(QColor(226, 232, 240, alpha))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

        painter.end()
