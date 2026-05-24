"""Floating desktop companion orb — frameless always-on-top presence window."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtProperty, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QApplication, QFrame, QMenu, QVBoxLayout, QWidget

from core import app_settings
from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot

_ORB_COLORS: dict[AssistantActivity, str] = {
    AssistantActivity.ASSISTANT_OFF: "#64748b",
    AssistantActivity.IDLE_LISTEN: "#89b4fa",
    AssistantActivity.CAPTURING: "#f38ba8",
    AssistantActivity.WORKING: "#74c7ec",
    AssistantActivity.SPEAKING: "#a6e3a1",
    AssistantActivity.NEEDS_ATTENTION: "#f9e2af",
    AssistantActivity.ERROR: "#f38ba8",
    AssistantActivity.BACKGROUND_BUSY: "#cba6f7",
}

_CAPTION_MAX_CHARS = 42
_DOCK_STRIP_HEIGHT = 24
_MAGNETIC_EDGE_PX = 12


class CompanionWindow(QWidget):
    """Small always-on-top translucent orb with optional caption chip."""

    open_requested = pyqtSignal()
    hide_for_one_hour_requested = pyqtSignal()
    snooze_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("Qube Companion")

        self._is_dark = True
        self._snapshot: AssistantPresenceSnapshot | None = None
        self._orb_size = app_settings.get_companion_size_px()
        self._glow_opacity = 1.0
        self._idle_faded = False
        self._reduced_motion = False
        self._dock_mode = False
        self._pulse_phase = 0.0
        self._drag_offset: QPoint | None = None
        self._notify_pulse = 0.0
        self._volume_ring = 0.0

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(500 if self._reduced_motion else 33)
        self._anim_timer.timeout.connect(self._on_anim_tick)

        self._caption_frame = QFrame(self)
        self._caption_frame.setObjectName("CompanionCaptionFrame")
        self._caption_frame.hide()
        caption_layout = QVBoxLayout(self._caption_frame)
        caption_layout.setContentsMargins(10, 4, 10, 4)
        self._caption_label = QFrame()
        self._caption_label.setObjectName("CompanionCaptionLabel")
        caption_layout.addWidget(self._caption_label)

        self.setAccessibleName("Qube assistant presence")
        self._apply_caption_style()
        self._resize_for_mode()

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self._apply_caption_style()
        self.update()

    def set_reduced_motion(self, enabled: bool) -> None:
        self._reduced_motion = enabled
        self._anim_timer.setInterval(500 if enabled else 33)
        if enabled:
            self._pulse_phase = 0.0
        self.update()

    def set_dock_mode(self, enabled: bool) -> None:
        self._dock_mode = enabled
        self._resize_for_mode()
        self.update()

    def set_idle_faded(self, faded: bool) -> None:
        self._idle_faded = faded
        if faded:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.update()

    def set_snapshot(self, snapshot: AssistantPresenceSnapshot) -> None:
        self._snapshot = snapshot
        self._volume_ring = snapshot.audio_level
        show_caption = app_settings.get_companion_show_caption() and bool(snapshot.caption_text)
        if show_caption:
            self._caption_frame.show()
        else:
            self._caption_frame.hide()
        self._resize_for_mode()
        self.update()

    def pulse_notification(self) -> None:
        self._notify_pulse = 1.0
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def orb_center_global(self) -> QPoint:
        rect = self.rect()
        if self._dock_mode:
            return self.mapToGlobal(QPoint(rect.width() // 2, _DOCK_STRIP_HEIGHT // 2))
        margin = 8
        radius = self._orb_size // 2
        return self.mapToGlobal(QPoint(margin + radius, margin + radius))

    def get_glow_opacity(self) -> float:
        return self._glow_opacity

    def set_glow_opacity(self, value: float) -> None:
        self._glow_opacity = max(0.15, min(1.0, float(value)))
        self.update()

    glowOpacity = pyqtProperty(float, get_glow_opacity, set_glow_opacity)

    def _apply_caption_style(self) -> None:
        bg = "#1e1e2e" if self._is_dark else "#ffffff"
        fg = "#cdd6f4" if self._is_dark else "#1e293b"
        border = "#313244" if self._is_dark else "#cbd5e1"
        self._caption_frame.setStyleSheet(
            f"QFrame#CompanionCaptionFrame {{ background-color: {bg}; border: 1px solid {border};"
            f" border-radius: 8px; }}"
            f"QFrame#CompanionCaptionLabel {{ background: transparent; color: {fg}; }}"
        )

    def _resize_for_mode(self) -> None:
        self._orb_size = app_settings.get_companion_size_px()
        margin = 8
        caption_h = 0
        caption_w = 0
        if self._caption_frame.isVisible() and self._snapshot and self._snapshot.caption_text:
            text = self._snapshot.caption_text[:_CAPTION_MAX_CHARS]
            if len(self._snapshot.caption_text) > _CAPTION_MAX_CHARS:
                text += "…"
            fm = QFontMetrics(self.font())
            caption_w = min(280, fm.horizontalAdvance(text) + 24)
            caption_h = fm.height() + 12

        if self._dock_mode:
            screen = QApplication.primaryScreen()
            w = screen.availableGeometry().width() if screen else 400
            self.setFixedSize(max(200, w // 4), _DOCK_STRIP_HEIGHT)
            return

        total_w = max(self._orb_size + margin * 2, caption_w)
        total_h = self._orb_size + margin * 2 + (caption_h + 6 if caption_h else 0)
        self.setFixedSize(total_w, total_h)
        if caption_h:
            self._caption_frame.setGeometry(
                (total_w - caption_w) // 2,
                self._orb_size + margin * 2 + 4,
                caption_w,
                caption_h,
            )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._anim_timer.isActive():
            self._anim_timer.start()

    def hideEvent(self, event) -> None:
        self._anim_timer.stop()
        super().hideEvent(event)

    def _on_anim_tick(self) -> None:
        activity = self._snapshot.activity if self._snapshot else AssistantActivity.IDLE_LISTEN
        active = activity in (
            AssistantActivity.CAPTURING,
            AssistantActivity.WORKING,
            AssistantActivity.SPEAKING,
        )
        if self._reduced_motion:
            self._pulse_phase = 0.0
        elif activity == AssistantActivity.IDLE_LISTEN and not self._idle_faded:
            self._pulse_phase = (self._pulse_phase + 0.04) % (2 * math.pi)
        elif activity == AssistantActivity.WORKING:
            self._pulse_phase = (self._pulse_phase + 0.12) % (2 * math.pi)
        elif activity == AssistantActivity.CAPTURING:
            self._pulse_phase = (self._pulse_phase + 0.2) % (2 * math.pi)

        if self._notify_pulse > 0:
            self._notify_pulse = max(0.0, self._notify_pulse - 0.08)

        if active or self._notify_pulse > 0 or (
            activity == AssistantActivity.IDLE_LISTEN and not self._reduced_motion
        ):
            self.update()
        elif self._idle_faded and self._notify_pulse <= 0:
            self._anim_timer.setInterval(500)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._dock_mode:
            self._paint_dock_strip(painter)
            return

        activity = self._snapshot.activity if self._snapshot else AssistantActivity.IDLE_LISTEN
        color_hex = _ORB_COLORS.get(activity, "#89b4fa")
        base = QColor(color_hex)
        margin = 8
        center = QPointF(margin + self._orb_size / 2, margin + self._orb_size / 2)
        radius = self._orb_size / 2

        opacity = 0.35 if self._idle_faded else self._glow_opacity
        if self._notify_pulse > 0:
            opacity = min(1.0, opacity + self._notify_pulse * 0.4)

        breathe = 1.0
        if not self._reduced_motion and activity == AssistantActivity.IDLE_LISTEN:
            breathe = 1.0 + 0.04 * math.sin(self._pulse_phase)

        glow_radius = radius * 1.35 * breathe
        gradient = QRadialGradient(center, glow_radius)
        glow = QColor(base)
        glow.setAlphaF(0.45 * opacity)
        gradient.setColorAt(0.0, glow)
        outer = QColor(base)
        outer.setAlphaF(0.0)
        gradient.setColorAt(1.0, outer)
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, glow_radius, glow_radius)

        core = QColor(base)
        core.setAlphaF(0.92 * opacity)
        painter.setBrush(core)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawEllipse(center, radius * 0.72 * breathe, radius * 0.72 * breathe)

        if activity == AssistantActivity.CAPTURING and self._volume_ring > 0.05:
            ring = QColor(base)
            ring.setAlphaF(0.6)
            painter.setPen(QPen(ring, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_r = radius * (0.85 + self._volume_ring * 0.2)
            painter.drawEllipse(center, ring_r, ring_r)

        if activity == AssistantActivity.WORKING and not self._reduced_motion:
            painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
            arc_r = radius * 0.55
            start = int(self._pulse_phase * 180 / math.pi * 16)
            painter.drawArc(
                int(center.x() - arc_r),
                int(center.y() - arc_r),
                int(arc_r * 2),
                int(arc_r * 2),
                start,
                120 * 16,
            )

        if self._caption_frame.isVisible() and self._snapshot and self._snapshot.caption_text:
            text = self._snapshot.caption_text[:_CAPTION_MAX_CHARS]
            if len(self._snapshot.caption_text) > _CAPTION_MAX_CHARS:
                text += "…"
            painter.setPen(QColor("#cdd6f4" if self._is_dark else "#1e293b"))
            font = QFont(self.font())
            font.setPointSize(max(9, font.pointSize() - 1))
            painter.setFont(font)
            cap_rect = self._caption_frame.geometry()
            painter.drawText(
                cap_rect.adjusted(8, 0, -8, 0),
                int(Qt.AlignmentFlag.AlignCenter),
                text,
            )

    def _paint_dock_strip(self, painter: QPainter) -> None:
        activity = self._snapshot.activity if self._snapshot else AssistantActivity.IDLE_LISTEN
        color_hex = _ORB_COLORS.get(activity, "#89b4fa")
        base = QColor(color_hex)
        rect = QRectF(0, 0, self.width(), self.height())
        bg = QColor("#1e1e2e" if self._is_dark else "#ffffff")
        bg.setAlphaF(0.85 if not self._idle_faded else 0.35)
        painter.setBrush(bg)
        painter.setPen(QPen(base, 2))
        painter.drawRoundedRect(rect, 6, 6)
        dot = QRectF(8, (self.height() - 10) / 2, 10, 10)
        painter.setBrush(base)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(dot)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_offset is not None:
                moved = (event.globalPosition().toPoint() - self._drag_offset) != self.pos()
                self._drag_offset = None
                if not moved:
                    self.open_requested.emit()
                else:
                    self._snap_to_edge()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit()
            event.accept()

    def _snap_to_edge(self) -> None:
        screen = QApplication.screenAt(self.orb_center_global())
        if screen is None:
            return
        geo = screen.availableGeometry()
        pos = self.pos()
        x, y = pos.x(), pos.y()
        dock_edge = "none"

        if abs(x - geo.left()) < _MAGNETIC_EDGE_PX:
            x = geo.left() + 4
            dock_edge = "left"
        elif abs(x + self.width() - geo.right()) < _MAGNETIC_EDGE_PX:
            x = geo.right() - self.width() - 4
            dock_edge = "right"
        if abs(y + self.height() - geo.bottom()) < _MAGNETIC_EDGE_PX:
            y = geo.bottom() - self.height() - 4
            dock_edge = "bottom"
        elif abs(y - geo.top()) < _MAGNETIC_EDGE_PX:
            y = geo.top() + 4

        self.move(x, y)
        app_settings.set_companion_position(
            x=x,
            y=y,
            screen=screen.name(),
            norm_x=(x - geo.left()) / max(1, geo.width()),
            norm_y=(y - geo.top()) / max(1, geo.height()),
            dock_edge=dock_edge,
        )

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        bg = "#1e1e2e" if self._is_dark else "#ffffff"
        fg = "#cdd6f4" if self._is_dark else "#1e293b"
        menu.setStyleSheet(
            f"QMenu {{ background-color: {bg}; color: {fg}; }}"
            f"QMenu::item:selected {{ background-color: {'#313244' if self._is_dark else '#e2e8f0'}; }}"
        )
        open_act = menu.addAction("Open Qube")
        hide_act = menu.addAction("Hide for 1 hour")
        menu.addSeparator()
        settings_act = menu.addAction("Companion settings…")
        chosen = menu.exec(global_pos)
        if chosen == open_act:
            self.open_requested.emit()
        elif chosen == hide_act:
            self.hide_for_one_hour_requested.emit()
        elif chosen == settings_act:
            self.snooze_requested.emit()
