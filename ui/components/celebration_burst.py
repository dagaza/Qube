"""Celebratory particle effects for first-run feature discovery."""

from __future__ import annotations

import math
import random
from typing import Callable

from PyQt6.QtCore import QPoint, QPointF, QRect, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPainter
from PyQt6.QtWidgets import QWidget

_PALETTE = ("#f9e2af", "#fab387", "#89b4fa", "#cba6f7", "#a6e3a1", "#f38ba8")


def _pad_rect(rect: QRect, margin: int) -> QRect:
    if rect.isNull():
        return rect
    return rect.adjusted(-margin, -margin, margin, margin)


def _global_widget_rect(widget: QWidget | None, *, margin: int = 0) -> QRect:
    if widget is None or not widget.isVisible():
        return QRect()
    top_left = widget.mapToGlobal(QPoint(0, 0))
    return _pad_rect(QRect(top_left, widget.size()), margin)


def _prefers_reduced_motion() -> bool:
    try:
        from core.app_settings import get_companion_reduced_motion

        override = get_companion_reduced_motion()
    except Exception:
        override = None
    if override is True:
        return True
    if override is None:
        hints = QGuiApplication.styleHints()
        try:
            if hints.useAnimations() is False:
                return True
        except Exception:
            pass
    return False


def _spawn_particles(
    particles: list[dict],
    origin: QPointF,
    *,
    count: int = 18,
    spread: float = 7.5,
) -> None:
    for _ in range(count):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(2.0, spread)
        particles.append(
            {
                "x": float(origin.x()),
                "y": float(origin.y()),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - random.uniform(0.5, 2.5),
                "life": random.uniform(0.55, 1.05),
                "color": random.choice(_PALETTE),
                "size": random.uniform(3.0, 6.5),
            }
        )


def _spawn_border_particles(particles: list[dict], rect: QRect, *, count: int = 28) -> None:
    """Emit bursts from points along the target widget border."""
    if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
        return
    margin = 4
    r = rect.adjusted(-margin, -margin, margin, margin)
    for _ in range(count):
        edge = random.randint(0, 3)
        if edge == 0:
            x = random.uniform(r.left(), r.right())
            y = float(r.top())
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(-6.0, -2.0)
        elif edge == 1:
            x = float(r.right())
            y = random.uniform(r.top(), r.bottom())
            vx = random.uniform(2.0, 6.0)
            vy = random.uniform(-2.0, 2.0)
        elif edge == 2:
            x = random.uniform(r.left(), r.right())
            y = float(r.bottom())
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(2.0, 6.0)
        else:
            x = float(r.left())
            y = random.uniform(r.top(), r.bottom())
            vx = random.uniform(-6.0, -2.0)
            vy = random.uniform(-2.0, 2.0)
        particles.append(
            {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "life": random.uniform(0.7, 1.2),
                "color": random.choice(_PALETTE),
                "size": random.uniform(3.5, 7.0),
            }
        )


class BorderFireworksHandle:
    """Controls a running border fireworks overlay."""

    def __init__(self, seq: _BorderFireworksSequence | None) -> None:
        self._seq = seq

    def stop(self) -> None:
        seq = self._seq
        if seq is None:
            return
        self._seq = None
        seq.stop()


class _BorderFireworksSequence(QWidget):
    """Top-level border bursts around ``target`` (works with Qt.Popup menus)."""

    finished = pyqtSignal()

    def __init__(
        self,
        target: QWidget,
        *,
        duration_ms: int = 3200,
        burst_interval_ms: int = 520,
        spread_margin: int = 20,
    ) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._target = target
        self._spread_margin = spread_margin
        self._particles: list[dict] = []
        self._elapsed_ms = 0
        self._duration_ms = duration_ms
        self._burst_interval_ms = burst_interval_ms
        self._next_burst_ms = 0
        self._stopped = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._sync_geometry()
        self._emit_burst()
        self._timer.start(16)
        self.show()
        self.raise_()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._timer.stop()
        except RuntimeError:
            pass
        self.hide()
        self.deleteLater()

    def _sync_geometry(self) -> None:
        outer = _global_widget_rect(self._target, margin=self._spread_margin)
        if outer.isNull():
            return
        self.setGeometry(outer)

    def _inner_target_rect(self) -> QRect:
        """Target widget bounds in this overlay's local coordinates."""
        inner = _global_widget_rect(self._target, margin=0)
        if inner.isNull():
            return QRect()
        origin = self.mapFromGlobal(inner.topLeft())
        return QRect(origin, inner.size())

    def _emit_burst(self) -> None:
        self._sync_geometry()
        inner = self._inner_target_rect()
        _spawn_border_particles(self._particles, inner, count=36)
        if not inner.isNull():
            center = inner.center()
            _spawn_particles(
                self._particles,
                QPointF(center),
                count=12,
                spread=6.0,
            )

    def _tick(self) -> None:
        self._elapsed_ms += 16
        dt = 0.016
        if self._elapsed_ms >= self._next_burst_ms and self._elapsed_ms < self._duration_ms:
            self._next_burst_ms = self._elapsed_ms + self._burst_interval_ms
            self._emit_burst()
        for p in self._particles:
            if p["life"] <= 0.0:
                continue
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 4.5 * dt
            p["life"] -= dt * 0.85
        self.update()
        if self._elapsed_ms >= self._duration_ms:
            self.stop()
            self.finished.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for p in self._particles:
            if p["life"] <= 0.0:
                continue
            color = QColor(p["color"])
            color.setAlphaF(max(0.0, min(1.0, p["life"])))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            radius = p["size"] * max(0.35, p["life"])
            painter.drawEllipse(QPointF(p["x"], p["y"]), radius, radius)


def show_celebration_burst(anchor_widget: QWidget, anchor_global: QPoint) -> None:
    """Play a short point burst unless reduced motion is preferred."""
    if _prefers_reduced_motion():
        return
    show_border_fireworks(
        anchor_widget,
        duration_ms=900,
        on_finished=None,
    )


def show_border_fireworks(
    host: QWidget,
    *,
    duration_ms: int = 3200,
    on_finished: Callable[[], None] | None = None,
) -> BorderFireworksHandle | None:
    """Surround ``host`` widget borders with fireworks for a few seconds."""
    if _prefers_reduced_motion():
        if on_finished is not None:
            on_finished()
        return None
    if host is None or not host.isVisible():
        if on_finished is not None:
            on_finished()
        return None
    seq = _BorderFireworksSequence(host, duration_ms=duration_ms)
    if on_finished is not None:
        seq.finished.connect(on_finished)
    return BorderFireworksHandle(seq)


def show_composer_border_fireworks(
    host: QWidget,
    *,
    duration_ms: int = 3200,
    on_finished: Callable[[], None] | None = None,
) -> None:
    """Alias for :func:`show_border_fireworks`."""
    show_border_fireworks(host, duration_ms=duration_ms, on_finished=on_finished)
