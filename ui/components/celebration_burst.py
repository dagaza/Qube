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


def _widget_rect_in_parent(widget: QWidget, parent: QWidget) -> QRect:
    if widget is None or parent is None or not widget.isVisible():
        return QRect()
    top_left = widget.mapTo(parent, QPoint(0, 0))
    return QRect(top_left, widget.size())


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
    """In-tree particle overlay around ``target`` (no extra top-level window)."""

    finished = pyqtSignal()

    def __init__(
        self,
        target: QWidget,
        *,
        overlay_parent: QWidget,
        duration_ms: int = 3200,
        burst_interval_ms: int = 520,
        spread_margin: int = 20,
    ) -> None:
        super().__init__(overlay_parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAutoFillBackground(False)
        self._target = target
        self._overlay_parent = overlay_parent
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

    def _target_rect_in_parent(self) -> QRect:
        return _widget_rect_in_parent(self._target, self._overlay_parent)

    def _sync_geometry(self) -> None:
        target_in_parent = self._target_rect_in_parent()
        if target_in_parent.isNull():
            return
        self.setGeometry(_pad_rect(target_in_parent, self._spread_margin))

    def _inner_target_rect(self) -> QRect:
        """Target widget bounds in this overlay's local coordinates."""
        target_in_parent = self._target_rect_in_parent()
        if target_in_parent.isNull():
            return QRect()
        outer = self.geometry()
        return QRect(
            target_in_parent.left() - outer.left(),
            target_in_parent.top() - outer.top(),
            target_in_parent.width(),
            target_in_parent.height(),
        )

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
        self._sync_geometry()
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
    target: QWidget,
    *,
    overlay_parent: QWidget | None = None,
    duration_ms: int = 3200,
    on_finished: Callable[[], None] | None = None,
) -> BorderFireworksHandle | None:
    """Surround ``target`` with fireworks inside ``overlay_parent`` (no new window)."""
    if _prefers_reduced_motion():
        if on_finished is not None:
            on_finished()
        return None
    if target is None or not target.isVisible():
        if on_finished is not None:
            on_finished()
        return None
    parent = overlay_parent or target.parentWidget() or target.window() or target
    seq = _BorderFireworksSequence(
        target,
        overlay_parent=parent,
        duration_ms=duration_ms,
    )
    if on_finished is not None:
        seq.finished.connect(on_finished)
    return BorderFireworksHandle(seq)


def show_composer_border_fireworks(
    host: QWidget,
    *,
    overlay_parent: QWidget | None = None,
    duration_ms: int = 3200,
    on_finished: Callable[[], None] | None = None,
) -> None:
    """Alias for :func:`show_border_fireworks`."""
    show_border_fireworks(
        host,
        overlay_parent=overlay_parent,
        duration_ms=duration_ms,
        on_finished=on_finished,
    )
