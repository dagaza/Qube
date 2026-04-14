"""
Animated "typing" / thinking dots for chat UIs.

Minimal usage::

    from ui.components.typing_indicator import TypingIndicatorWidget, TypingIndicatorMode

    row = QHBoxLayout()
    dots = TypingIndicatorWidget(mode=TypingIndicatorMode.FADE, dot_count=3)
    dots.set_dark_theme(True)
    dots.setFixedSize(48, 22)
    row.addWidget(dots)
    dots.start()
"""

from __future__ import annotations

import math
from enum import Enum, auto

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QBrush
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget, QLabel


def _smoothstep01(t: float) -> float:
    """Hermite ease in/out on [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class TypingIndicatorMode(Enum):
    FADE = auto()
    SCALE = auto()


class TypingIndicatorWidget(QWidget):
    """Three (or N) staggered dots with fade or scale animation driven by QTimer + QPainter."""

    COLOR_DARK = "#89b4fa"
    COLOR_LIGHT = "#000000"

    def __init__(
        self,
        parent=None,
        *,
        dot_count: int = 3,
        mode: TypingIndicatorMode = TypingIndicatorMode.FADE,
        fps: int = 50,
        cycle_ms: float = 1100.0,
        fixed_width: int = 48,
        fixed_height: int = 22,
    ) -> None:
        super().__init__(parent)
        self._dot_count = max(1, int(dot_count))
        self._mode = mode
        self._cycle_s = max(0.35, float(cycle_ms) / 1000.0)
        self._is_dark = True
        self._phase_s = 0.0
        self._margin_x = 2
        self._base_radius = 3.25
        self._min_opacity = 0.28
        self._max_opacity = 1.0
        self._min_scale = 0.58
        self._max_scale = 1.08

        self.setFixedSize(fixed_width, fixed_height)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        interval = max(8, int(round(1000 / max(12, min(60, int(fps))))))
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._on_tick)

    def dot_count(self) -> int:
        return self._dot_count

    def set_dot_count(self, n: int) -> None:
        self._dot_count = max(1, int(n))
        self.update()

    def mode(self) -> TypingIndicatorMode:
        return self._mode

    def set_mode(self, mode: TypingIndicatorMode) -> None:
        self._mode = mode
        self.update()

    def set_dark_theme(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.update()

    def start(self) -> None:
        self._phase_s = 0.0
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self.update()

    def is_animating(self) -> bool:
        return self._timer.isActive()

    def _on_tick(self) -> None:
        dt = self._timer.interval() / 1000.0
        self._phase_s += dt
        # keep bounded to avoid float drift over long runs
        if self._phase_s > 1e6:
            self._phase_s %= self._cycle_s
        self.update()

    def _dot_wave(self, index: int) -> float:
        """Returns eased pulse strength in [0, 1] for dot ``index``."""
        t = self._phase_s / self._cycle_s
        # phase offset so dots animate left → right in sequence
        local = t * 2.0 * math.pi - index * (2.0 * math.pi / self._dot_count) * 1.15
        raw = 0.5 + 0.5 * math.sin(local)
        return _smoothstep01(raw)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        base = QColor(self.COLOR_DARK if self._is_dark else self.COLOR_LIGHT)

        w, h = self.width(), self.height()
        cy = h * 0.5
        inner_w = max(1.0, w - 2 * self._margin_x)
        n = self._dot_count
        if n == 1:
            xs = [self._margin_x + inner_w * 0.5]
        else:
            xs = [self._margin_x + inner_w * (i / (n - 1)) for i in range(n)]

        painter.setPen(Qt.PenStyle.NoPen)

        for i, cx in enumerate(xs):
            pulse = self._dot_wave(i)
            if self._mode is TypingIndicatorMode.FADE:
                c = QColor(base)
                o = self._min_opacity + (self._max_opacity - self._min_opacity) * pulse
                c.setAlphaF(o)
                painter.setBrush(QBrush(c))
                r = self._base_radius
            else:
                c = QColor(base)
                c.setAlphaF(0.92)
                painter.setBrush(QBrush(c))
                s = self._min_scale + (self._max_scale - self._min_scale) * pulse
                r = self._base_radius * s

            x = int(round(cx - r))
            y = int(round(cy - r))
            d = int(round(2 * r))
            painter.drawEllipse(x, y, d, d)


class ScalingTypingIndicatorWidget(TypingIndicatorWidget):
    """Same as :class:`TypingIndicatorWidget` but defaults to pulsing scale instead of opacity."""

    def __init__(self, parent=None, **kwargs) -> None:
        kwargs.setdefault("mode", TypingIndicatorMode.SCALE)
        super().__init__(parent, **kwargs)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    app = QApplication(sys.argv)
    root = QWidget()
    root.setWindowTitle("TypingIndicatorWidget demo")
    outer = QVBoxLayout(root)

    outer.addWidget(QLabel("Fade (default)"))
    fade_row = QHBoxLayout()
    fade = TypingIndicatorWidget(mode=TypingIndicatorMode.FADE, dot_count=3)
    fade.set_dark_theme(True)
    fade_row.addWidget(fade)
    fade_row.addStretch(1)
    outer.addLayout(fade_row)

    outer.addWidget(QLabel("Scale"))
    scale_row = QHBoxLayout()
    scale = ScalingTypingIndicatorWidget(dot_count=3)
    scale.set_dark_theme(True)
    scale_row.addWidget(scale)
    scale_row.addStretch(1)
    outer.addLayout(scale_row)

    fade.start()
    scale.start()
    root.resize(320, 160)
    root.show()
    sys.exit(app.exec())