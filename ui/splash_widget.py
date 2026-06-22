"""Branded splash card content used by :mod:`ui.splash_overlay`."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

from core.paths import install_root
from ui.qube_wireframe_cube import (
    fit_radius_for_widget_side,
    paint_qube_wireframe_cube,
)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Startup steps shown beside the spinner (index must match splash_overlay phase order).
SPLASH_STEP_LABELS: tuple[str, ...] = (
    "Search models (Balanced)",
    "Document store & qube_data.db",
    "Audio, STT, native LLM, sidecar",
    "Memory enrichment workers",
    "Main window UI",
    "Service connections & sync",
    "Language model (optional)",
    "Kokoro TTS & audio runtime",
)

_SPLASH_CHUNK_PROGRESS_HEIGHT = 22


class _SplashChunkProgressBar(QProgressBar):
    """Pill progress bar; chunk stays rounded from the first pixel (QSS ::chunk cannot)."""

    _TRACK = QColor(255, 255, 255, 20)
    _CHUNK = QColor("#8b5cf6")
    _TEXT = QColor(148, 163, 184, 230)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("QubeSplashChunkProgress")
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("%p%")
        self.setFixedHeight(_SPLASH_CHUNK_PROGRESS_HEIGHT)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._TRACK)
        painter.drawRoundedRect(rect, radius, radius)

        span = self.maximum() - self.minimum()
        if span > 0:
            fraction = (self.value() - self.minimum()) / span
            if fraction > 0:
                chunk_rect = QRectF(rect)
                chunk_rect.setWidth(rect.width() * fraction)
                painter.setBrush(self._CHUNK)
                painter.drawRoundedRect(chunk_rect, radius, radius)

        if self.isTextVisible():
            text_font = QFont(painter.font())
            text_font.setPixelSize(10)
            painter.setFont(text_font)
            painter.setPen(self._TEXT)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


def resolve_splash_logo_path(repo_root: Path | None = None) -> Path | None:
    """Return the best available logo for the splash card, or ``None``."""
    root = repo_root or install_root()
    candidates = (
        root / "assets" / "logos" / "qube_logo_256.png",
        root / "assets" / "icons" / "qube_logo_256.png",
        root / "assets" / "qube_logo_256.png",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


class SplashCircleSpinner(QWidget):
    """Timer-driven ring spinner (decorative; may pause if the GUI thread blocks)."""

    def __init__(self, size: int = 40, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self._angle_deg = 0.0
        self._track = QColor(255, 255, 255, 28)
        self._arc = QColor("#8b5cf6")

    def apply_theme(self, is_dark: bool) -> None:
        if is_dark:
            self._track = QColor(255, 255, 255, 28)
            self._arc = QColor("#89b4fa")
        else:
            self._track = QColor(0, 0, 0, 22)
            self._arc = QColor("#3b82f6")
        self.update()

    def advance(self, delta_ms: float = 16.67) -> None:
        self._angle_deg = (self._angle_deg + delta_ms * 0.35) % 360.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        margin = max(2.0, side * 0.18)
        rect = QRectF(margin, margin, side - 2 * margin, side - 2 * margin)
        pen_width = max(1.5, side * 0.16)

        pen_track = QPen(self._track)
        pen_track.setWidthF(pen_width)
        pen_track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_track)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

        pen_arc = QPen(self._arc)
        pen_arc.setWidthF(pen_width)
        pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arc)
        span = 100 * 16
        start = int(self._angle_deg * 16)
        painter.drawArc(rect.toRect(), start, span)


class SplashStepList(QWidget):
    """Vertical list of startup items with pending / active / done styling."""

    def __init__(self, labels: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels = labels
        self._rows: list[QLabel] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for text in labels:
            row = QLabel(text)
            row.setWordWrap(True)
            layout.addWidget(row)
            self._rows.append(row)
        self._active_index = -1
        self._apply_row_styles()

    def set_active(self, index: int) -> None:
        self._active_index = index
        for i, row in enumerate(self._rows):
            if i < index:
                row.setProperty("step_state", "done")
            elif i == index:
                row.setProperty("step_state", "active")
            else:
                row.setProperty("step_state", "pending")
            row.style().unpolish(row)
            row.style().polish(row)

    def mark_done_through(self, index: int) -> None:
        for i, row in enumerate(self._rows):
            if i <= index:
                row.setProperty("step_state", "done")
            else:
                row.setProperty("step_state", "pending")
            row.style().unpolish(row)
            row.style().polish(row)
        self._active_index = -1

    def _apply_row_styles(self) -> None:
        self.setStyleSheet(
            """
            QLabel[step_state="pending"] {
                color: rgba(148, 163, 184, 0.55);
                font-size: 11px;
            }
            QLabel[step_state="active"] {
                color: #c4b5fd;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel[step_state="done"] {
                color: rgba(134, 239, 172, 0.85);
                font-size: 11px;
            }
            """
        )
        for row in self._rows:
            row.setProperty("step_state", "pending")
            row.style().unpolish(row)
            row.style().polish(row)


# Tutorial uses glRotatef(1, 3, 1, 1) with ~10ms waits (~0.1 deg/ms). Splash is slower.
_ROTATION_DEG_PER_MS = 0.035
# Grab-and-throw easter egg while the splash cube is spinning.
_DRAG_SENSITIVITY = 0.012  # radians per pixel of tangential drag
_THROW_DECAY_PER_16MS = 0.965
_THROW_STOP_THRESHOLD_RAD_PER_MS = 0.00004
_MAX_THROW_VELOCITY_RAD_PER_MS = 0.12
_THROW_VELOCITY_SAMPLES = 6


def _tangential_drag_angle_delta(
    center_x: float,
    center_y: float,
    pos_x: float,
    pos_y: float,
    prev_x: float,
    prev_y: float,
    *,
    sensitivity: float = _DRAG_SENSITIVITY,
) -> float:
    """Map a screen drag to spin-axis rotation using tangential motion."""
    rx = pos_x - center_x
    ry = pos_y - center_y
    dx = pos_x - prev_x
    dy = pos_y - prev_y
    radius = math.hypot(rx, ry)
    if radius < 1e-3:
        return (dx - dy) * sensitivity
    tangent_x = -ry / radius
    tangent_y = rx / radius
    tangential_px = dx * tangent_x + dy * tangent_y
    return tangential_px * sensitivity


class RotatingQubeCube(QWidget):
    """Isometric Qube logo cube with tutorial-style (3, 1, 1) axis rotation."""

    def __init__(
        self,
        logo_path: str | Path | None = None,
        *,
        size: int = 140,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._size = size
        self._spin_angle = 0.0
        self._rotating = False
        self._throw_velocity_rad_per_ms = 0.0
        self._dragging = False
        self._last_drag_pos: QPointF | None = None
        self._last_drag_mono: float | None = None
        self._drag_velocity_samples: deque[tuple[float, float]] = deque(
            maxlen=_THROW_VELOCITY_SAMPLES,
        )
        # logo_path retained for future texture use; vector style is primary.
        _ = logo_path
        self.setFixedSize(size, size)

    def set_rotating(self, rotating: bool) -> None:
        self._rotating = rotating
        if not rotating:
            self._reset_throw_state()
            self._spin_angle = 0.0
        self.update()

    def _reset_throw_state(self) -> None:
        self._throw_velocity_rad_per_ms = 0.0
        self._dragging = False
        self._last_drag_pos = None
        self._last_drag_mono = None
        self._drag_velocity_samples.clear()
        if self.mouseGrabber() is self:
            self.releaseMouse()

    def _auto_step_rad(self, delta_ms: float) -> float:
        return delta_ms * _ROTATION_DEG_PER_MS * (math.pi / 180.0)

    def advance(self, delta_ms: float = 16.67) -> None:
        if not self._rotating:
            return
        if not self._dragging:
            auto_step = self._auto_step_rad(delta_ms)
            self._spin_angle += auto_step + self._throw_velocity_rad_per_ms * delta_ms
            if abs(self._throw_velocity_rad_per_ms) > _THROW_STOP_THRESHOLD_RAD_PER_MS:
                decay = _THROW_DECAY_PER_16MS ** (delta_ms / 16.67)
                self._throw_velocity_rad_per_ms *= decay
                if abs(self._throw_velocity_rad_per_ms) < _THROW_STOP_THRESHOLD_RAD_PER_MS:
                    self._throw_velocity_rad_per_ms = 0.0
        self.update()

    def _interaction_enabled(self) -> bool:
        return self._rotating

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if (
            event is None
            or not self._interaction_enabled()
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return
        self._dragging = True
        self._throw_velocity_rad_per_ms = 0.0
        self._drag_velocity_samples.clear()
        self._last_drag_pos = event.position()
        self._last_drag_mono = time.monotonic()
        self.grabMouse()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            super().mouseMoveEvent(event)
            return
        if self._dragging and self._last_drag_pos is not None:
            pos = event.position()
            center_x = self.width() / 2.0
            center_y = self.height() / 2.0
            angle_delta = _tangential_drag_angle_delta(
                center_x,
                center_y,
                pos.x(),
                pos.y(),
                self._last_drag_pos.x(),
                self._last_drag_pos.y(),
            )
            self._spin_angle += angle_delta
            now = time.monotonic()
            if self._last_drag_mono is not None:
                delta_ms = max((now - self._last_drag_mono) * 1000.0, 0.001)
                self._drag_velocity_samples.append((angle_delta, delta_ms))
            self._last_drag_pos = pos
            self._last_drag_mono = now
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if (
            event is None
            or not self._dragging
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mouseReleaseEvent(event)
            return
        self._dragging = False
        self._last_drag_pos = None
        self._last_drag_mono = None
        if self.mouseGrabber() is self:
            self.releaseMouse()
        if self._drag_velocity_samples:
            total_angle = sum(angle for angle, _ in self._drag_velocity_samples)
            total_ms = sum(ms for _, ms in self._drag_velocity_samples)
            if total_ms > 0.0:
                velocity = total_angle / total_ms
                velocity = max(
                    -_MAX_THROW_VELOCITY_RAD_PER_MS,
                    min(_MAX_THROW_VELOCITY_RAD_PER_MS, velocity),
                )
                self._throw_velocity_rad_per_ms = velocity
        self._drag_velocity_samples.clear()
        event.accept()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        spin = self._spin_angle if self._rotating else 0.0
        paint_qube_wireframe_cube(
            painter,
            center_x=self.width() / 2.0,
            center_y=self.height() / 2.0,
            fit_radius=fit_radius_for_widget_side(side),
            spin_angle=spin,
        )
        painter.end()


# Backwards-compatible alias used by older splash code paths.
RotatingLogoLabel = RotatingQubeCube


# Frameless splash cards: QSS rounded chrome; pixels outside the radius stay
# transparent so the translucent shell shows the desktop through (Qt-native path).
_SPLASH_CARD_SURFACE_QSS = """
    background-color: #12151f;
    border: none;
    border-radius: 16px;
"""


class _SplashCardChrome(QWidget):
    """Rounded splash card surface — QSS only; no custom paint (avoids corner seams)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet(_SPLASH_CARD_SURFACE_QSS.strip())


class QubeFirstRunSplitSplash(QWidget):
    """First-run split view: branded splash left, consent panel slot right."""

    _LEFT_WIDTH = 300
    _RIGHT_MIN_WIDTH = 620
    _PROCESSING_LEFT_WIDTH = 400
    _PROCESSING_MIN_WIDTH = 400
    _PROCESSING_MIN_HEIGHT = 520

    def __init__(
        self,
        logo_path: str | Path | None = None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("QubeFirstRunSplitSplashRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(False)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.card = _SplashCardChrome()
        self.card.setObjectName("QubeFirstRunSplitSplash")

        outer = QHBoxLayout(self.card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._left = QWidget(parent=self.card)
        self._left.setObjectName("QubeFirstRunSplashLeft")
        self._left.setFixedWidth(self._LEFT_WIDTH)
        self._left.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        self._left_layout = QVBoxLayout(self._left)
        self._left_layout.setContentsMargins(24, 28, 20, 28)
        self._left_layout.setSpacing(10)
        self._left_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Top stretch stays collapsed during consent; balanced in processing layout.
        self._left_layout.addStretch(0)

        self.logo = RotatingQubeCube(logo_path=logo_path, size=184, parent=self._left)
        self._left_layout.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignHCenter)
        self._left_layout.addSpacing(4)

        self.title = QLabel("Qube")
        self.title.setObjectName("QubeSplashTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(self.title.font())
        title_font.setPointSize(22)
        title_font.setWeight(QFont.Weight.ExtraBold)
        self.title.setFont(title_font)
        self._left_layout.addWidget(self.title)

        self.hint = QLabel("Choose models on the right to begin.")
        self.hint.setObjectName("QubeFirstRunSplashHint")
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._left_layout.addWidget(self.hint)

        self._left_layout.addSpacing(8)

        self._load_section = QWidget()
        load_layout = QVBoxLayout(self._load_section)
        load_layout.setContentsMargins(0, 0, 0, 0)
        load_layout.setSpacing(8)

        self.steps = SplashStepList(SPLASH_STEP_LABELS)
        self.steps.hide()
        load_layout.addWidget(self.steps)

        self.progress = _SplashChunkProgressBar()
        load_layout.addWidget(self.progress)

        self.detail = QLabel()
        self.detail.setObjectName("QubeSplashDetail")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.detail.hide()
        load_layout.addWidget(self.detail)

        self._left_layout.addWidget(self._load_section)
        self._left_layout.addStretch(1)
        outer.addWidget(self._left)

        self.divider = QFrame()
        self.divider.setObjectName("QubeFirstRunSplitDivider")
        self.divider.setFrameShape(QFrame.Shape.VLine)
        self.divider.setFixedWidth(1)
        outer.addWidget(self.divider)

        self._consent_host = QWidget(parent=self.card)
        self._consent_host.setObjectName("QubeFirstRunConsentHost")
        self._consent_host.setMinimumWidth(self._RIGHT_MIN_WIDTH)
        consent_layout = QVBoxLayout(self._consent_host)
        consent_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._consent_host, 1)

        root_layout.addWidget(self.card)
        self.setMinimumSize(self._LEFT_WIDTH + self._RIGHT_MIN_WIDTH + 1, 700)
        self._apply_styles()
        self.enter_consent_layout()

    def set_consent_widget(self, widget: QWidget) -> None:
        layout = self._consent_host.layout()
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        layout.addWidget(widget)

    def dismiss_consent_side(self) -> None:
        self._consent_host.hide()
        self.divider.hide()
        self._left.setFixedWidth(self._PROCESSING_LEFT_WIDTH)
        self.card.setFixedSize(self._PROCESSING_MIN_WIDTH, self._PROCESSING_MIN_HEIGHT)
        self.setFixedSize(self._PROCESSING_MIN_WIDTH, self._PROCESSING_MIN_HEIGHT)
        self.setMinimumSize(self._PROCESSING_MIN_WIDTH, self._PROCESSING_MIN_HEIGHT)
        self.enter_processing_layout()

    def enter_consent_layout(self) -> None:
        """Configuration split view: logo + title and hint on the left."""
        self.logo.show()
        self.logo.set_rotating(False)
        self._load_section.hide()
        self.hint.setText("Choose models on the right to begin.")
        last_index = self._left_layout.count() - 1
        self._left_layout.setStretch(0, 1)
        self._left_layout.setStretch(last_index, 1)
        self.adjustSize()

    def enter_processing_layout(self) -> None:
        """Downloading view: logo, progress, and centered branding."""
        self.logo.show()
        self._load_section.show()
        self.hint.setText("Downloading and starting Qube…")
        last_index = self._left_layout.count() - 1
        self._left_layout.setStretch(0, 1)
        self._left_layout.setStretch(last_index, 1)
        self.adjustSize()

    def set_logo_rotating(self, rotating: bool) -> None:
        self.logo.set_rotating(rotating)

    def advance_logo(self, delta_ms: float) -> None:
        self.logo.advance(delta_ms)

    def set_progress_percent(self, percent: int) -> None:
        self.progress.setValue(max(0, min(100, int(percent))))

    def set_active_step(self, index: int) -> None:
        if 0 <= index < len(SPLASH_STEP_LABELS):
            self.steps.show()
            self.steps.set_active(index)

    def complete_step(self, index: int) -> None:
        if 0 <= index < len(SPLASH_STEP_LABELS):
            self.steps.show()
            self.steps.mark_done_through(index)

    def set_download_detail(self, text: str) -> None:
        if text.strip():
            self.detail.setText(text)
            self.detail.show()
        else:
            self.detail.clear()
            self.detail.hide()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#QubeFirstRunSplitSplashRoot {
                background: transparent;
            }
            QWidget#QubeFirstRunSplitSplash {
            """
            + _SPLASH_CARD_SURFACE_QSS
            + """
            }
            QWidget#QubeFirstRunSplashLeft,
            QWidget#QubeFirstRunConsentHost {
                background: transparent;
            }
            QFrame#QubeFirstRunSplitDivider {
                background: rgba(255, 255, 255, 0.12);
                border: none;
                max-width: 1px;
            }
            QLabel#QubeSplashTitle {
                color: #f8fafc;
            }
            QLabel#QubeFirstRunSplashHint {
                color: rgba(148, 163, 184, 0.9);
                font-size: 12px;
            }
            QLabel#QubeSplashDetail {
                color: rgba(148, 163, 184, 0.95);
                font-size: 10px;
                line-height: 1.35;
            }
            """
        )


class QubeSplashCard(QWidget):
    """Compact floating startup card: logo, circle spinner + step list, chunked progress."""

    def __init__(
        self,
        logo_path: str | Path | None = None,
        *,
        compact: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("QubeSplashCardRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.card = _SplashCardChrome()
        self.card.setObjectName("QubeSplashCard")
        self.card.setFixedWidth(440 if compact else 720)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(28, 24, 28, 22)
        card_layout.setSpacing(0)

        self.logo = QLabel()
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_width = 88 if compact else 300
        if logo_path is not None:
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                self.logo.setPixmap(
                    pix.scaledToWidth(logo_width, Qt.TransformationMode.SmoothTransformation)
                )
        card_layout.addWidget(self.logo)
        card_layout.addSpacing(10)

        self.title = QLabel("Qube")
        self.title.setObjectName("QubeSplashTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(self.title.font())
        title_font.setPointSize(22)
        title_font.setWeight(QFont.Weight.ExtraBold)
        self.title.setFont(title_font)
        card_layout.addWidget(self.title)
        card_layout.addSpacing(16)

        load_row = QHBoxLayout()
        load_row.setSpacing(14)
        load_row.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.spinner = SplashCircleSpinner(size=40)
        load_row.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignTop)

        self.steps = SplashStepList(SPLASH_STEP_LABELS)
        load_row.addWidget(self.steps, 1)
        card_layout.addLayout(load_row)
        card_layout.addSpacing(14)

        self.progress = _SplashChunkProgressBar()
        card_layout.addWidget(self.progress)
        card_layout.addSpacing(6)

        self.detail = QLabel()
        self.detail.setObjectName("QubeSplashDetail")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.detail.hide()
        card_layout.addWidget(self.detail)

        outer.addWidget(self.card)
        self._apply_styles()

    def set_progress_percent(self, percent: int) -> None:
        self.progress.setValue(max(0, min(100, int(percent))))

    def set_active_step(self, index: int) -> None:
        if 0 <= index < len(SPLASH_STEP_LABELS):
            self.steps.set_active(index)

    def complete_step(self, index: int) -> None:
        if 0 <= index < len(SPLASH_STEP_LABELS):
            self.steps.mark_done_through(index)

    def set_download_detail(self, text: str) -> None:
        if text.strip():
            self.detail.setText(text)
            self.detail.show()
        else:
            self.detail.clear()
            self.detail.hide()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#QubeSplashCardRoot {
                background: transparent;
            }
            QWidget#QubeSplashCard {
            """
            + _SPLASH_CARD_SURFACE_QSS
            + """
            }
            QLabel#QubeSplashTitle {
                color: #f8fafc;
            }
            QLabel#QubeSplashDetail {
                color: rgba(148, 163, 184, 0.95);
                font-size: 10px;
                line-height: 1.35;
            }
            """
        )
