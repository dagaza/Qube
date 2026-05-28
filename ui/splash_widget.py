"""Branded splash card content used by :mod:`ui.splash_overlay`."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

# Startup steps shown beside the spinner (index must match splash_overlay phase order).
SPLASH_STEP_LABELS: tuple[str, ...] = (
    "nomic-embed-text-v1.5.Q4_K_M.gguf",
    "Document store & qube_data.db",
    "Audio, STT, native LLM, sidecar",
    "Memory enrichment workers",
    "Main window UI",
    "Service connections & sync",
    "Language model (optional)",
    "kokoro-v1.0.onnx & audio runtime",
)


def resolve_splash_logo_path(repo_root: Path | None = None) -> Path | None:
    """Return the best available logo for the splash card, or ``None``."""
    root = repo_root or Path(__file__).resolve().parent.parent
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

    def advance(self, delta_ms: float = 16.67) -> None:
        self._angle_deg = (self._angle_deg + delta_ms * 0.35) % 360.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        margin = 4.0
        rect = QRectF(margin, margin, side - 2 * margin, side - 2 * margin)

        pen_track = QPen(self._track)
        pen_track.setWidthF(3.0)
        pen_track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_track)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

        pen_arc = QPen(self._arc)
        pen_arc.setWidthF(3.0)
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.card = QWidget()
        self.card.setObjectName("QubeSplashCard")
        self.card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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

        self.progress = QProgressBar()
        self.progress.setObjectName("QubeSplashChunkProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setFixedHeight(6)
        card_layout.addWidget(self.progress)

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

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#QubeSplashCardRoot {
                background: transparent;
            }
            QWidget#QubeSplashCard {
                background: #12151f;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 16px;
            }
            QLabel#QubeSplashTitle {
                color: #f8fafc;
            }
            QProgressBar#QubeSplashChunkProgress {
                background: rgba(255, 255, 255, 0.08);
                border: none;
                border-radius: 3px;
                color: rgba(148, 163, 184, 0.9);
                font-size: 10px;
                text-align: center;
            }
            QProgressBar#QubeSplashChunkProgress::chunk {
                background: #8b5cf6;
                border-radius: 3px;
            }
            """
        )
