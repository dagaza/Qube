"""Compact spinner + progress bar row for library ingest and reindex operations."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from ui.splash_widget import SplashCircleSpinner


class IngestProgressRow(QWidget):
    """Spinner, optional detail label, and thin progress bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("IngestProgressRow")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("IngestProgressDetail")
        self.detail_label.setWordWrap(True)
        self.detail_label.hide()

        bar_row = QWidget()
        bar_layout = QHBoxLayout(bar_row)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(8)

        self.spinner = SplashCircleSpinner(size=16, parent=bar_row)
        self.spinner.hide()

        self.progress = QProgressBar()
        self.progress.setObjectName("IngestProgressBar")
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)

        bar_layout.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignVCenter)
        bar_layout.addWidget(self.progress, 1)

        outer.addWidget(self.detail_label)
        outer.addWidget(bar_row)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(16)
        self._spinner_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._spinner_timer.timeout.connect(self._advance_spinner)

    def _advance_spinner(self) -> None:
        self.spinner.advance(float(self._spinner_timer.interval()))

    def apply_theme(self, is_dark: bool) -> None:
        self.spinner.apply_theme(is_dark)
        if is_dark:
            self.detail_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        else:
            self.detail_label.setStyleSheet("color: #64748b; font-size: 11px;")

    def begin(self, *, detail: str = "") -> None:
        self.progress.setValue(0)
        self.set_detail(detail)
        self.spinner.show()
        self._advance_spinner()
        if not self._spinner_timer.isActive():
            self._spinner_timer.start()

    def set_detail(self, detail: str) -> None:
        text = str(detail or "").strip()
        if text:
            self.detail_label.setText(text)
            self.detail_label.show()
        else:
            self.detail_label.clear()
            self.detail_label.hide()

    def update_progress(self, percent: int, *, detail: str | None = None) -> None:
        pct = max(0, min(100, int(percent)))
        if pct > 0:
            self._stop_spinner()
        if detail is not None:
            self.set_detail(detail)
        self.progress.setValue(pct)

    def finish(self) -> None:
        self._stop_spinner()
        self.detail_label.clear()
        self.detail_label.hide()
        self.progress.setValue(0)

    def _stop_spinner(self) -> None:
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()
        self.spinner.hide()
