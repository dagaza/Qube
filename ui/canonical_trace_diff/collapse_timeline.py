"""Collapse risk timeline widget for Canonical Trace Diff."""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.theme.tokens import ResolvedTheme
from ui.canonical_trace_diff.trace_diff_theme import (
    collapse_risk_chip_stylesheet,
    resolve_trace_diff_theme,
)


class _TurnChip(QFrame):
    clicked = pyqtSignal(int)

    def __init__(
        self,
        *,
        turn_index: int,
        risk: str,
        preview: str,
        metrics: dict[str, Any],
        theme: ResolvedTheme,
        selected: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.turn_index = turn_index
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("CollapseTurnChip")
        frame_qss, title_qss, subtitle_qss = collapse_risk_chip_stylesheet(
            theme,
            risk,
            selected=selected,
        )
        self.setStyleSheet(frame_qss)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        title = QLabel(f"T{turn_index} · {str(risk or 'LOW').upper()}")
        title.setStyleSheet(title_qss)
        layout.addWidget(title)

        subtitle = QLabel(preview[:56] or "—")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(subtitle_qss)
        layout.addWidget(subtitle)

        tool = (
            f"Turn {turn_index} · {risk}\n"
            f"prompt={metrics.get('collapse_prompt_length', 0)} chars · "
            f"output={metrics.get('collapse_output_length', 0)} chars\n"
            f"deg={metrics.get('collapse_degeneration_score', 0):.2f} · "
            f"halluc={metrics.get('collapse_hallucination_score', 0):.2f} · "
            f"drift={metrics.get('collapse_format_drift_score', 0):.2f}\n"
            f"score={metrics.get('collapse_score', 0):.2f} · "
            f"rewrite={metrics.get('collapse_rewrite_confidence', 0):.2f}"
        )
        self.setToolTip(tool)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.turn_index)
        super().mouseReleaseEvent(event)


class CollapseTimelineWidget(QFrame):
    """Horizontal timeline of per-turn collapse risk for one backend session."""

    turn_selected = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CollapseTimelineWidget")
        self._is_dark = True
        self._chips: dict[int, _TurnChip] = {}
        self._selected_turn: int | None = None
        self._entries: list[dict[str, Any]] = []
        self._backend_label = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._title = QLabel("Collapse timeline")
        self._title.setObjectName("ViewSubtitle")
        outer.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(92)

        self._row_host = QWidget()
        self._row = QHBoxLayout(self._row_host)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(8)
        self._row.addStretch()
        scroll.setWidget(self._row_host)
        outer.addWidget(scroll)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setObjectName("ViewSubtitle")
        outer.addWidget(self._summary)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        if self._entries:
            self.set_timeline(
                self._entries,
                backend_label=self._backend_label,
                selected_turn=self._selected_turn,
            )

    def set_timeline(
        self,
        entries: list[dict[str, Any]] | None,
        *,
        backend_label: str = "",
        selected_turn: int | None = None,
    ) -> None:
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._chips.clear()

        entries = list(entries or [])
        self._entries = entries
        self._backend_label = str(backend_label or "").strip()
        label = self._backend_label
        self._title.setText(
            f"Collapse timeline{(' · ' + label) if label else ''}"
        )
        theme = resolve_trace_diff_theme(is_dark=self._is_dark)

        if not entries:
            self._summary.setText("No turn diagnostics available.")
            self.hide()
            return

        self.show()
        high_turns = [
            int(e.get("collapse_turn_index", idx))
            for idx, e in enumerate(entries)
            if str(e.get("collapse_risk", "LOW")).upper() == "HIGH"
        ]
        medium_turns = [
            int(e.get("collapse_turn_index", idx))
            for idx, e in enumerate(entries)
            if str(e.get("collapse_risk", "LOW")).upper() == "MEDIUM"
        ]
        if high_turns:
            onset = min(high_turns)
            self._summary.setText(
                f"Degradation onset: turn {onset} (HIGH). "
                f"HIGH turns: {', '.join(str(t) for t in high_turns)}."
            )
        elif medium_turns:
            onset = min(medium_turns)
            self._summary.setText(
                f"Elevated risk from turn {onset} (MEDIUM). "
                f"MEDIUM turns: {', '.join(str(t) for t in medium_turns)}."
            )
        else:
            self._summary.setText("All turns LOW risk — no collapse onset detected.")

        for entry in entries:
            turn_index = int(entry.get("collapse_turn_index", 0))
            risk = str(entry.get("collapse_risk", "LOW")).upper()
            preview = str(entry.get("user_message_preview", "") or "")
            chip = _TurnChip(
                turn_index=turn_index,
                risk=risk,
                preview=preview,
                metrics=entry,
                theme=theme,
                selected=selected_turn == turn_index,
            )
            chip.clicked.connect(self.turn_selected.emit)
            self._chips[turn_index] = chip
            self._row.insertWidget(self._row.count() - 1, chip)

        self._selected_turn = selected_turn

    def set_selected_turn(self, turn_index: int | None) -> None:
        if turn_index == self._selected_turn:
            return
        self.set_timeline(
            self._entries,
            backend_label=self._backend_label,
            selected_turn=turn_index,
        )
