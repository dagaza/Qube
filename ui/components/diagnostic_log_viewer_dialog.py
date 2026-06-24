"""Read-only in-app viewer for diagnostic log files."""

from __future__ import annotations

import logging
from collections.abc import Callable

import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from core.app_log_sink import app_log_env_override
from core.llm_debug_sink import llm_debug_log_env_override
from core.web_search_audit import web_search_audit_log_env_override
from core.diagnostic_logs import (
    DiagnosticLogSpec,
    describe_log_status,
    diagnostic_log_recording_enabled,
    open_path_in_system,
    read_log_tail,
)
from mcp.routing_debug import routing_debug_log_env_override
from ui.components.toggle import PrestigeToggle

logger = logging.getLogger("Qube.UI.DiagnosticLogViewer")


class DiagnosticLogViewerDialog(QDialog):
    def __init__(
        self,
        spec: DiagnosticLogSpec,
        parent=None,
        *,
        is_dark: bool | None = None,
        on_recording_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = getattr(parent.window() if parent else None, "_is_dark_theme", True)
        self._spec = spec
        self._is_dark = is_dark
        self._on_recording_toggle = on_recording_toggle
        self._recording_toggle: PrestigeToggle | None = None
        self._recording_note_lbl: QLabel | None = None

        self.setWindowTitle(spec.title)
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(820, 620)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._refresh)

        self._build_ui()
        self._apply_theme_styles()
        self._refresh()

    def refresh_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self._apply_theme_styles()

    def sync_recording_toggle(self) -> None:
        if self._recording_toggle is None:
            return

        env_override = None
        if self._spec.id == "routing_debug":
            env_override = routing_debug_log_env_override()
        elif self._spec.id == "web_search_audit":
            env_override = web_search_audit_log_env_override()
        elif self._spec.id == "app_log":
            env_override = app_log_env_override()
        elif self._spec.id == "llm_debug":
            env_override = llm_debug_log_env_override()

        self._recording_toggle.blockSignals(True)
        self._recording_toggle.setChecked(diagnostic_log_recording_enabled(self._spec.id))
        self._recording_toggle.blockSignals(False)

        if env_override is None:
            self._recording_toggle.setEnabled(True)
            if self._recording_note_lbl is not None:
                self._recording_note_lbl.hide()
        else:
            self._recording_toggle.setEnabled(False)
            self._recording_toggle.setChecked(bool(env_override))
            if self._recording_note_lbl is not None:
                self._recording_note_lbl.setText(
                    "Recording for this log is controlled by how Qube was launched. "
                    "Use Settings here when no launch override is present."
                )
                self._recording_note_lbl.show()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        self.container = QFrame()
        self.container.setObjectName("DiagnosticLogViewerContainer")
        root = QVBoxLayout(self.container)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        self.header_title_lbl = QLabel(self._spec.title.upper())
        self.header_title_lbl.setObjectName("DiagnosticLogViewerTitle")
        self.path_lbl = QLabel(str(self._spec.path_fn()))
        self.path_lbl.setObjectName("DiagnosticLogViewerPath")
        self.path_lbl.setWordWrap(True)
        title_col.addWidget(self.header_title_lbl)
        title_col.addWidget(self.path_lbl)
        self.close_btn = QPushButton()
        self.close_btn.setObjectName("DiagnosticLogViewerClose")
        self.close_btn.setFixedSize(32, 32)
        self.close_btn.setFlat(True)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(self.close)
        header_row.addLayout(title_col, 1)
        header_row.addWidget(
            self.close_btn,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        root.addLayout(header_row)

        if self._spec.description:
            desc = QLabel(self._spec.description)
            desc.setWordWrap(True)
            desc.setObjectName("DiagnosticLogViewerDescription")
            root.addWidget(desc)

        if self._spec.supports_recording_toggle:
            recording_row = QHBoxLayout()
            recording_row.setSpacing(10)
            recording_label = self._spec.recording_toggle_label or "Record entries to this log"
            recording_lbl = QLabel(recording_label)
            recording_lbl.setWordWrap(True)
            recording_lbl.setObjectName("DiagnosticLogViewerDescription")
            self._recording_toggle = PrestigeToggle()
            self._recording_toggle.setToolTip(
                "When enabled, Qube appends one JSON line per chat turn on your next message."
            )
            self._recording_toggle.toggled.connect(self._on_recording_toggle_changed)
            recording_row.addWidget(
                self._recording_toggle,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            )
            recording_row.addWidget(recording_lbl, stretch=1)
            root.addLayout(recording_row)

            self._recording_note_lbl = QLabel("")
            self._recording_note_lbl.setWordWrap(True)
            self._recording_note_lbl.setObjectName("DiagnosticLogViewerNote")
            self._recording_note_lbl.hide()
            root.addWidget(self._recording_note_lbl)
            self.sync_recording_toggle()

        elif self._spec.note:
            note = QLabel(self._spec.note)
            note.setWordWrap(True)
            note.setObjectName("DiagnosticLogViewerNote")
            root.addWidget(note)

        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("DiagnosticLogViewerStatus")
        self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl)

        self._text = QPlainTextEdit()
        self._text.setObjectName("DiagnosticLogViewerText")
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.setMaximumBlockCount(8000)
        font = QFont("Cascadia Mono")
        if not font.exactMatch():
            font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setPointSize(10)
        self._text.setFont(font)
        self._text.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._text, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Reload the latest lines from disk.")
        self.refresh_btn.clicked.connect(self._refresh)
        self.live_cb = QCheckBox("Live tail (2s)")
        self.live_cb.setToolTip("Automatically refresh while this window is open.")
        self.live_cb.toggled.connect(self._on_live_toggled)
        self.external_btn = QPushButton("Open externally")
        self.external_btn.setToolTip("Open this log in your default system editor.")
        self.external_btn.clicked.connect(self._on_open_external)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.live_cb)
        controls.addWidget(self.external_btn)
        controls.addStretch()
        root.addLayout(controls)

        outer.addWidget(self.container)

    def _apply_theme_styles(self) -> None:
        is_dark = self._is_dark
        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.12)" if is_dark else "#cbd5e1"
        surface = "#181825" if is_dark else "#f8fafc"
        note_bg = "#45475a" if is_dark else "#e2e8f0"

        self.container.setStyleSheet(
            f"""
            QFrame#DiagnosticLogViewerContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel#DiagnosticLogViewerTitle {{
                color: {accent};
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 2px;
            }}
            QLabel#DiagnosticLogViewerPath,
            QLabel#DiagnosticLogViewerDescription,
            QLabel#DiagnosticLogViewerStatus {{
                color: {fg};
                font-size: 12px;
            }}
            QLabel#DiagnosticLogViewerNote {{
                background: {note_bg};
                color: {fg};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
            }}
            QPlainTextEdit#DiagnosticLogViewerText {{
                background: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 12px 14px;
                selection-background-color: {accent};
            }}
            QPushButton#DiagnosticLogViewerClose {{
                background: transparent;
                color: {fg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 0px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }}
            QPushButton#DiagnosticLogViewerClose:hover {{
                background: rgba(255, 255, 255, 0.06);
            }}
            """
        )
        self.close_btn.setIcon(qta.icon("fa5s.times", color=fg))
        self.close_btn.setIconSize(QSize(14, 14))
        btn_style = f"""
            QPushButton {{
                padding: 10px 16px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 12px;
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """
        for btn in (self.refresh_btn, self.external_btn):
            btn.setStyleSheet(btn_style)

    def _on_recording_toggle_changed(self, enabled: bool) -> None:
        if self._on_recording_toggle is not None:
            self._on_recording_toggle(enabled)

    def _refresh(self) -> None:
        path = self._spec.path_fn()
        self._text.setPlainText(read_log_tail(path))
        self.status_lbl.setText(describe_log_status(self._spec))

    def _on_live_toggled(self, enabled: bool) -> None:
        if enabled:
            self._poll_timer.start()
            self._refresh()
        else:
            self._poll_timer.stop()

    def _on_open_external(self) -> None:
        if not open_path_in_system(self._spec.path_fn()):
            logger.warning("Could not open diagnostic log externally: %s", self._spec.path_fn())

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        super().closeEvent(event)
