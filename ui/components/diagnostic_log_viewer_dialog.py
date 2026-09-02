"""Read-only in-app viewer for diagnostic log files."""

from __future__ import annotations

import logging
from collections.abc import Callable

import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QFont, QShowEvent
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
from core.diagnostic_logs import (
    DiagnosticLogSpec,
    describe_log_status,
    diagnostic_log_recording_enabled,
    open_path_in_system,
    read_log_tail,
)
from core.llm_debug_sink import llm_debug_log_env_override
from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_BODY_LABEL,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_SOURCE_CONTAINER,
)
from core.web_search_audit import web_search_audit_log_env_override
from core.platform.frameless_window import (
    apply_frameless_dialog_chrome,
    configure_frameless_dialog_host,
)
from mcp.routing_debug import routing_debug_log_env_override
from ui.components.prestige_dialog import _resolve_is_dark_from_parent
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
            is_dark = _resolve_is_dark_from_parent(parent)
        self._spec = spec
        self._is_dark = is_dark
        self._on_recording_toggle = on_recording_toggle
        self._recording_toggle: PrestigeToggle | None = None
        self._recording_note_lbl: QLabel | None = None

        self.setWindowTitle(spec.title)
        self.setModal(False)
        configure_frameless_dialog_host(self)
        self.resize(820, 620)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._refresh)

        self._build_ui()
        self._apply_theme_styles()
        self._refresh()

    def refresh_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        if self._recording_toggle is not None:
            self._recording_toggle.apply_theme(is_dark=is_dark)
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
        theme = theme_for(is_dark=self._is_dark)
        border = theme.border_subtle if theme.is_dark else theme.border
        surface = theme.surface_elevated if theme.is_dark else theme.surface
        hover_bg = with_alpha(theme.text_primary, 0.06 if theme.is_dark else 0.05)
        body_style = theme.style(PRESTIGE_BODY_LABEL, font_size="12px", font_weight="400")

        self.container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="DiagnosticLogViewerContainer",
            )
            + f"""
            QLabel#DiagnosticLogViewerPath,
            QLabel#DiagnosticLogViewerDescription,
            QLabel#DiagnosticLogViewerStatus {{
                {body_style}
            }}
            QLabel#DiagnosticLogViewerNote {{
                background: {theme.surface_pressed};
                color: {theme.text_primary};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
            }}
            QPlainTextEdit#DiagnosticLogViewerText {{
                background: {surface};
                color: {theme.text_primary};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 12px 14px;
                selection-background-color: {theme.link};
            }}
            QPushButton#DiagnosticLogViewerClose {{
                {theme.style(PRESTIGE_GHOST_BUTTON)}
                border-radius: 8px;
                padding: 0px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }}
            QPushButton#DiagnosticLogViewerClose:hover {{
                background: {hover_bg};
            }}
            """
        )
        self.header_title_lbl.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="11px")
        )
        self.close_btn.setIcon(qta.icon("fa5s.times", color=theme.text_primary))
        self.close_btn.setIconSize(QSize(14, 14))
        btn_style = (
            theme.style(PRESTIGE_GHOST_BUTTON)
            + f"""
            QPushButton:hover {{
                background: {hover_bg};
            }}
            """
        )
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

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_dialog_chrome(self)

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        super().closeEvent(event)
