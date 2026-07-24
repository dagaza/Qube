"""Telemetry panel for session integration egress summary (Phase 3 / #61)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from core.integrations.egress_summary import format_session_egress_summary

__all__ = ["SessionEgressPanel"]


class SessionEgressPanel(QFrame):
    """Read-only summary of integration capability calls this session."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("TelemetryCard")
        self._session_id: str | None = None
        self._title = QLabel("Session integrations")
        self._title.setObjectName("TelemetryCardTitle")
        self._body = QLabel("No active session.")
        self._body.setWordWrap(True)
        self._body.setObjectName("SettingsLogDescription")
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._body)

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = str(session_id) if session_id else None
        self.refresh()

    def refresh(self) -> None:
        if not self._session_id:
            self._body.setText(
                "Open a conversation to see integration calls for this session."
            )
            return
        self._body.setText(
            format_session_egress_summary(
                self._session_id,
                empty_message="No integration calls recorded this session yet.",
            )
        )
