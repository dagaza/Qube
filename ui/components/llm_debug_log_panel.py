"""
Developer-only panel: tail of logs/llm_debug.log (Qube.NativeLLM.Debug).

Shown only when QUBE_LLM_LOG_UI=1. Not part of chat UI.
"""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

_ROOT = Path(__file__).resolve().parent.parent.parent


def _log_path() -> Path:
    try:
        from core.llm_debug_sink import default_llm_debug_log_path

        return default_llm_debug_log_path()
    except Exception:
        return _ROOT / "logs" / "llm_debug.log"


def _read_last_lines(path: Path, max_lines: int = 500) -> str:
    if not path.is_file():
        return f"(no file yet: {path})"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(read error: {e})"
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


class LLMDebugLogPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("LLMDebugLogPanel")
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._refresh)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 0)
        head = QLabel("LLM debug log (developer — QUBE_LLM_LOG_UI)")
        head.setObjectName("ViewSubtitle")
        layout.addWidget(head)

        row = QHBoxLayout()
        self._btn = QPushButton("Refresh")
        self._btn.clicked.connect(self._refresh)
        row.addWidget(self._btn)
        self._live = QCheckBox("Live tail (2s)")
        self._live.toggled.connect(self._on_live_toggled)
        row.addWidget(self._live)
        row.addStretch()
        layout.addLayout(row)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(8000)
        mono = QFont("monospace")
        if not mono.exactMatch():
            mono = QFont("Consolas", 10)
        self._text.setFont(mono)
        self._text.setPlaceholderText("logs/llm_debug.log — enable native LLM debug env vars to populate.")
        layout.addWidget(self._text)

        self._refresh()

    def _on_live_toggled(self, on: bool) -> None:
        if on:
            self._poll_timer.start(2000)
        else:
            self._poll_timer.stop()

    def _refresh(self) -> None:
        body = _read_last_lines(_log_path(), 500)
        self._text.setPlainText(body)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())
