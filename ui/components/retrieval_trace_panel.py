"""Retrieval trace viewer widget."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from core.knowledge.retrieval_trace_reader import (
    format_retrieval_trace_summary,
    read_last_retrieval_trace,
)


class RetrievalTracePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel("Last retrieval trace")
        self._title.setWordWrap(True)
        self._hint = QLabel(
            "Retrieval records are always saved for v2 turns. "
            "Enable web search audit logging for verbose JSONL traces."
        )
        self._hint.setWordWrap(True)
        self._body = QPlainTextEdit()
        self._body.setReadOnly(True)
        self._body.setPlaceholderText("No retrieval trace recorded yet.")
        layout.addWidget(self._title)
        layout.addWidget(self._hint)
        layout.addWidget(self._body)
        self.refresh()

    def refresh(
        self,
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
    ) -> None:
        trace = read_last_retrieval_trace(session_id=session_id, turn_id=turn_id)
        if trace is None:
            self._body.setPlainText(
                "No retrieval trace recorded yet.\n\n"
                "Run a turn with External knowledge pipeline (v2) enabled. "
                "Audit logging adds verbose JSONL detail under ~/.qube/logs/web_search.log."
            )
            return
        lines = [format_retrieval_trace_summary(trace)]
        if trace.get("preset_id"):
            lines.append(f"Preset: {trace.get('preset_id')}")
        if trace.get("retrieval_profile"):
            lines.append(f"Retrieval profile: {trace.get('retrieval_profile')}")
        fingerprint = trace.get("context_fingerprint")
        if isinstance(fingerprint, dict) and fingerprint:
            adapters = fingerprint.get("adapter_filter") or []
            if adapters:
                lines.append(f"Adapter filter: {', '.join(str(a) for a in adapters)}")
        warnings = trace.get("warnings") or []
        if warnings:
            lines.append(f"Warnings: {'; '.join(str(w) for w in warnings)}")
        diag = trace.get("relevance_diag") or {}
        if diag:
            bits = []
            for key in ("ranking_profile", "query_planner", "retrieval_profile", "http_summary"):
                if key in diag and key != "http_summary":
                    bits.append(f"{key}={diag[key]}")
            if bits:
                lines.append("Diagnostics: " + ", ".join(bits))
        stages = trace.get("pipeline_stages") or []
        if stages:
            lines.append(f"Pipeline stages: {len(stages)}")
        self._body.setPlainText("\n".join(lines))
