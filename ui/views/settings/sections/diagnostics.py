"""Diagnostics settings — technical troubleshooting logs."""

from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from core.diagnostic_logs import iter_diagnostic_logs_by_category
from core.paths import logs_dir
from ui.views.settings.sections.diagnostic_log_ui import (
    add_diagnostic_log_sections,
    add_diagnostic_logs_intro_card,
    ensure_diagnostic_log_host_attrs,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    logs_path = logs_dir()
    add_diagnostic_logs_intro_card(
        host,
        layout,
        is_dark=is_dark,
        hint_text=(
            f"Qube writes rotating debug logs under {logs_path}. "
            "Use the application and skills logs below for crashes, worker errors, "
            "and model load failures. Privacy-sensitive audit logs (LLM, routing, "
            "web search) live under Settings → Privacy & data. "
            "Before sharing excerpts, review redaction flags (@help log redaction)."
        ),
    )

    ensure_diagnostic_log_host_attrs(host)
    add_diagnostic_log_sections(
        host,
        layout,
        iter_diagnostic_logs_by_category("technical"),
        is_dark=is_dark,
    )

    if hasattr(host, "_sync_all_diagnostic_log_recording_toggles"):
        host._sync_all_diagnostic_log_recording_toggles()

    return widget
