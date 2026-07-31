"""Shared diagnostic log cards for Settings → Diagnostics and Privacy & data."""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.diagnostic_logs import DiagnosticLogSpec, describe_log_status
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.toggle import PrestigeToggle
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_settings_full_width_row,
    add_subsection_to_form,
    make_settings_action_row,
    make_settings_hint,
)


def ensure_diagnostic_log_host_attrs(host) -> None:
    """Initialize shared diagnostic log widget registries without resetting existing entries."""
    if not hasattr(host, "diagnostic_log_status_labels"):
        host.diagnostic_log_status_labels = {}
    if not hasattr(host, "diagnostic_log_view_buttons"):
        host.diagnostic_log_view_buttons = {}
    if not hasattr(host, "diagnostic_log_clear_buttons"):
        host.diagnostic_log_clear_buttons = {}
    if not hasattr(host, "diagnostic_log_recording_toggles"):
        host.diagnostic_log_recording_toggles = {}
    if not hasattr(host, "diagnostic_log_recording_env_notes"):
        host.diagnostic_log_recording_env_notes = {}
    if not hasattr(host, "diagnostic_log_redaction_toggles"):
        host.diagnostic_log_redaction_toggles = {}
    if not hasattr(host, "diagnostic_log_redaction_env_notes"):
        host.diagnostic_log_redaction_env_notes = {}


def add_diagnostic_logs_intro_card(
    host,
    layout: QVBoxLayout,
    *,
    is_dark: bool,
    hint_text: str,
) -> None:
    """Intro card with folder hint and Open logs folder button."""
    from core.paths import logs_dir

    logs_path = logs_dir()
    logs_intro_card, logs_intro_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    logs_form = add_settings_card_form(logs_intro_card_layout)
    add_subsection_to_form(logs_form, "Diagnostic logs", anchor="logs")

    host.diagnostic_logs_hint_lbl = make_settings_hint(hint_text)
    add_settings_full_width_row(logs_form, host.diagnostic_logs_hint_lbl)

    host.open_logs_folder_btn = QPushButton("Open logs folder")
    apply_brand_primary(host.open_logs_folder_btn, icon_name="fa5s.folder-open")
    host.open_logs_folder_btn.setToolTip(f"Reveal {logs_path} in your file manager.")
    host.open_logs_folder_btn.clicked.connect(host._on_open_logs_folder_clicked)
    add_settings_full_width_row(
        logs_form, make_settings_action_row(host.open_logs_folder_btn)
    )
    layout.addWidget(logs_intro_card)


def add_diagnostic_log_section(
    host,
    layout: QVBoxLayout,
    spec: DiagnosticLogSpec,
    *,
    is_dark: bool,
) -> None:
    ensure_diagnostic_log_host_attrs(host)

    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    card_form = add_settings_card_form(card_layout)
    add_subsection_to_form(card_form, spec.title, anchor=spec.id)

    desc = QLabel(spec.description)
    desc.setWordWrap(True)
    desc.setObjectName("SettingsLogDescription")
    add_settings_full_width_row(card_form, desc)

    if spec.note:
        note = QLabel(spec.note)
        note.setWordWrap(True)
        note.setObjectName("SettingsLogNote")
        add_settings_full_width_row(card_form, note)

    if spec.supports_recording_toggle:
        recording_row = QWidget()
        recording_layout = QHBoxLayout(recording_row)
        recording_layout.setContentsMargins(0, 0, 0, 0)
        recording_layout.setSpacing(10)
        recording_lbl = QLabel(
            spec.recording_toggle_label or "Record entries to this log"
        )
        recording_lbl.setWordWrap(True)
        recording_lbl.setObjectName("SettingsLogDescription")
        toggle = PrestigeToggle()
        toggle.setToolTip(
            "When enabled, Qube writes new entries to this log file. "
            "Existing lines stay on disk until you clear the log."
        )
        log_id = spec.id
        toggle.toggled.connect(
            lambda checked, lid=log_id: host._on_diagnostic_log_recording_toggled(
                lid, checked
            )
        )
        host.diagnostic_log_recording_toggles[spec.id] = toggle
        recording_layout.addWidget(
            toggle, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        recording_layout.addWidget(recording_lbl, stretch=1)
        add_settings_full_width_row(card_form, recording_row)

        env_note = QLabel("")
        env_note.setWordWrap(True)
        env_note.setObjectName("SettingsLogNote")
        env_note.hide()
        host.diagnostic_log_recording_env_notes[spec.id] = env_note
        add_settings_full_width_row(card_form, env_note)

    if spec.supports_redaction_toggle:
        redaction_row = QWidget()
        redaction_layout = QHBoxLayout(redaction_row)
        redaction_layout.setContentsMargins(0, 0, 0, 0)
        redaction_layout.setSpacing(10)
        redaction_lbl = QLabel(
            spec.redaction_toggle_label or "Redact sensitive fields in new log entries"
        )
        redaction_lbl.setWordWrap(True)
        redaction_lbl.setObjectName("SettingsLogDescription")
        redaction_toggle = PrestigeToggle()
        redaction_toggle.setToolTip(
            "When enabled, new log entries omit or hash sensitive query text. "
            "Existing lines stay on disk until you clear the log."
        )
        log_id = spec.id
        redaction_toggle.toggled.connect(
            lambda checked, lid=log_id: host._on_diagnostic_log_redaction_toggled(
                lid, checked
            )
        )
        host.diagnostic_log_redaction_toggles[spec.id] = redaction_toggle
        redaction_layout.addWidget(
            redaction_toggle,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        redaction_layout.addWidget(redaction_lbl, stretch=1)
        add_settings_full_width_row(card_form, redaction_row)

        redaction_env_note = QLabel("")
        redaction_env_note.setWordWrap(True)
        redaction_env_note.setObjectName("SettingsLogNote")
        redaction_env_note.hide()
        host.diagnostic_log_redaction_env_notes[spec.id] = redaction_env_note
        add_settings_full_width_row(card_form, redaction_env_note)

    status = QLabel(describe_log_status(spec))
    status.setObjectName("SettingsLogStatus")
    host.diagnostic_log_status_labels[spec.id] = status
    add_settings_full_width_row(card_form, status)

    view_btn = QPushButton(f"View {spec.title}")
    apply_brand_primary(view_btn, icon_name="fa5s.file-alt")
    view_btn.setToolTip(
        f"Open an in-app viewer for {spec.path_fn()} with refresh and live tail."
    )
    log_id = spec.id
    view_btn.clicked.connect(
        lambda _checked=False, lid=log_id: host._on_view_diagnostic_log_clicked(lid)
    )
    host.diagnostic_log_view_buttons[spec.id] = view_btn

    clear_btn = QPushButton("Clear log")
    apply_brand_danger(clear_btn, icon_name="fa5s.trash")
    clear_btn.setToolTip(
        f"Delete all contents of {spec.path_fn()} and any rotated backup files. "
        "New entries are recorded automatically when logging resumes."
    )
    clear_btn.clicked.connect(
        lambda _checked=False, lid=log_id: host._on_clear_diagnostic_log_clicked(lid)
    )
    host.diagnostic_log_clear_buttons[spec.id] = clear_btn

    btn_row = QWidget()
    btn_row_layout = QHBoxLayout(btn_row)
    btn_row_layout.setContentsMargins(0, 0, 0, 0)
    btn_row_layout.setSpacing(8)
    btn_row_layout.addWidget(view_btn)
    btn_row_layout.addWidget(clear_btn)
    btn_row_layout.addStretch(1)
    add_settings_full_width_row(card_form, btn_row)

    layout.addWidget(card)


def add_diagnostic_log_sections(
    host,
    layout: QVBoxLayout,
    specs: Iterable[DiagnosticLogSpec],
    *,
    is_dark: bool,
) -> None:
    ensure_diagnostic_log_host_attrs(host)
    for spec in specs:
        add_diagnostic_log_section(host, layout, spec, is_dark=is_dark)
