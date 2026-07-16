"""Advanced settings section — JSON settings editor and diagnostic logs."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.diagnostic_logs import (
    DiagnosticLogSpec,
    describe_log_status,
    iter_diagnostic_logs,
)
from core.paths import logs_dir
from core.settings_store import default_user_settings_path
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.toggle import PrestigeToggle
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_subsection_to_layout,
    make_settings_action_row,
    make_settings_action_status_label,
    make_settings_hint,
)


def _add_diagnostic_log_section(
    host,
    layout: QVBoxLayout,
    spec: DiagnosticLogSpec,
    *,
    is_dark: bool,
) -> None:
    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(card_layout, spec.title, anchor=spec.id)

    desc = QLabel(spec.description)
    desc.setWordWrap(True)
    desc.setObjectName("SettingsLogDescription")
    card_layout.addWidget(desc)

    if spec.note:
        note = QLabel(spec.note)
        note.setWordWrap(True)
        note.setObjectName("SettingsLogNote")
        card_layout.addWidget(note)

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
        card_layout.addWidget(recording_row)

        env_note = QLabel("")
        env_note.setWordWrap(True)
        env_note.setObjectName("SettingsLogNote")
        env_note.hide()
        host.diagnostic_log_recording_env_notes[spec.id] = env_note
        card_layout.addWidget(env_note)

    status = QLabel(describe_log_status(spec))
    status.setObjectName("SettingsLogStatus")
    host.diagnostic_log_status_labels[spec.id] = status
    card_layout.addWidget(status)

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
    card_layout.addWidget(btn_row)

    layout.addWidget(card)


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    # --- JSON settings card ---
    json_card, json_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(json_card_layout, "JSON settings", anchor="json")

    host.settings_json_hint_lbl = make_settings_hint(
        f"Edit preferences in {default_user_settings_path()} "
        "(schema: assets/config/settings.schema.json). "
        "Use the built-in editor to format, validate, and save — "
        "or reload when the file changes on disk."
    )
    host.settings_json_hint_lbl.setToolTip(
        "User settings file path, JSON schema location, and editor capabilities."
    )
    json_card_layout.addWidget(host.settings_json_hint_lbl)

    host.open_settings_json_btn = QPushButton("Edit settings.json")
    apply_brand_primary(host.open_settings_json_btn, icon_name="fa5s.code")
    host.open_settings_json_btn.setToolTip(
        "Open the built-in JSON editor for user settings. "
        "Format, validate, and save — or reload when the file changes on disk."
    )
    host.open_settings_json_btn.clicked.connect(host._on_open_settings_json_clicked)
    json_card_layout.addWidget(make_settings_action_row(host.open_settings_json_btn))

    host.settings_file_status_lbl = make_settings_action_status_label()
    host._settings_file_status_sequence = 0
    host._settings_file_status_fade_anim = None
    json_card_layout.addWidget(host.settings_file_status_lbl)
    layout.addWidget(json_card)

    # --- Diagnostic logs intro card ---
    logs_intro_card, logs_intro_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    add_subsection_to_layout(logs_intro_card_layout, "Diagnostic logs", anchor="logs")

    logs_path = logs_dir()
    host.diagnostic_logs_hint_lbl = make_settings_hint(
        f"Qube writes rotating debug logs under {logs_path}. "
        "Most Qube.* messages still go to the terminal only; use the viewers below "
        "for dedicated application, LLM, routing, web search, and skills log files."
    )
    logs_intro_card_layout.addWidget(host.diagnostic_logs_hint_lbl)

    host.open_logs_folder_btn = QPushButton("Open logs folder")
    apply_brand_primary(host.open_logs_folder_btn, icon_name="fa5s.folder-open")
    host.open_logs_folder_btn.setToolTip(f"Reveal {logs_path} in your file manager.")
    host.open_logs_folder_btn.clicked.connect(host._on_open_logs_folder_clicked)
    logs_intro_card_layout.addWidget(make_settings_action_row(host.open_logs_folder_btn))
    layout.addWidget(logs_intro_card)

    host.diagnostic_log_status_labels: dict[str, QLabel] = {}
    host.diagnostic_log_view_buttons: dict[str, QPushButton] = {}
    host.diagnostic_log_clear_buttons: dict[str, QPushButton] = {}
    host.diagnostic_log_recording_toggles: dict[str, PrestigeToggle] = {}
    host.diagnostic_log_recording_env_notes: dict[str, QLabel] = {}

    for spec in iter_diagnostic_logs():
        _add_diagnostic_log_section(host, layout, spec, is_dark=is_dark)

    host._sync_all_diagnostic_log_recording_toggles()

    return widget
