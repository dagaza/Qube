"""Advanced settings section — JSON settings editor and diagnostic logs."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.diagnostic_logs import describe_log_file, iter_diagnostic_logs
from core.paths import logs_dir
from core.settings_store import default_user_settings_path
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.widgets import add_subsection_to_layout


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(8)

    add_subsection_to_layout(layout, "JSON settings", anchor="json")

    host.settings_json_hint_lbl = QLabel(
        f"Edit preferences in {default_user_settings_path()} "
        "(schema: assets/config/settings.schema.json). "
        "Use the built-in editor to format, validate, and save — "
        "or reload when the file changes on disk."
    )
    host.settings_json_hint_lbl.setWordWrap(True)
    host.settings_json_hint_lbl.setProperty("class", "ToolsPaneControl")

    host.open_settings_json_btn = QPushButton("Edit settings.json")
    apply_brand_primary(host.open_settings_json_btn, icon_name="fa5s.code")
    host.open_settings_json_btn.setToolTip(
        "Open the built-in JSON editor for user settings. "
        "Format, validate, and save — or reload when the file changes on disk."
    )
    host.open_settings_json_btn.clicked.connect(host._on_open_settings_json_clicked)

    host.settings_file_status_lbl = QLabel("")
    host.settings_file_status_lbl.setProperty("class", "ToolsPaneControl")
    host._settings_file_status_sequence = 0
    host._settings_file_status_fade_anim = None

    layout.addWidget(host.settings_json_hint_lbl)
    layout.addWidget(
        host.open_settings_json_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )
    layout.addWidget(host.settings_file_status_lbl)

    add_subsection_to_layout(layout, "Diagnostics logs", anchor="logs")

    logs_path = logs_dir()
    host.diagnostic_logs_hint_lbl = QLabel(
        f"Qube writes rotating debug logs under {logs_path}. "
        "Most Qube.* messages still go to the terminal only; use the viewers below "
        "for the dedicated LLM and routing log files."
    )
    host.diagnostic_logs_hint_lbl.setWordWrap(True)
    host.diagnostic_logs_hint_lbl.setProperty("class", "ToolsPaneControl")
    layout.addWidget(host.diagnostic_logs_hint_lbl)

    host.open_logs_folder_btn = QPushButton("Open logs folder")
    apply_brand_primary(host.open_logs_folder_btn, icon_name="fa5s.folder-open")
    host.open_logs_folder_btn.setToolTip(
        f"Reveal {logs_path} in your file manager."
    )
    host.open_logs_folder_btn.clicked.connect(host._on_open_logs_folder_clicked)
    layout.addWidget(
        host.open_logs_folder_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )

    host.diagnostic_log_status_labels: dict[str, QLabel] = {}
    host.diagnostic_log_view_buttons: dict[str, QPushButton] = {}

    for spec in iter_diagnostic_logs():
        row = QVBoxLayout()
        row.setSpacing(4)

        title = QLabel(spec.title)
        title.setProperty("class", "ToolsPaneControl")
        row.addWidget(title)

        desc = QLabel(spec.description)
        desc.setWordWrap(True)
        desc.setProperty("class", "ToolsPaneControl")
        row.addWidget(desc)

        if spec.note:
            note = QLabel(spec.note)
            note.setWordWrap(True)
            note.setProperty("class", "ToolsPaneControl")
            row.addWidget(note)

        status = QLabel(describe_log_file(spec.path_fn()))
        status.setProperty("class", "ToolsPaneControl")
        host.diagnostic_log_status_labels[spec.id] = status
        row.addWidget(status)

        btn_row = QHBoxLayout()
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
        btn_row.addWidget(view_btn)
        btn_row.addStretch()
        row.addLayout(btn_row)

        layout.addLayout(row)

    return widget
