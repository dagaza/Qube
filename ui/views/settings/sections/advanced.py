"""Advanced settings section — local LLM tour replay and JSON settings editor."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from core.settings_store import default_user_settings_path
from ui.components.brand_buttons import apply_brand_primary


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(8)

    help_hint = QLabel(
        "Replay the guided tour for choosing Internal Engine and loading a local .gguf model. "
        "The tour includes model picks matched to your hardware."
    )
    help_hint.setWordWrap(True)
    help_hint.setProperty("class", "ToolsPaneControl")
    host.local_llm_tour_hint_lbl = help_hint

    host.replay_local_llm_tour_btn = QPushButton("Replay Local LLM Setup Tour")
    apply_brand_primary(host.replay_local_llm_tour_btn, icon_name="fa5s.play-circle")
    host.replay_local_llm_tour_btn.setToolTip(
        "Walk through Settings, AI Engine, Select AI Model, and Model Manager with "
        "spotlight hints."
    )
    host.replay_local_llm_tour_btn.clicked.connect(host._on_replay_local_llm_tour_clicked)

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

    layout.addWidget(help_hint)
    layout.addWidget(
        host.replay_local_llm_tour_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )
    layout.addWidget(host.settings_json_hint_lbl)
    layout.addWidget(
        host.open_settings_json_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )
    layout.addWidget(host.settings_file_status_lbl)

    return widget
