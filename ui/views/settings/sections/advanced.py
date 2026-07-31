"""Advanced settings section — JSON settings editor."""

from __future__ import annotations

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from core.settings_store import default_user_settings_path
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_settings_full_width_row,
    add_subsection_to_form,
    make_settings_action_row,
    make_settings_action_status_label,
    make_settings_hint,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    json_card, json_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    json_form = add_settings_card_form(json_card_layout)
    add_subsection_to_form(json_form, "JSON settings", anchor="json")

    host.settings_json_hint_lbl = make_settings_hint(
        f"Edit preferences in {default_user_settings_path()} "
        "(schema: assets/config/settings.schema.json). "
        "Use the built-in editor to format, validate, and save — "
        "or reload when the file changes on disk. Prefer other settings pages "
        "when possible; invalid JSON can affect startup."
    )
    host.settings_json_hint_lbl.setToolTip(
        "User settings file path, JSON schema location, and editor capabilities."
    )
    add_settings_full_width_row(json_form, host.settings_json_hint_lbl)

    host.open_settings_json_btn = QPushButton("Edit settings.json")
    apply_brand_primary(host.open_settings_json_btn, icon_name="fa5s.code")
    host.open_settings_json_btn.setToolTip(
        "Open the built-in JSON editor for user settings. "
        "Format, validate, and save — or reload when the file changes on disk."
    )
    host.open_settings_json_btn.clicked.connect(host._on_open_settings_json_clicked)
    add_settings_full_width_row(
        json_form, make_settings_action_row(host.open_settings_json_btn)
    )

    host.settings_file_status_lbl = make_settings_action_status_label()
    host._settings_file_status_sequence = 0
    host._settings_file_status_fade_anim = None
    add_settings_full_width_row(json_form, host.settings_file_status_lbl)
    layout.addWidget(json_card)

    return widget
