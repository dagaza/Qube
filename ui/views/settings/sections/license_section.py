"""License settings — import and remove signed Qube licenses."""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.licensing.license_schema import LICENSE_FILE_EXTENSION
from core.licensing.store import default_license_cache_path
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.views.settings.license_status_ui import build_license_status_banner
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_settings_full_width_row,
    add_subsection_to_form,
    make_settings_hint,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    license_card, license_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    host.license_section_card = license_card
    license_form = add_settings_card_form(license_card_layout)
    add_subsection_to_form(license_form, "License", anchor="license")

    host.license_hint_lbl = make_settings_hint(
        "Import a signed .qube-license file when you receive one from Qube or your "
        f"organization. Licenses are cached at {default_license_cache_path()}. "
        "Nothing prompts you on startup — this section is optional."
    )
    add_settings_full_width_row(license_form, host.license_hint_lbl)

    add_settings_full_width_row(
        license_form, build_license_status_banner(host, is_dark=is_dark)
    )

    host.license_status_lbl = QLabel("")
    host.license_status_lbl.setWordWrap(True)
    host.license_status_lbl.setObjectName("SettingsLogDescription")
    add_settings_full_width_row(license_form, host.license_status_lbl)

    host.import_license_btn = QPushButton("Import license file")
    apply_brand_primary(host.import_license_btn, icon_name="fa5s.file-import")
    host.import_license_btn.setToolTip(
        f"Select a signed {LICENSE_FILE_EXTENSION} or JSON license file to import."
    )
    host.import_license_btn.clicked.connect(host._on_import_license_clicked)

    host.remove_license_btn = QPushButton("Remove cached license")
    apply_brand_danger(host.remove_license_btn, icon_name="fa5s.trash")
    host.remove_license_btn.setToolTip(
        f"Delete the cached license at {default_license_cache_path()}."
    )
    host.remove_license_btn.clicked.connect(host._on_remove_license_clicked)

    license_btn_row = QWidget()
    license_btn_row_layout = QHBoxLayout(license_btn_row)
    license_btn_row_layout.setContentsMargins(0, 0, 0, 0)
    license_btn_row_layout.setSpacing(8)
    license_btn_row_layout.addWidget(host.import_license_btn)
    license_btn_row_layout.addWidget(host.remove_license_btn)
    license_btn_row_layout.addStretch(1)
    add_settings_full_width_row(license_form, license_btn_row)
    layout.addWidget(license_card)

    if hasattr(host, "_refresh_license_status_ui"):
        host._refresh_license_status_ui()

    return widget
