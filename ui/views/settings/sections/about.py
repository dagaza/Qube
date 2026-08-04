"""About settings section — version and software updates."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QPushButton, QVBoxLayout, QWidget

from core.__version__ import __version__
from core.support_feedback import QUBE_WEBSITE_URL
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_subsection_to_form,
    add_settings_full_width_row,
    make_settings_action_row,
    make_settings_hint,
)


def _add_about_action_to_form(
    form: QFormLayout, hint_lbl: QWidget, button: QPushButton
) -> None:
    add_settings_full_width_row(form, hint_lbl)
    add_settings_full_width_row(form, make_settings_action_row(button))


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    layout.addWidget(
        make_settings_hint(
            "Your installed Qube version, release updates, and links to the project website."
        )
    )

    # --- About Qube card ---
    about_card, about_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    about_form = add_settings_card_form(about_card_layout)
    add_subsection_to_form(about_form, "About Qube", anchor="about-qube")

    host.about_qube_hint_lbl = make_settings_hint(
        f"You are running Qube {__version__}."
    )

    host.open_qube_website_btn = QPushButton("Open Qube website")
    apply_brand_primary(host.open_qube_website_btn, icon_name="fa5s.external-link-alt")
    host.open_qube_website_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.open_qube_website_btn.clicked.connect(host._on_open_qube_website_clicked)

    _add_about_action_to_form(
        about_form,
        host.about_qube_hint_lbl,
        host.open_qube_website_btn,
    )
    layout.addWidget(about_card)

    # --- Software updates card ---
    updates_card, updates_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    updates_form = add_settings_card_form(updates_card_layout)
    add_subsection_to_form(updates_form, "Software updates", anchor="software-updates")

    host.software_updates_hint_lbl = make_settings_hint(
        "Check GitHub Releases for a newer build, or open the update guide in Library → Qube."
    )

    host.check_for_updates_btn = QPushButton("Check for updates")
    apply_brand_primary(host.check_for_updates_btn, icon_name="fa5s.sync-alt")
    host.check_for_updates_btn.setToolTip(
        "Contact GitHub Releases and compare with your installed version."
    )
    host.check_for_updates_btn.clicked.connect(host._on_check_for_updates_clicked)

    _add_about_action_to_form(
        updates_form,
        host.software_updates_hint_lbl,
        host.check_for_updates_btn,
    )
    layout.addWidget(updates_card)

    return widget
