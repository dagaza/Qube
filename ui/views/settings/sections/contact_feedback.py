"""Contact & Feedback settings section — bug reports and feature requests."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core.support_feedback import QUBE_WEBSITE_URL
from core.ui_language import tr
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_subsection_to_form,
    make_settings_action_row,
    make_settings_hint,
    add_settings_full_width_row,
)


def _add_feedback_action_to_form(
    form: QFormLayout, description: str, button: QPushButton
) -> None:
    desc = QLabel(description)
    desc.setWordWrap(True)
    desc.setObjectName("SettingsLogDescription")
    desc.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    add_settings_full_width_row(form, desc)
    add_settings_full_width_row(form, make_settings_action_row(button))


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    layout.addWidget(
        make_settings_hint(
            "Report bugs, request features, and send other feedback through the "
            "forms on the Qube website."
        )
    )

    # --- Report a problem card ---
    bug_card, bug_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    bug_form = add_settings_card_form(bug_card_layout)
    add_subsection_to_form(bug_form, "Report a problem", anchor="report-bug")

    host.report_bug_btn = QPushButton("Report a bug")
    apply_brand_primary(host.report_bug_btn, icon_name="fa5s.bug")
    host.report_bug_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.report_bug_btn.clicked.connect(host._on_open_qube_website_clicked)

    _add_feedback_action_to_form(
        bug_form,
        "Open the website to submit a bug report with the details we need to investigate.",
        host.report_bug_btn,
    )
    layout.addWidget(bug_card)

    # --- Suggest an improvement card ---
    feature_card, feature_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    feature_form = add_settings_card_form(feature_card_layout)
    add_subsection_to_form(
        feature_form, "Suggest an improvement", anchor="feature-request"
    )

    host.request_feature_btn = QPushButton("Request a feature")
    apply_brand_primary(host.request_feature_btn, icon_name="fa5s.lightbulb")
    host.request_feature_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.request_feature_btn.clicked.connect(host._on_open_qube_website_clicked)

    _add_feedback_action_to_form(
        feature_form,
        tr(
            "Open the website to share an idea for a new capability or a change "
            "to existing behaviour."
        ),
        host.request_feature_btn,
    )
    layout.addWidget(feature_card)

    return widget
