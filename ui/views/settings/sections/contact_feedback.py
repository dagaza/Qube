"""Contact & Feedback settings section — bug reports and feature requests."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from core.support_feedback import QUBE_WEBSITE_URL
from core.ui_language import tr
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.widgets import (
    add_section_divider_to_layout,
    add_subsection_to_layout,
    make_settings_action_row,
    make_settings_hint,
)


def _build_feedback_card(description: str, button: QPushButton) -> QWidget:
    card = QWidget()
    card.setObjectName("SettingsLogCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(12, 10, 12, 10)
    card_layout.setSpacing(8)

    desc = QLabel(description)
    desc.setWordWrap(True)
    desc.setObjectName("SettingsLogDescription")
    card_layout.addWidget(desc)

    card_layout.addWidget(make_settings_action_row(button))

    return card


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    # --- Get in touch ---
    add_subsection_to_layout(layout, "Get in touch", anchor="get-in-touch")

    intro_block = QWidget()
    intro_layout = QVBoxLayout(intro_block)
    intro_layout.setContentsMargins(0, 0, 0, 0)
    intro_layout.setSpacing(10)

    intro_layout.addWidget(
        make_settings_hint(
            "Report bugs, request features, and send other feedback through the "
            "forms on the Qube website."
        )
    )

    layout.addWidget(intro_block)

    add_section_divider_to_layout(layout, is_dark=is_dark)

    # --- Report a problem ---
    add_subsection_to_layout(layout, "Report a problem", anchor="report-bug")

    host.report_bug_btn = QPushButton("Report a bug")
    apply_brand_primary(host.report_bug_btn, icon_name="fa5s.bug")
    host.report_bug_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.report_bug_btn.clicked.connect(host._on_open_qube_website_clicked)

    layout.addWidget(
        _build_feedback_card(
            "Open the website to submit a bug report with the details we need to investigate.",
            host.report_bug_btn,
        )
    )

    add_section_divider_to_layout(layout, is_dark=is_dark)

    # --- Suggest an improvement ---
    add_subsection_to_layout(layout, "Suggest an improvement", anchor="feature-request")

    host.request_feature_btn = QPushButton("Request a feature")
    apply_brand_primary(host.request_feature_btn, icon_name="fa5s.lightbulb")
    host.request_feature_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.request_feature_btn.clicked.connect(host._on_open_qube_website_clicked)

    layout.addWidget(
        _build_feedback_card(
            tr(
                "Open the website to share an idea for a new capability or a change "
                "to existing behaviour."
            ),
            host.request_feature_btn,
        )
    )

    return widget
