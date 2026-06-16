"""Contact & Feedback settings section — bug reports and feature requests."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from core.support_feedback import QUBE_WEBSITE_URL
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.widgets import add_subsection_to_layout


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(8)

    add_subsection_to_layout(layout, "Get in touch", anchor="get-in-touch")

    intro_hint = QLabel(
        "Report bugs, request features, and send other feedback through the forms on "
        "the Qube website."
    )
    intro_hint.setWordWrap(True)
    intro_hint.setProperty("class", "ToolsPaneControl")
    layout.addWidget(intro_hint)

    add_subsection_to_layout(layout, "Report a problem", anchor="report-bug")

    bug_hint = QLabel(
        "Open the website to submit a bug report with the details we need to investigate."
    )
    bug_hint.setWordWrap(True)
    bug_hint.setProperty("class", "ToolsPaneControl")
    layout.addWidget(bug_hint)

    host.report_bug_btn = QPushButton("Report a bug")
    apply_brand_primary(host.report_bug_btn, icon_name="fa5s.bug")
    host.report_bug_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.report_bug_btn.clicked.connect(host._on_open_qube_website_clicked)
    layout.addWidget(host.report_bug_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    add_subsection_to_layout(layout, "Suggest an improvement", anchor="feature-request")

    feature_hint = QLabel(
        "Open the website to share an idea for a new capability or a change to existing behavior."
    )
    feature_hint.setWordWrap(True)
    feature_hint.setProperty("class", "ToolsPaneControl")
    layout.addWidget(feature_hint)

    host.request_feature_btn = QPushButton("Request a feature")
    apply_brand_primary(host.request_feature_btn, icon_name="fa5s.lightbulb")
    host.request_feature_btn.setToolTip(f"Open {QUBE_WEBSITE_URL} in your browser.")
    host.request_feature_btn.clicked.connect(host._on_open_qube_website_clicked)
    layout.addWidget(host.request_feature_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    return widget
