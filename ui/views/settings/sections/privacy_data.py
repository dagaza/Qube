"""Privacy & data settings — audit logs and local data overview."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_mcp_internet_hybrid_enabled
from ui.components.brand_buttons import apply_brand_primary
from core.diagnostic_logs import iter_diagnostic_logs_by_category
from core.paths import logs_dir
from ui.components.toggle import PrestigeToggle
from ui.views.settings.sections.diagnostic_log_ui import (
    add_diagnostic_log_sections,
    ensure_diagnostic_log_host_attrs,
)
from ui.views.settings.sections.knowledge_web_discovery import (
    build_what_leaves_device_info_card,
)
from ui.views.settings.sections.privacy_tier_controls import (
    add_open_knowledge_discovery_button,
    add_privacy_tier_selector_row,
)
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

    overview_card, overview_layout = begin_settings_section_card(host, is_dark=is_dark)
    overview_form = add_settings_card_form(overview_layout)
    add_subsection_to_form(overview_form, "Data & egress overview", anchor="overview")

    host.privacy_data_overview_hint = make_settings_hint(
        "Review what Qube stores locally and what may leave your device during web "
        "discovery or integrations. Session summaries appear on Telemetry; audit logs "
        f"below can record queries and prompts under {logs_dir()}."
    )
    add_settings_full_width_row(overview_form, host.privacy_data_overview_hint)
    layout.addWidget(overview_card)

    session_card, session_layout = begin_settings_section_card(host, is_dark=is_dark)
    session_form = add_settings_card_form(session_layout)
    add_subsection_to_form(session_form, "Session audit", anchor="session_audit")

    host.privacy_data_session_audit_hint = make_settings_hint(
        "Telemetry shows live web discovery budgets and integration calls for the "
        "active conversation. Open a session in Conversations first for integration "
        "summaries."
    )
    add_settings_full_width_row(session_form, host.privacy_data_session_audit_hint)

    session_btn_row = QWidget()
    session_btn_layout = QHBoxLayout(session_btn_row)
    session_btn_layout.setContentsMargins(0, 0, 0, 0)
    session_btn_layout.setSpacing(10)

    host.privacy_data_open_telemetry_discovery_btn = QPushButton(
        "Open Telemetry → Web discovery"
    )
    apply_brand_primary(
        host.privacy_data_open_telemetry_discovery_btn,
        icon_name="fa5s.external-link-alt",
    )
    host.privacy_data_open_telemetry_discovery_btn.setToolTip(
        "Jump to the Web discovery card on Advanced Telemetry."
    )
    host.privacy_data_open_telemetry_discovery_btn.clicked.connect(
        host._on_privacy_data_open_telemetry_discovery_clicked
    )

    host.privacy_data_open_telemetry_integrations_btn = QPushButton(
        "Open Telemetry → Session integrations"
    )
    apply_brand_primary(
        host.privacy_data_open_telemetry_integrations_btn,
        icon_name="fa5s.external-link-alt",
    )
    host.privacy_data_open_telemetry_integrations_btn.setToolTip(
        "Jump to the Session integrations panel on Advanced Telemetry."
    )
    host.privacy_data_open_telemetry_integrations_btn.clicked.connect(
        host._on_privacy_data_open_telemetry_integrations_clicked
    )

    session_btn_layout.addWidget(host.privacy_data_open_telemetry_discovery_btn)
    session_btn_layout.addWidget(host.privacy_data_open_telemetry_integrations_btn)
    session_btn_layout.addStretch(1)
    add_settings_full_width_row(session_form, session_btn_row)
    layout.addWidget(session_card)

    web_card, web_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    web_form = add_settings_card_form(web_card_layout)
    add_subsection_to_form(web_form, "Web discovery privacy", anchor="web_discovery_privacy")

    web_controls = QWidget()
    web_controls_form = QFormLayout(web_controls)
    web_controls_form.setContentsMargins(0, 0, 0, 0)
    web_controls_form.setSpacing(8)
    add_privacy_tier_selector_row(
        host,
        web_controls_form,
        is_dark=is_dark,
        selector_attr="privacy_data_privacy_tier_selector",
        description_attr="privacy_data_privacy_tier_description",
    )
    add_open_knowledge_discovery_button(host, web_controls_form)
    add_settings_full_width_row(web_form, web_controls)

    hybrid_row = QWidget()
    hybrid_layout = QHBoxLayout(hybrid_row)
    hybrid_layout.setContentsMargins(0, 0, 0, 0)
    hybrid_layout.setSpacing(10)
    host.privacy_data_internet_hybrid_toggle = PrestigeToggle()
    host.privacy_data_internet_hybrid_toggle.setToolTip(
        "When enabled, Qube may auto-route to web search when context warrants it. "
        "Same setting as Hybrid Internet Mode in the Conversations tools panel."
    )
    host.privacy_data_internet_hybrid_toggle.blockSignals(True)
    host.privacy_data_internet_hybrid_toggle.setChecked(get_mcp_internet_hybrid_enabled())
    host.privacy_data_internet_hybrid_toggle.blockSignals(False)
    host.privacy_data_internet_hybrid_toggle.toggled.connect(
        host._on_privacy_data_internet_hybrid_toggled
    )
    hybrid_lbl = QLabel("Hybrid Internet Mode")
    hybrid_lbl.setWordWrap(True)
    hybrid_lbl.setObjectName("SettingsLogDescription")
    hybrid_lbl.setToolTip(host.privacy_data_internet_hybrid_toggle.toolTip())
    hybrid_layout.addWidget(
        host.privacy_data_internet_hybrid_toggle,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
    )
    hybrid_layout.addWidget(hybrid_lbl, stretch=1)
    add_settings_full_width_row(web_form, hybrid_row)

    host.privacy_data_what_leaves_card = build_what_leaves_device_info_card(
        is_dark=is_dark,
    )
    add_settings_full_width_row(web_form, host.privacy_data_what_leaves_card)
    layout.addWidget(web_card)

    ensure_diagnostic_log_host_attrs(host)
    add_diagnostic_log_sections(
        host,
        layout,
        iter_diagnostic_logs_by_category("audit"),
        is_dark=is_dark,
    )

    if hasattr(host, "_build_privacy_tier_menus"):
        host._build_privacy_tier_menus()
    if hasattr(host, "_sync_privacy_data_section_ui"):
        host._sync_privacy_data_section_ui()

    return widget
