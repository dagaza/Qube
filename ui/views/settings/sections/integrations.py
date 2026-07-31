"""Integrations settings section — MCP registry + capability permission/consent UI."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QFileSystemWatcher
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.integrations.consent_controller import (
    CapabilityConsentRow,
    ConsentUIState,
    IntegrationsConsentController,
)
from core.integrations.mcp_server_registry import list_mcp_server_summaries
from core.integrations.registry.provider_registry import list_capability_providers
from ui.components.brand_buttons import apply_brand_secondary
from ui.components.toggle import PrestigeToggle
from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import add_subsection_to_layout, make_settings_hint

_TIER_LABELS = {
    "read": "Read",
    "write": "Write",
    "destructive": "Destructive",
}

_STATE_HINTS = {
    ConsentUIState.NEEDS_REVIEW: "Needs explicit review before grant",
    ConsentUIState.REREVIEW_REQUIRED: "Capability changed — re-review required",
    ConsentUIState.DENIED: "Not granted (default-deny)",
    ConsentUIState.ALLOWED: "Granted",
}


def _tier_badge(tier_value: str, *, is_dark: bool) -> QLabel:
    label = QLabel(_TIER_LABELS.get(tier_value, tier_value.title()))
    color = {
        "read": ("#a6e3a1", "#4f7a55") if is_dark else ("#15803d", "#22c55e"),
        "write": ("#f9e2af", "#9a7b3c") if is_dark else ("#b45309", "#f59e0b"),
        "destructive": ("#f38ba8", "#9a4a62") if is_dark else ("#be123c", "#f43f5e"),
    }.get(tier_value, ("#a6adc8", "#5c6078") if is_dark else ("#64748b", "#94a3b8"))
    fg, border = color
    label.setStyleSheet(
        f"color: {fg}; border: 1px solid {border}; border-radius: 8px; "
        f"padding: 2px 8px; font-size: 11px; font-weight: 600;"
    )
    label.setFixedHeight(24)
    return label


def _make_capability_row(
    host,
    row: CapabilityConsentRow,
    controller: IntegrationsConsentController,
    *,
    is_dark: bool,
) -> QWidget:
    outer = QWidget()
    layout = QHBoxLayout(outer)
    layout.setContentsMargins(0, 4, 0, 4)
    layout.setSpacing(12)

    toggle = PrestigeToggle()
    toggle.setEnabled(row.ui_state is not ConsentUIState.NEEDS_REVIEW)
    toggle.blockSignals(True)
    toggle.setChecked(row.ui_state is ConsentUIState.ALLOWED)
    toggle.blockSignals(False)

    def _on_toggled(checked: bool, *, descriptor=row.descriptor, ctrl=controller) -> None:
        if checked:
            try:
                ctrl.grant_capability(descriptor)
            except ValueError:
                toggle.blockSignals(True)
                toggle.setChecked(False)
                toggle.blockSignals(False)
                return
        else:
            ctrl.deny_capability(descriptor)
        sync_integrations_consent_panel(host, is_dark=is_dark)
        sync_integrations_mcp_servers_panel(host, is_dark=is_dark)

    toggle.toggled.connect(_on_toggled)

    title = QLabel(row.descriptor.action.replace("-", " ").title())
    title.setWordWrap(True)
    title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    tier = _tier_badge(row.tier.value, is_dark=is_dark)

    hint_text = _STATE_HINTS.get(row.ui_state, row.decision.reason)
    if row.needs_review:
        hint_text = _STATE_HINTS[ConsentUIState.NEEDS_REVIEW]
    hint = QLabel(hint_text)
    hint.setWordWrap(True)
    hint.setStyleSheet(
        f"color: {'#6c7086' if is_dark else '#64748b'}; font-size: 11px;"
    )

    text_col = QVBoxLayout()
    text_col.setContentsMargins(0, 0, 0, 0)
    text_col.setSpacing(2)
    title_row = QHBoxLayout()
    title_row.setContentsMargins(0, 0, 0, 0)
    title_row.addWidget(title)
    title_row.addWidget(tier)
    title_row.addStretch(1)
    text_col.addLayout(title_row)
    text_col.addWidget(hint)

    layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignTop)
    layout.addLayout(text_col, stretch=1)
    return outer


def _configure_mcp_servers_table(table: QTableWidget, *, is_dark: bool) -> None:
    table.setObjectName("IntegrationsMcpServersTable")
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Server", "Capabilities", "Granted", "Health"])
    header = table.horizontalHeader()
    for col in range(4):
        header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setShowGrid(False)
    table.setFrameShape(QFrame.Shape.NoFrame)
    table.setMinimumHeight(120)
    muted = "#6c7086" if is_dark else "#64748b"
    table.setStyleSheet(
        f"""
        QTableWidget#IntegrationsMcpServersTable {{
            background: transparent;
            border: none;
        }}
        QHeaderView::section {{
            color: {muted};
            background: transparent;
            border: none;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 0;
        }}
        """
    )


def build_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(container)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    servers_card, servers_layout = begin_settings_section_card(
        host,
        is_dark=is_dark,
        card_title="MCP servers",
        card_anchor="integrations_mcp_servers",
    )
    add_subsection_to_layout(servers_layout, "MCP servers", anchor="integrations_mcp_servers")

    servers_intro = make_settings_hint(
        "MCP servers are configured under Knowledge → Custom sources. "
        "After save or test, Qube discovers capabilities and prompts you to review permissions."
    )
    servers_layout.addWidget(servers_intro)

    host.integrations_mcp_servers_table = QTableWidget()
    _configure_mcp_servers_table(host.integrations_mcp_servers_table, is_dark=is_dark)
    servers_layout.addWidget(host.integrations_mcp_servers_table)

    manage_row = QHBoxLayout()
    manage_row.setContentsMargins(0, 0, 0, 0)
    manage_btn = QPushButton("Manage in Knowledge → Custom sources")
    apply_brand_secondary(manage_btn)
    manage_btn.clicked.connect(lambda: _open_knowledge_custom_sources(host))
    manage_row.addWidget(manage_btn, alignment=Qt.AlignmentFlag.AlignLeft)
    manage_row.addStretch(1)
    manage_host = QWidget()
    manage_host.setLayout(manage_row)
    servers_layout.addWidget(manage_host)

    layout.addWidget(servers_card)

    consent_card, card_layout = begin_settings_section_card(
        host,
        is_dark=is_dark,
        card_title="Capability permissions",
        card_anchor="integrations_consent",
    )
    add_subsection_to_layout(card_layout, "Capability permissions", anchor="integrations_consent")

    intro = make_settings_hint(
        "Review and grant capabilities discovered from integration providers. "
        "Write and destructive capabilities stay off until you explicitly allow them. "
        "Capabilities flagged for review cannot be granted here."
    )
    card_layout.addWidget(intro)

    scroll_host = QWidget()
    scroll_layout = QVBoxLayout(scroll_host)
    scroll_layout.setContentsMargins(0, 0, 0, 0)
    scroll_layout.setSpacing(8)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(scroll_host)
    scroll.setMinimumHeight(240)

    host.integrations_consent_scroll = scroll
    host.integrations_consent_body = scroll_host
    host.integrations_consent_layout = scroll_layout

    card_layout.addWidget(scroll)
    layout.addWidget(consent_card)
    setup_integrations_dir_watcher(host)
    from core.integrations.descriptor_cache import reconcile_mcp_integration_state

    reconcile_mcp_integration_state()
    sync_integrations_mcp_servers_panel(host, is_dark=is_dark)
    sync_integrations_consent_panel(host, is_dark=is_dark)
    return container


def _open_knowledge_custom_sources(host) -> None:
    """Navigate to Knowledge → Custom sources and expand that card."""
    if hasattr(host, "select_settings_section"):
        host.select_settings_section("knowledge", anchor="knowledge_custom_sources")
        return
    window = host.window()
    settings = getattr(window, "settings_view", None)
    if settings is not None and hasattr(settings, "select_settings_section"):
        settings.select_settings_section("knowledge", anchor="knowledge_custom_sources")


def setup_integrations_dir_watcher(host) -> None:
    """Refresh consent rows when descriptor or consent files change on disk."""
    from core.integrations.capabilities.persistence import integrations_dir

    timer = QTimer(host)
    timer.setSingleShot(True)
    timer.setInterval(400)
    timer.timeout.connect(lambda: _refresh_integrations_panels(host))
    host._integrations_reload_timer = timer

    watcher = QFileSystemWatcher(host)
    watcher.directoryChanged.connect(lambda _path: timer.start())
    host._integrations_dir_watcher = watcher

    root = str(integrations_dir("mcp").parent)
    if root not in watcher.directories():
        watcher.addPath(root)


def _refresh_integrations_panels(host) -> None:
    from core.integrations.descriptor_cache import reconcile_mcp_integration_state

    reconcile_mcp_integration_state()
    sync_integrations_mcp_servers_panel(host)
    sync_integrations_consent_panel(host)


def sync_integrations_mcp_servers_panel(host, *, is_dark: bool | None = None) -> None:
    table = getattr(host, "integrations_mcp_servers_table", None)
    if table is None:
        return
    if is_dark is None:
        is_dark = coalesce_settings_is_dark(host)

    summaries = list_mcp_server_summaries()
    table.setRowCount(len(summaries))
    for row_idx, summary in enumerate(summaries):
        table.setItem(row_idx, 0, QTableWidgetItem(summary.label))
        table.setItem(row_idx, 1, QTableWidgetItem(str(summary.capability_count)))
        table.setItem(row_idx, 2, QTableWidgetItem(str(summary.granted_count)))
        health_item = QTableWidgetItem(summary.health_label)
        if summary.rereview_count:
            health_item.setForeground(
                Qt.GlobalColor.yellow if is_dark else Qt.GlobalColor.darkYellow
            )
        table.setItem(row_idx, 3, health_item)

    if not summaries:
        table.setRowCount(1)
        placeholder = QTableWidgetItem("No MCP servers configured yet.")
        placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        table.setItem(0, 0, placeholder)
        for col in range(1, 4):
            table.setItem(0, col, QTableWidgetItem(""))

    _configure_mcp_servers_table(table, is_dark=is_dark)


def sync_integrations_consent_panel(host, *, is_dark: bool | None = None) -> None:
    """Rebuild the consent rows for every registered provider with cached descriptors."""
    body_layout = getattr(host, "integrations_consent_layout", None)
    if body_layout is None:
        return

    if is_dark is None:
        is_dark = coalesce_settings_is_dark(host)

    while body_layout.count():
        item = body_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    provider_ids = list_capability_providers()
    if not provider_ids:
        empty = make_settings_hint("No integration providers are registered yet.")
        body_layout.addWidget(empty)
        body_layout.addStretch(1)
        return

    any_rows = False
    for provider_id in provider_ids:
        controller = IntegrationsConsentController(provider_id)
        rows = controller.list_capability_rows()
        if not rows:
            continue
        any_rows = True
        header = QLabel(provider_id.upper())
        header.setStyleSheet("font-weight: 700; font-size: 12px; padding-top: 8px;")
        body_layout.addWidget(header)

        current_group: str | None = None
        for row in rows:
            if row.group != current_group:
                current_group = row.group
                group_label = QLabel(current_group.replace("-", " ").title())
                group_label.setStyleSheet(
                    f"color: {'#bac2de' if is_dark else '#475569'}; "
                    "font-size: 11px; font-weight: 600; padding-left: 4px;"
                )
                body_layout.addWidget(group_label)
            body_layout.addWidget(
                _make_capability_row(host, row, controller, is_dark=is_dark)
            )

    if not any_rows:
        empty = make_settings_hint(
            "No discovered capabilities yet. Connect an MCP server under "
            "Knowledge → Custom sources and run discovery to review permissions here."
        )
        body_layout.addWidget(empty)

    body_layout.addStretch(1)
