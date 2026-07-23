"""Integrations settings section — capability permission/consent UI."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.integrations.consent_controller import (
    CapabilityConsentRow,
    ConsentUIState,
    IntegrationsConsentController,
)
from core.integrations.registry.provider_registry import list_capability_providers
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


def build_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(container)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
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
    layout.addWidget(card)
    sync_integrations_consent_panel(host, is_dark=is_dark)
    return container


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
            "No discovered capabilities yet. Connect a provider and run discovery "
            "to review permissions here."
        )
        body_layout.addWidget(empty)

    body_layout.addStretch(1)
