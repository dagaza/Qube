"""Shared web discovery privacy tier controls for Knowledge and Privacy & data."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QLabel, QMenu, QPushButton, QWidget

from core.app_settings import get_discovery_privacy_tier
from core.knowledge.discovery.privacy_policy import (
    TIER_BALANCED,
    TIER_ENHANCED,
    TIER_PRIVATE,
    TIER_SEARXNG,
    privacy_tier_description,
    privacy_tier_label,
)
from ui.components.brand_buttons import apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.views.settings.widgets import (
    add_settings_full_width_row,
    register_settings_selector_width,
)


def add_privacy_tier_selector_row(
    host,
    form: QFormLayout,
    *,
    is_dark: bool,
    selector_attr: str,
    description_attr: str,
) -> None:
    """Add Privacy tier selector + description label; store widgets on host."""
    tiers = (TIER_PRIVATE, TIER_BALANCED, TIER_ENHANCED, TIER_SEARXNG)
    selector = SelectorButton(
        privacy_tier_label(get_discovery_privacy_tier()),
        is_dark=is_dark,
    )
    selector.setMaximumWidth(280)
    selector.setMenu(QMenu(selector))
    selector.setToolTip(
        "Balance privacy vs optional API fallbacks for @internet and general web search."
    )
    register_settings_selector_width(
        selector,
        *[privacy_tier_label(tier) for tier in tiers],
    )
    setattr(host, selector_attr, selector)
    form.addRow("Privacy tier", selector)

    description = QLabel(privacy_tier_description(get_discovery_privacy_tier()))
    description.setWordWrap(True)
    description.setObjectName("SettingsLogDescription")
    setattr(host, description_attr, description)
    add_settings_full_width_row(form, description)


def add_open_knowledge_discovery_button(host, form: QFormLayout) -> None:
    """Link to advanced discovery controls on the Knowledge settings page."""
    btn = QPushButton("Open Knowledge → Web search discovery")
    apply_brand_primary(btn, icon_name="fa5s.external-link-alt")
    btn.setToolTip(
        "Open advanced web discovery limits, provider setup, DDG usage, and SearXNG."
    )
    btn.clicked.connect(
        lambda _checked=False: host.select_settings_section(
            "knowledge",
            anchor="web_discovery",
        )
    )
    host.privacy_data_open_knowledge_discovery_btn = btn
    add_settings_full_width_row(form, btn)


def add_open_privacy_data_button(host, form: QFormLayout) -> None:
    """Link to Privacy & data for tier, Hybrid Internet Mode, and audit logs."""
    btn = QPushButton("Open Privacy & data")
    apply_brand_primary(btn, icon_name="fa5s.external-link-alt")
    btn.setToolTip(
        "Open Privacy & data for web discovery tier, Hybrid Internet Mode, "
        "and audit log controls."
    )
    btn.clicked.connect(
        lambda _checked=False: host.select_settings_section(
            "privacy.data",
            anchor="web_discovery_privacy",
        )
    )
    host.discovery_open_privacy_data_btn = btn
    add_settings_full_width_row(form, btn)
