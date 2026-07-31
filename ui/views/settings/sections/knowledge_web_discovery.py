"""Web search discovery policy UI (Settings → Knowledge)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import (
    get_advanced_discovery_unlocked,
    get_ddg_session_budget_override,
    get_discovery_pacing_enabled,
    get_discovery_privacy_tier,
    get_discovery_searxng_base_url,
)
from core.knowledge.adapters.brave_search import brave_search_configured
from core.knowledge.credentials import resolve_credential
from core.knowledge.discovery.health import conservative_mode_summary
from core.knowledge.discovery.policy import (
    BRAVE_DISCOVERY_PROVIDER_ID,
    PRIMARY_DISCOVERY_PROVIDER_ID,
    WIKIPEDIA_DISCOVERY_PROVIDER_ID,
    discovery_policy_summary_lines,
    discovery_provider_label,
)
from core.knowledge.discovery.backoff import get_provider_backoff
from core.knowledge.discovery.privacy_policy import (
    TIER_BALANCED,
    TIER_ENHANCED,
    TIER_PRIVATE,
    TIER_SEARXNG,
    privacy_tier_description,
    privacy_tier_label,
    what_leaves_device_lines,
)
from core.knowledge.discovery.searxng import SEARXNG_DISCOVERY_PROVIDER_ID, searxng_configured
from core.knowledge.discovery.session_budget import (
    DEFAULT_DDG_SESSION_BUDGET,
    get_ddg_burst_budget_status,
    get_ddg_session_budget_status,
)
from core.theme.accessors import theme_for
from core.theme.widget_styles import DISCOVERY_DIVIDER
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.discovery_card_style import (
    apply_discovery_info_card_theme,
    apply_discovery_provider_card_theme,
    build_discovery_divider,
    style_discovery_body_text,
    style_discovery_info_bullet,
    style_discovery_info_highlight,
    style_discovery_info_kv_key,
    style_discovery_info_kv_value,
    style_discovery_info_status,
    style_discovery_info_title,
    style_discovery_privacy_chip,
    style_discovery_provider_name,
    style_discovery_role_chip,
)
from ui.views.settings.knowledge_access_badge import (
    coalesce_settings_is_dark,
    make_knowledge_configure_action_row,
    resolve_settings_is_dark,
    style_access_badge,
    style_configure_button,
)
from ui.views.settings.sections.privacy_tier_controls import add_open_privacy_data_button
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_subsection_to_form,
    make_settings_hint,
    register_settings_selector_width,
    wrap_subsection,
    add_settings_full_width_row,
    add_settings_span_row,
)

_POLICY_KV_KEYS = frozenset(
    {
        "Privacy tier",
        "Primary",
        "Burst",
        "Session",
        "Pacing",
        "On primary failure",
    }
)


@contextmanager
def _preserve_settings_page_scroll(host):
    """Keep the active settings page scroll position across nested layout updates."""
    scroll: QScrollArea | None = None
    scroll_value = 0
    stack = getattr(host, "settings_section_stack", None)
    if stack is not None:
        current = stack.currentWidget()
        if isinstance(current, QScrollArea):
            scroll = current
            bar = scroll.verticalScrollBar()
            if bar is not None:
                scroll_value = bar.value()
    yield
    if scroll is None:
        return

    def _restore() -> None:
        bar = scroll.verticalScrollBar()
        if bar is not None:
            bar.setValue(scroll_value)

    QTimer.singleShot(0, _restore)


def _run_with_preserved_settings_scroll(host, fn: Callable[[], None]) -> None:
    with _preserve_settings_page_scroll(host):
        fn()


_PRIVACY_BADGES: dict[str, str] = {
    PRIMARY_DISCOVERY_PROVIDER_ID: "Free · No API key · Direct",
    WIKIPEDIA_DISCOVERY_PROVIDER_ID: "Free · No API key",
    BRAVE_DISCOVERY_PROVIDER_ID: "Optional API key · Third-party",
    SEARXNG_DISCOVERY_PROVIDER_ID: "Self-hosted · Advanced",
}


def _status_badge_text(provider_id: str) -> tuple[str, str]:
    """Return (badge_label, badge_kind) for a discovery provider row."""
    if provider_id == PRIMARY_DISCOVERY_PROVIDER_ID:
        backoff = get_provider_backoff(PRIMARY_DISCOVERY_PROVIDER_ID)
        if backoff is not None:
            minutes = max(1, (backoff.remaining_seconds + 59) // 60)
            return f"Paused ~{minutes}m", "coming_soon"
        return "Primary", "free"
    if provider_id == BRAVE_DISCOVERY_PROVIDER_ID:
        if brave_search_configured():
            mode = resolve_credential(BRAVE_DISCOVERY_PROVIDER_ID).mode.value
            if mode == "env_key":
                return "Configured (env)", "optional_key"
            return "Configured", "optional_key"
        return "Not configured", "coming_soon"
    if provider_id == SEARXNG_DISCOVERY_PROVIDER_ID:
        if searxng_configured():
            return "Configured", "optional_key"
        return "Not configured", "coming_soon"
    if provider_id == WIKIPEDIA_DISCOVERY_PROVIDER_ID:
        return "Always available", "free"
    return "—", "coming_soon"


def _split_privacy_note(note: str) -> list[str]:
    return [part.strip() for part in note.split("·") if part.strip()]


@dataclass(frozen=True)
class _ProviderCardParts:
    card: QWidget
    badge: QLabel
    configure_btn: QPushButton | None
    role_chip: QLabel
    provider_name: QLabel
    description: QLabel
    privacy_chips: tuple[QLabel, ...]
    divider: QWidget | None


def _build_discovery_provider_card(
    host,
    *,
    provider_id: str,
    role_label: str,
    description: str,
    is_dark: bool,
    show_configure: bool = False,
    configure_handler=None,
    configure_tooltip: str = "",
) -> _ProviderCardParts:
    """Discovery provider card with role chip, accent shell, and footer actions."""

    card = QWidget()
    apply_discovery_provider_card_theme(card, role_label=role_label, is_dark=is_dark)
    card_layout = QVBoxLayout(card)
    bottom_margin = 14 if show_configure else 12
    card_layout.setContentsMargins(14, 12, 14, bottom_margin)
    card_layout.setSpacing(10)

    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(10)

    role_chip = QLabel(role_label.upper())
    style_discovery_role_chip(role_chip, role_label=role_label, is_dark=is_dark)
    header_layout.addWidget(role_chip, alignment=Qt.AlignmentFlag.AlignTop)

    name_col = QWidget()
    name_layout = QVBoxLayout(name_col)
    name_layout.setContentsMargins(0, 0, 0, 0)
    name_layout.setSpacing(4)

    provider_name = QLabel(discovery_provider_label(provider_id))
    provider_name.setWordWrap(True)
    style_discovery_provider_name(provider_name, is_dark=is_dark)
    name_layout.addWidget(provider_name)

    privacy_chips: list[QLabel] = []
    privacy_note = _PRIVACY_BADGES.get(provider_id)
    if privacy_note:
        chips_row = QWidget()
        chips_layout = QHBoxLayout(chips_row)
        chips_layout.setContentsMargins(0, 0, 0, 0)
        chips_layout.setSpacing(6)
        for segment in _split_privacy_note(privacy_note):
            chip = QLabel(segment)
            style_discovery_privacy_chip(chip, is_dark=is_dark)
            chips_layout.addWidget(chip, alignment=Qt.AlignmentFlag.AlignLeft)
            privacy_chips.append(chip)
        chips_layout.addStretch(1)
        name_layout.addWidget(chips_row)

    header_layout.addWidget(name_col, stretch=1)

    badge_text, badge_kind = _status_badge_text(provider_id)
    badge = QLabel(badge_text)
    style_access_badge(badge, badge_kind, is_dark=is_dark)
    header_layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

    card_layout.addWidget(header)

    desc = QLabel(description)
    desc.setWordWrap(True)
    style_discovery_body_text(desc, is_dark=is_dark)
    card_layout.addWidget(desc)

    configure_btn: QPushButton | None = None
    divider: QWidget | None = None
    if show_configure:
        divider = build_discovery_divider(is_dark=is_dark)
        card_layout.addWidget(divider)
        configure_btn = QPushButton("Configure")
        style_configure_button(configure_btn, is_dark=is_dark)
        configure_btn.setToolTip(configure_tooltip or "Configure provider")
        handler = configure_handler or host._on_brave_search_configure_clicked
        configure_btn.clicked.connect(handler)
        card_layout.addWidget(make_knowledge_configure_action_row(configure_btn))

    return _ProviderCardParts(
        card=card,
        badge=badge,
        configure_btn=configure_btn,
        role_chip=role_chip,
        provider_name=provider_name,
        description=desc,
        privacy_chips=tuple(privacy_chips),
        divider=divider,
    )


class _DiscoveryInfoCard(QWidget):
    """Structured info card with title, highlight, and bullet / key-value rows."""

    def __init__(
        self,
        *,
        title: str,
        variant: str,
        is_dark: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._variant = variant
        self._is_dark = is_dark
        self._title_text = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._title_label = QLabel(title.upper())
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        layout.addWidget(self._content)
        self._last_privacy_lines: list[str] | None = None
        self._last_policy_lines: list[str] | None = None
        self._policy_structure: tuple[str, ...] | None = None
        self._policy_value_labels: list[QLabel] = []

        self.refresh_theme(is_dark)

    def refresh_theme(self, is_dark: bool) -> None:
        if is_dark != self._is_dark:
            self._last_privacy_lines = None
            self._last_policy_lines = None
            self._policy_structure = None
            self._policy_value_labels = []
        self._is_dark = is_dark
        apply_discovery_info_card_theme(self, variant=self._variant, is_dark=is_dark)
        style_discovery_info_title(
            self._title_label, variant=self._variant, is_dark=is_dark
        )

    @staticmethod
    def _policy_line_structure(lines: list[str]) -> tuple[str, ...]:
        structure: list[str] = []
        for line in lines:
            if ": " in line:
                key, _value = line.split(": ", 1)
                if key in _POLICY_KV_KEYS:
                    structure.append(f"kv:{key}")
                    continue
            lower = line.lower()
            if lower.startswith(("brave ", "conservative", "paused")):
                structure.append("status")
                continue
            structure.append("bullet")
        return tuple(structure)

    @staticmethod
    def _policy_line_value_text(line: str) -> str:
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in _POLICY_KV_KEYS:
                return value
        return line if line.startswith("•  ") else f"•  {line}"

    def _swap_content(self, populate) -> None:
        """Replace card body in one layout step to avoid collapse/expand flicker."""
        self.setUpdatesEnabled(False)
        try:
            new_content = QWidget()
            new_layout = QVBoxLayout(new_content)
            new_layout.setContentsMargins(0, 0, 0, 0)
            new_layout.setSpacing(6)
            populate(new_layout)

            root = self.layout()
            if root is None:
                return
            root.removeWidget(self._content)
            self._content.deleteLater()
            self._content = new_content
            self._content_layout = new_layout
            root.addWidget(new_content)
        finally:
            self.setUpdatesEnabled(True)
            self.updateGeometry()

    def set_privacy_lines(self, lines: list[str]) -> None:
        normalized = list(lines)
        if normalized == self._last_privacy_lines:
            return
        self._last_privacy_lines = normalized
        is_dark = self._is_dark

        def _populate(content_layout: QVBoxLayout) -> None:
            if not normalized:
                return
            highlight = QLabel(normalized[0])
            highlight.setWordWrap(True)
            style_discovery_info_highlight(highlight, is_dark=is_dark)
            content_layout.addWidget(highlight)

            for line in normalized[1:]:
                bullet = QLabel(f"•  {line}")
                bullet.setWordWrap(True)
                style_discovery_info_bullet(bullet, is_dark=is_dark)
                content_layout.addWidget(bullet)

        self._swap_content(_populate)

    def set_policy_lines(self, lines: list[str]) -> None:
        normalized = list(lines)
        if normalized == self._last_policy_lines:
            return

        structure = self._policy_line_structure(normalized)
        if (
            structure == self._policy_structure
            and self._policy_value_labels
            and len(self._policy_value_labels) == len(normalized)
        ):
            for label, line in zip(self._policy_value_labels, normalized):
                label.setText(self._policy_line_value_text(line))
            self._last_policy_lines = normalized
            return

        self._last_policy_lines = normalized
        self._policy_structure = structure
        value_labels: list[QLabel] = []
        is_dark = self._is_dark

        def _populate(content_layout: QVBoxLayout) -> None:
            if not normalized:
                return
            for line in normalized:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    if key in _POLICY_KV_KEYS:
                        row = QWidget()
                        row_layout = QHBoxLayout(row)
                        row_layout.setContentsMargins(0, 0, 0, 0)
                        row_layout.setSpacing(10)

                        key_lbl = QLabel(key.upper())
                        key_lbl.setMinimumWidth(108)
                        style_discovery_info_kv_key(key_lbl, is_dark=is_dark)
                        row_layout.addWidget(key_lbl)

                        value_lbl = QLabel(value)
                        value_lbl.setWordWrap(True)
                        style_discovery_info_kv_value(value_lbl, is_dark=is_dark)
                        row_layout.addWidget(value_lbl, stretch=1)
                        content_layout.addWidget(row)
                        value_labels.append(value_lbl)
                        continue

                if line.lower().startswith(("brave ", "conservative", "paused")):
                    status = QLabel(line)
                    status.setWordWrap(True)
                    style_discovery_info_status(status, is_dark=is_dark)
                    content_layout.addWidget(status)
                    value_labels.append(status)
                    continue

                bullet = QLabel(f"•  {line}")
                bullet.setWordWrap(True)
                style_discovery_info_bullet(bullet, is_dark=is_dark)
                content_layout.addWidget(bullet)
                value_labels.append(bullet)

        self._swap_content(_populate)
        self._policy_value_labels = value_labels


class _DiscoveryPolicyRow(QWidget):
    """Thin wrapper so sync can find provider cards and refresh status badges."""

    def __init__(
        self,
        host,
        *,
        provider_id: str,
        role_label: str,
        description: str,
        is_dark: bool,
        show_configure: bool = False,
        configure_handler=None,
        configure_tooltip: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._provider_id = provider_id
        self._role_label = role_label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        parts = _build_discovery_provider_card(
            host,
            provider_id=provider_id,
            role_label=role_label,
            description=description,
            is_dark=is_dark,
            show_configure=show_configure,
            configure_handler=configure_handler,
            configure_tooltip=configure_tooltip,
        )
        self._parts = parts
        self.badge = parts.badge
        self.configure_btn = parts.configure_btn
        self._card = parts.card
        layout.addWidget(parts.card)

    def refresh_theme(self, host, *, is_dark: bool | None = None) -> None:
        is_dark = coalesce_settings_is_dark(host, is_dark=is_dark)
        parts = self._parts
        apply_discovery_provider_card_theme(
            parts.card, role_label=self._role_label, is_dark=is_dark
        )
        style_discovery_role_chip(
            parts.role_chip, role_label=self._role_label, is_dark=is_dark
        )
        style_discovery_provider_name(parts.provider_name, is_dark=is_dark)
        for chip in parts.privacy_chips:
            style_discovery_privacy_chip(chip, is_dark=is_dark)
        style_discovery_body_text(parts.description, is_dark=is_dark)
        badge_text, badge_kind = _status_badge_text(self._provider_id)
        parts.badge.setText(badge_text)
        style_access_badge(parts.badge, badge_kind, is_dark=is_dark)
        if parts.divider is not None:
            parts.divider.setStyleSheet(
                theme_for(is_dark=is_dark).style(DISCOVERY_DIVIDER)
            )
        if parts.configure_btn is not None:
            style_configure_button(parts.configure_btn, is_dark=is_dark)


def sync_web_discovery_policy_section(host, *, is_dark: bool | None = None) -> None:
    """Refresh discovery policy controls and provider badges."""

    def _sync() -> None:
        section = getattr(host, "web_discovery_policy_section", None)
        if section is None:
            return
        resolved_dark = coalesce_settings_is_dark(host, is_dark=is_dark)

        if hasattr(host, "_sync_discovery_privacy_tier_selector"):
            host._sync_discovery_privacy_tier_selector()
        tier = get_discovery_privacy_tier()
        desc = getattr(host, "discovery_privacy_tier_description", None)
        if desc is not None:
            desc.setText(privacy_tier_description(tier))

        pacing = getattr(host, "discovery_pacing_toggle", None)
        if pacing is not None:
            pacing_checked = get_discovery_pacing_enabled()
            if pacing.isChecked() != pacing_checked:
                pacing.blockSignals(True)
                pacing.setChecked(pacing_checked)
                pacing.blockSignals(False)

        budget_spin = getattr(host, "discovery_budget_spin", None)
        if budget_spin is not None:
            budget_spin.blockSignals(True)
            budget_spin.setValue(get_ddg_session_budget_override())
            budget_spin.blockSignals(False)
            host._discovery_budget_last_applied = get_ddg_session_budget_override()

        burst_status = getattr(host, "discovery_burst_usage_label", None)
        if burst_status is not None:
            burst = get_ddg_burst_budget_status()
            burst_min = max(1, (burst.window_seconds + 59) // 60)
            burst_status.setText(
                f"Burst ({burst_min} min rolling): {burst.used}/{burst.limit} live DDG queries"
            )

        budget_status = getattr(host, "discovery_budget_status_label", None)
        if budget_status is not None:
            session = get_ddg_session_budget_status()
            session_min = max(1, (session.window_seconds + 59) // 60)
            budget_status.setText(
                f"Session ({session_min} min rolling): {session.used}/{session.limit} live DDG queries"
            )

        advanced_panel = getattr(host, "advanced_discovery_panel", None)
        advanced_toggle = getattr(host, "advanced_discovery_toggle", None)
        if advanced_panel is not None and advanced_toggle is not None:
            advanced_visible = get_advanced_discovery_unlocked()
            if advanced_panel.isVisible() != advanced_visible:
                advanced_panel.setVisible(advanced_visible)
            if advanced_toggle.isChecked() != advanced_visible:
                advanced_toggle.blockSignals(True)
                advanced_toggle.setChecked(advanced_visible)
                advanced_toggle.blockSignals(False)

        conservative = getattr(host, "discovery_conservative_label", None)
        if conservative is not None:
            line = conservative_mode_summary()
            conservative.setText(line or "")
            conservative_visible = bool(line)
            if conservative.isVisible() != conservative_visible:
                conservative.setVisible(conservative_visible)

        searxng_field = getattr(host, "discovery_searxng_url_field", None)
        if searxng_field is not None:
            searxng_field.blockSignals(True)
            searxng_field.setText(get_discovery_searxng_base_url())
            searxng_field.blockSignals(False)

        policy_card = getattr(host, "discovery_policy_summary_card", None)
        if policy_card is not None:
            policy_card.refresh_theme(resolved_dark)
            policy_card.set_policy_lines(discovery_policy_summary_lines())

        privacy_card = getattr(host, "discovery_privacy_help_card", None)
        if privacy_card is not None:
            privacy_card.refresh_theme(resolved_dark)
            privacy_card.set_privacy_lines(what_leaves_device_lines())

        for row in section.findChildren(_DiscoveryPolicyRow):
            row.refresh_theme(host, is_dark=resolved_dark)

    _run_with_preserved_settings_scroll(host, _sync)


def build_web_discovery_policy_section(host, *, is_dark: bool) -> QWidget:
    coalesce_settings_is_dark(host, is_dark=is_dark)
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    intro = make_settings_hint(
        "Privacy tier and Hybrid Internet Mode live on Settings → Privacy & data. "
        "This page holds advanced discovery limits, provider setup, DDG usage, and SearXNG. "
        "Privacy-first web search: DuckDuckGo is the default primary provider. "
        "Live DDG queries are paced and limited by rolling burst/session windows "
        "(heuristic defaults — not official DuckDuckGo quotas)."
    )
    layout.addWidget(intro)

    controls = QWidget()
    controls_form = QFormLayout(controls)
    controls_form.setContentsMargins(0, 0, 0, 0)
    controls_form.setSpacing(8)

    _privacy_tiers = (TIER_PRIVATE, TIER_BALANCED, TIER_ENHANCED, TIER_SEARXNG)
    host.discovery_privacy_tier_selector = SelectorButton(
        privacy_tier_label(get_discovery_privacy_tier()),
        is_dark=is_dark,
    )
    host.discovery_privacy_tier_selector.setMaximumWidth(280)
    host.discovery_privacy_tier_selector.setMenu(
        QMenu(host.discovery_privacy_tier_selector)
    )
    host.discovery_privacy_tier_selector.setToolTip(
        "Balance privacy vs optional API fallbacks for @internet and general web search."
    )
    register_settings_selector_width(
        host.discovery_privacy_tier_selector,
        *[privacy_tier_label(tier) for tier in _privacy_tiers],
    )
    controls_form.addRow("Privacy tier", host.discovery_privacy_tier_selector)

    host.discovery_privacy_tier_description = QLabel()
    host.discovery_privacy_tier_description.setWordWrap(True)
    host.discovery_privacy_tier_description.setObjectName("SettingsLogDescription")
    add_settings_full_width_row(controls_form, host.discovery_privacy_tier_description)
    add_open_privacy_data_button(host, controls_form)

    _pacing_tip = (
        "Adds a short gap between DDG HTTP requests to reduce bot challenges."
    )
    host.discovery_pacing_toggle = PrestigeToggle()
    host.discovery_pacing_label = QLabel(
        "Slow down live DuckDuckGo searches slightly (recommended)"
    )
    host.discovery_pacing_label.setWordWrap(True)
    host.discovery_pacing_toggle.setToolTip(_pacing_tip)
    host.discovery_pacing_label.setToolTip(_pacing_tip)
    pacing_row = QWidget()
    pacing_row_layout = QHBoxLayout(pacing_row)
    pacing_row_layout.setContentsMargins(0, 0, 0, 0)
    pacing_row_layout.addWidget(
        host.discovery_pacing_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    pacing_row_layout.addWidget(host.discovery_pacing_label, stretch=1)
    host.discovery_pacing_toggle.blockSignals(True)
    host.discovery_pacing_toggle.setChecked(get_discovery_pacing_enabled())
    host.discovery_pacing_toggle.blockSignals(False)
    host.discovery_pacing_toggle.toggled.connect(host._on_discovery_pacing_toggled)
    add_settings_full_width_row(controls_form, pacing_row)

    host.discovery_burst_usage_label = QLabel()
    host.discovery_burst_usage_label.setObjectName("SettingsLogDescription")
    controls_form.addRow("Live DDG usage", host.discovery_burst_usage_label)

    host.discovery_budget_status_label = QLabel()
    host.discovery_budget_status_label.setObjectName("SettingsLogDescription")
    add_settings_full_width_row(controls_form, host.discovery_budget_status_label)

    limits_hint = make_settings_hint(
        "Limits apply only to live DuckDuckGo HTTP calls (not cache hits or fallbacks). "
        "After a limit is reached, Wikipedia and other fallbacks continue to work."
    )
    add_settings_full_width_row(controls_form, limits_hint)

    _adv_discovery_tip = (
        "Override session query limits. Raising limits increases bot-challenge risk."
    )
    host.advanced_discovery_toggle = PrestigeToggle()
    host.advanced_discovery_label = QLabel("Show advanced discovery limits")
    host.advanced_discovery_toggle.setToolTip(_adv_discovery_tip)
    host.advanced_discovery_label.setToolTip(_adv_discovery_tip)
    adv_toggle_row = QWidget()
    adv_toggle_layout = QHBoxLayout(adv_toggle_row)
    adv_toggle_layout.setContentsMargins(0, 0, 0, 0)
    adv_toggle_layout.addWidget(
        host.advanced_discovery_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    adv_toggle_layout.addWidget(host.advanced_discovery_label, stretch=1)
    host.advanced_discovery_toggle.blockSignals(True)
    host.advanced_discovery_toggle.setChecked(get_advanced_discovery_unlocked())
    host.advanced_discovery_toggle.blockSignals(False)
    host.advanced_discovery_toggle.toggled.connect(host._on_advanced_discovery_toggled)
    add_settings_full_width_row(controls_form, adv_toggle_row)

    host.advanced_discovery_panel = QWidget()
    adv_panel_layout = QFormLayout(host.advanced_discovery_panel)
    adv_panel_layout.setContentsMargins(16, 0, 0, 0)
    adv_panel_layout.setSpacing(8)

    host.discovery_budget_spin = QSpinBox()
    host.discovery_budget_spin.setRange(0, 500)
    host.discovery_budget_spin.setSpecialValueText(
        f"Default ({DEFAULT_DDG_SESSION_BUDGET}/60 min)"
    )
    host.discovery_budget_spin.setToolTip(
        f"Maximum live DDG SERP HTTP calls per rolling 60-minute window. "
        f"0 uses the default of {DEFAULT_DDG_SESSION_BUDGET}. "
        "Unlimited is not available in the UI."
    )
    host.discovery_budget_spin.valueChanged.connect(host._on_discovery_budget_override_changed)
    host._discovery_budget_last_applied = get_ddg_session_budget_override()
    adv_panel_layout.addRow("Session limit override", host.discovery_budget_spin)

    add_settings_full_width_row(adv_panel_layout, make_settings_hint(
            "Burst limit is fixed at 6 live queries per 10 minutes. "
            "Lowering the session limit is always allowed; raising above the "
            f"default ({DEFAULT_DDG_SESSION_BUDGET}) requires confirmation."
        ),
    )
    host.advanced_discovery_panel.setVisible(get_advanced_discovery_unlocked())
    add_settings_full_width_row(controls_form, host.advanced_discovery_panel)

    host.discovery_searxng_url_field = QLineEdit()
    host.discovery_searxng_url_field.setPlaceholderText("https://search.example.org")
    host.discovery_searxng_url_field.setToolTip(
        "Base URL of your SearXNG instance (used with the SearXNG privacy tier)."
    )
    host.discovery_searxng_url_field.editingFinished.connect(
        host._on_discovery_searxng_url_changed
    )
    host.discovery_searxng_setup_btn = QPushButton("Set up SearXNG…")
    host.discovery_searxng_setup_btn.setToolTip(
        "Detect local instances, test connectivity, and apply the SearXNG privacy tier."
    )
    host.discovery_searxng_setup_btn.clicked.connect(host._on_searxng_setup_wizard_clicked)
    searxng_url_row = QWidget()
    searxng_url_row_layout = QHBoxLayout(searxng_url_row)
    searxng_url_row_layout.setContentsMargins(0, 0, 0, 0)
    searxng_url_row_layout.setSpacing(8)
    searxng_url_row_layout.addWidget(host.discovery_searxng_url_field, stretch=1)
    searxng_url_row_layout.addWidget(host.discovery_searxng_setup_btn)
    controls_form.addRow("SearXNG base URL", searxng_url_row)

    host.discovery_conservative_label = QLabel()
    host.discovery_conservative_label.setWordWrap(True)
    host.discovery_conservative_label.setObjectName("SettingsLogDescription")
    add_settings_full_width_row(controls_form, host.discovery_conservative_label)

    host.discovery_reset_health_btn = QPushButton("Reset discovery health")
    host.discovery_reset_health_btn.setToolTip(
        "Clear conservative pacing and challenge counters after network issues resolve."
    )
    host.discovery_reset_health_btn.clicked.connect(host._on_discovery_reset_health_clicked)
    add_settings_full_width_row(controls_form, host.discovery_reset_health_btn)

    layout.addWidget(controls)

    host.discovery_privacy_help_card = _DiscoveryInfoCard(
        title="What leaves your device",
        variant="privacy",
        is_dark=is_dark,
    )
    layout.addWidget(host.discovery_privacy_help_card)

    providers_block = QWidget()
    providers_layout = QVBoxLayout(providers_block)
    providers_layout.setContentsMargins(0, 0, 0, 0)
    providers_layout.setSpacing(10)

    providers_layout.addWidget(
        make_settings_hint(
            "Discovery providers used for web search, in priority order for the "
            "active privacy tier."
        )
    )

    primary_row = _DiscoveryPolicyRow(
        host,
        provider_id=PRIMARY_DISCOVERY_PROVIDER_ID,
        role_label="Primary",
        description="Default for @internet / general web when tier uses DuckDuckGo.",
        is_dark=is_dark,
    )
    host.discovery_primary_provider_card = primary_row._card
    providers_layout.addWidget(primary_row)
    brave_row = _DiscoveryPolicyRow(
        host,
        provider_id=BRAVE_DISCOVERY_PROVIDER_ID,
        role_label="Fallback",
        description=(
            "Full web SERP via Brave Search API when tier allows API fallback. "
            "Also used as primary for site-biased @recipe queries when configured."
        ),
        is_dark=is_dark,
        show_configure=True,
        configure_handler=host._on_brave_search_configure_clicked,
        configure_tooltip="Add or update your Brave Search API key.",
    )
    host.discovery_brave_configure_btn = brave_row.configure_btn
    providers_layout.addWidget(brave_row)
    searxng_row = _DiscoveryPolicyRow(
        host,
        provider_id=SEARXNG_DISCOVERY_PROVIDER_ID,
        role_label="Optional",
        description=(
            "Your self-hosted SearXNG instance (privacy tier: Self-hosted). "
            "Upstream engines depend on your server configuration."
        ),
        is_dark=is_dark,
        show_configure=True,
        configure_handler=host._on_searxng_setup_wizard_clicked,
        configure_tooltip="Open the SearXNG setup wizard (detect, test, configure).",
    )
    host.discovery_searxng_configure_btn = searxng_row.configure_btn
    providers_layout.addWidget(searxng_row)
    wiki_row = _DiscoveryPolicyRow(
        host,
        provider_id=WIKIPEDIA_DISCOVERY_PROVIDER_ID,
        role_label="Fallback",
        description=(
            "Wikipedia article search when earlier providers fail. "
            "Best for encyclopedic queries; site bias is stripped."
        ),
        is_dark=is_dark,
    )
    host.discovery_wikipedia_provider_card = wiki_row._card
    providers_layout.addWidget(wiki_row)

    host.discovery_policy_summary_card = _DiscoveryInfoCard(
        title="Active discovery route",
        variant="policy",
        is_dark=is_dark,
    )
    providers_layout.addWidget(host.discovery_policy_summary_card)

    layout.addWidget(wrap_subsection(providers_block, anchor="web_discovery_policy"))
    host.web_discovery_policy_section = container
    if hasattr(host, "_build_discovery_privacy_tier_menu"):
        host._build_discovery_privacy_tier_menu()
    sync_web_discovery_policy_section(host, is_dark=is_dark)
    return container


def build_knowledge_web_discovery_section(host, *, is_dark: bool) -> QWidget:
    """Subsection wrapper for settings layout."""
    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    card_form = add_settings_card_form(card_layout)
    add_subsection_to_form(card_form, "Web search discovery", anchor="web_discovery")
    add_settings_span_row(
        card_form,
        wrap_subsection(
            build_web_discovery_policy_section(host, is_dark=is_dark),
            anchor="web_discovery",
        ),
    )
    return card


def build_what_leaves_device_info_card(*, is_dark: bool) -> _DiscoveryInfoCard:
    """Shared 'What leaves your device' card for Privacy & data and Knowledge."""
    card = _DiscoveryInfoCard(
        title="What leaves your device",
        variant="privacy",
        is_dark=is_dark,
    )
    card.set_privacy_lines(what_leaves_device_lines())
    return card


def sync_what_leaves_device_info_card(
    card: _DiscoveryInfoCard | None,
    *,
    is_dark: bool | None = None,
) -> None:
    if card is None:
        return
    if is_dark is not None:
        card.refresh_theme(is_dark)
    card.set_privacy_lines(what_leaves_device_lines())
