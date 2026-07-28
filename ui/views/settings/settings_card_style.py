"""Theme-aware card shells for Settings mainstage (Model Manager surface parity)."""

from __future__ import annotations

from dataclasses import dataclass

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from core.theme.accessors import theme_for
from core.theme.widget_styles import SETTINGS_SECTION_CARD

_SETTINGS_SECTION_CARD = "SettingsSectionCard"
_COLLAPSE_ICON = "fa5s.chevron-right"
_EXPAND_ICON = "fa5s.chevron-down"


@dataclass
class SettingsCollapsibleCardHandle:
    """One collapsible settings card on a settings page."""

    section_id: str
    wrapper: QWidget
    header: QWidget
    card: QFrame
    inner_layout: QVBoxLayout
    toggle_btn: QPushButton
    title_lbl: QLabel
    expanded: bool = True

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.card.setVisible(expanded)
        icon_name = _EXPAND_ICON if expanded else _COLLAPSE_ICON
        color = self.toggle_btn.property("_chevron_color") or "#89b4fa"
        self.toggle_btn.setIcon(qta.icon(icon_name, color=str(color)))
        tooltip = "Collapse section" if expanded else "Expand section"
        self.toggle_btn.setToolTip(tooltip)
        self.title_lbl.setToolTip(tooltip)

# Outer card shell padding (border → form host).
SETTINGS_SECTION_CARD_CONTENT_MARGINS = (16, 14, 16, 14)
# Inner form inset: matches the old empty-label + field-column rhythm (~12px).
SETTINGS_CARD_FORM_HORIZONTAL_INSET = 12


def settings_card_content_horizontal_inset() -> tuple[int, int]:
    """Total left/right inset from card outer edge to form content."""
    card_left, _, card_right, _ = SETTINGS_SECTION_CARD_CONTENT_MARGINS
    inset = SETTINGS_CARD_FORM_HORIZONTAL_INSET
    return card_left + inset, card_right + inset


def settings_card_content_horizontal_padding_total() -> int:
    """Sum of left + right inset (card shell + form) for preview width math."""
    left, right = settings_card_content_horizontal_inset()
    return left + right


def _settings_theme(*, is_dark: bool):
    return theme_for(is_dark=is_dark)


def apply_settings_section_card_theme(card: QFrame, *, is_dark: bool) -> None:
    """Paint a Model Manager-style panel surface on ``card``."""
    theme = _settings_theme(is_dark=is_dark)
    card.setObjectName(_SETTINGS_SECTION_CARD)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(
        theme.style(SETTINGS_SECTION_CARD, object_name=_SETTINGS_SECTION_CARD)
    )


def begin_settings_section_card(host, *, is_dark: bool) -> tuple[QWidget, QVBoxLayout]:
    """Create a settings section card (optionally wrapped with collapse chrome)."""
    card = QFrame()
    card.setMinimumWidth(0)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    apply_settings_section_card_theme(card, is_dark=is_dark)
    inner_layout = QVBoxLayout(card)
    inner_layout.setContentsMargins(*SETTINGS_SECTION_CARD_CONTENT_MARGINS)
    inner_layout.setSpacing(10)
    register_settings_section_card(host, card)

    section_id = str(getattr(host, "_current_settings_section_id", "") or "")
    wrapper, handle = _wrap_collapsible_card(
        host,
        section_id=section_id,
        card=card,
        inner_layout=inner_layout,
        is_dark=is_dark,
    )
    if handle is not None:
        inner_layout._settings_collapsible_handle = handle  # type: ignore[attr-defined]
        from core.app_settings import get_settings_section_cards_default_expanded

        handle.set_expanded(get_settings_section_cards_default_expanded())
    return wrapper, inner_layout


def register_settings_section_card(host, card: QFrame) -> None:
    cards = getattr(host, "_settings_section_cards", None)
    if cards is None:
        host._settings_section_cards = []
        cards = host._settings_section_cards
    cards.append(card)


def refresh_settings_section_cards(host, *, is_dark: bool) -> None:
    for card in getattr(host, "_settings_section_cards", ()) or ():
        if card is not None:
            apply_settings_section_card_theme(card, is_dark=is_dark)
    sync_settings_collapsible_cards(host, is_dark=is_dark)


def sync_settings_collapsible_cards(host, *, is_dark: bool) -> None:
    """Apply theme + visibility for collapsible card chrome from persisted settings."""
    from core.app_settings import get_settings_section_cards_collapsible

    enabled = get_settings_section_cards_collapsible()
    chevron_color = "#89b4fa" if is_dark else "#64748b"
    by_section = getattr(host, "_settings_collapsible_cards_by_section", {}) or {}
    for handles in by_section.values():
        for handle in handles:
            handle.header.setVisible(enabled)
            handle.toggle_btn.setProperty("_chevron_color", chevron_color)
            if enabled:
                handle.set_expanded(handle.expanded)
            else:
                handle.card.setVisible(True)
    if hasattr(host, "_sync_settings_collapse_all_button"):
        section_id = _current_settings_section_id(host)
        host._sync_settings_collapse_all_button(section_id)


def set_settings_collapsible_cards_expanded(
    host,
    section_id: str,
    *,
    expanded: bool,
) -> None:
    handles = (
        getattr(host, "_settings_collapsible_cards_by_section", {}) or {}
    ).get(section_id, ())
    for handle in handles:
        handle.set_expanded(expanded)
    if hasattr(host, "_sync_settings_collapse_all_button"):
        host._sync_settings_collapse_all_button(section_id)


def resolve_collapsible_handle_for_layout(layout) -> SettingsCollapsibleCardHandle | None:
    return getattr(layout, "_settings_collapsible_handle", None)


def _wrap_collapsible_card(
    host,
    *,
    section_id: str,
    card: QFrame,
    inner_layout: QVBoxLayout,
    is_dark: bool,
) -> tuple[QWidget, SettingsCollapsibleCardHandle | None]:
    wrapper = QWidget()
    wrapper.setObjectName("SettingsCollapsibleCard")
    wrapper_layout = QVBoxLayout(wrapper)
    wrapper_layout.setContentsMargins(0, 0, 0, 0)
    wrapper_layout.setSpacing(4)

    chevron_color = "#89b4fa" if is_dark else "#64748b"
    toggle_btn = QPushButton()
    toggle_btn.setObjectName("SettingsSectionCardToggle")
    toggle_btn.setFixedSize(30, 30)
    toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    toggle_btn.setStyleSheet("background: transparent; border: none;")
    toggle_btn.setProperty("_chevron_color", chevron_color)

    title_lbl = QLabel("")
    title_lbl.setObjectName("SettingsSubsectionLabel")

    header = QWidget()
    header.setObjectName("SettingsCollapsibleCardHeader")
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(4)
    header_layout.addWidget(
        toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    header_layout.addWidget(title_lbl, stretch=1)

    wrapper_layout.addWidget(header)
    wrapper_layout.addWidget(card)

    handle = SettingsCollapsibleCardHandle(
        section_id=section_id,
        wrapper=wrapper,
        header=header,
        card=card,
        inner_layout=inner_layout,
        toggle_btn=toggle_btn,
        title_lbl=title_lbl,
    )
    title_lbl._settings_collapsible_handle = handle  # type: ignore[attr-defined]

    toggle_btn.clicked.connect(
        lambda _checked=False, h=handle, view=host: _on_card_toggle_clicked(h, view)
    )

    by_section = getattr(host, "_settings_collapsible_cards_by_section", None)
    if by_section is None:
        host._settings_collapsible_cards_by_section = {}
        by_section = host._settings_collapsible_cards_by_section
    by_section.setdefault(section_id, []).append(handle)

    from core.app_settings import get_settings_section_cards_collapsible

    header.setVisible(get_settings_section_cards_collapsible())
    handle.set_expanded(True)
    return wrapper, handle


def _on_card_toggle_clicked(handle: SettingsCollapsibleCardHandle, host) -> None:
    handle.set_expanded(not handle.expanded)
    if hasattr(host, "_sync_settings_collapse_all_button"):
        host._sync_settings_collapse_all_button(handle.section_id or None)


def _current_settings_section_id(host) -> str | None:
    row = getattr(host, "settings_section_list", None)
    if row is None:
        return None
    current_row = row.currentRow()
    if current_row < 0:
        return None
    item = row.item(current_row)
    if item is None:
        return None
    role = getattr(host, "_SETTINGS_SECTION_ID_ROLE", None)
    if role is None:
        return None
    section_id = item.data(role)
    return str(section_id) if section_id else None
