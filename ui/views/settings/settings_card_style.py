"""Theme-aware card shells for Settings mainstage (Model Manager surface parity)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QVBoxLayout

_SETTINGS_SECTION_CARD = "SettingsSectionCard"


def apply_settings_section_card_theme(card: QFrame, *, is_dark: bool) -> None:
    """Paint a Model Manager-style panel surface on ``card``."""
    bg_hex = "#232337" if is_dark else "#E9EFF5"
    border = "rgba(255, 255, 255, 0.08)" if is_dark else "#dbe4ee"
    card.setObjectName(_SETTINGS_SECTION_CARD)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(
        f"#{_SETTINGS_SECTION_CARD} {{"
        f" background-color: {bg_hex};"
        f" border: 1px solid {border};"
        f" border-radius: 10px;"
        f" }}"
    )


def begin_settings_section_card(host, *, is_dark: bool) -> tuple[QFrame, QVBoxLayout]:
    """Create a registered settings section card with an empty inner layout."""
    card = QFrame()
    apply_settings_section_card_theme(card, is_dark=is_dark)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)
    register_settings_section_card(host, card)
    return card, layout


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
