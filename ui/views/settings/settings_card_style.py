"""Theme-aware card shells for Settings mainstage (Model Manager surface parity)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QSizePolicy

from core.theme.accessors import theme_for
from core.theme.widget_styles import SETTINGS_SECTION_CARD

_SETTINGS_SECTION_CARD = "SettingsSectionCard"

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


def begin_settings_section_card(host, *, is_dark: bool) -> tuple[QFrame, QVBoxLayout]:
    """Create a registered settings section card with an empty inner layout."""
    card = QFrame()
    card.setMinimumWidth(0)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    apply_settings_section_card_theme(card, is_dark=is_dark)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*SETTINGS_SECTION_CARD_CONTENT_MARGINS)
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
