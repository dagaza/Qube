"""Theme-aware styling for Web search discovery cards (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

from core.theme.accessors import theme_for
from core.theme.widget_styles import (
    DISCOVERY_BODY_TEXT,
    DISCOVERY_DIVIDER,
    DISCOVERY_INFO_BULLET,
    DISCOVERY_INFO_CARD,
    DISCOVERY_INFO_HIGHLIGHT,
    DISCOVERY_INFO_KV_KEY,
    DISCOVERY_INFO_KV_VALUE,
    DISCOVERY_INFO_STATUS,
    DISCOVERY_INFO_TITLE,
    DISCOVERY_PRIVACY_CHIP,
    DISCOVERY_PROVIDER_CARD,
    DISCOVERY_PROVIDER_NAME,
    DISCOVERY_ROLE_CHIP,
)

_ROLE_KEYS = {
    "primary": "primary",
    "fallback": "fallback",
    "optional": "optional",
}


def _theme(*, is_dark: bool):
    return theme_for(is_dark=is_dark)


def _repolish_widget(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(False)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _normalize_role(role_label: str) -> str:
    return _ROLE_KEYS.get(role_label.strip().lower(), "fallback")


def apply_discovery_provider_card_theme(
    card: QWidget, *, role_label: str, is_dark: bool
) -> None:
    role = _normalize_role(role_label)
    theme = _theme(is_dark=is_dark)
    card.setObjectName("DiscoveryProviderCard")
    card.setProperty("discovery_role", role)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(theme.style(DISCOVERY_PROVIDER_CARD, discovery_role=role))
    _repolish_widget(card)


def style_discovery_role_chip(label: QLabel, *, role_label: str, is_dark: bool) -> None:
    role = _normalize_role(role_label)
    theme = _theme(is_dark=is_dark)
    label.setObjectName("DiscoveryCardRoleChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(DISCOVERY_ROLE_CHIP, discovery_role=role))
    _repolish_widget(label)


def style_discovery_provider_name(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryCardProviderName")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_PROVIDER_NAME))


def style_discovery_privacy_chip(label: QLabel, *, is_dark: bool) -> None:
    theme = _theme(is_dark=is_dark)
    label.setObjectName("DiscoveryCardPrivacyChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(DISCOVERY_PRIVACY_CHIP))
    _repolish_widget(label)


def style_discovery_body_text(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryCardBody")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_BODY_TEXT))


def build_discovery_divider(*, is_dark: bool) -> QWidget:
    line = QWidget()
    line.setFixedHeight(1)
    line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    line.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_DIVIDER))
    return line


def apply_discovery_info_card_theme(
    card: QWidget, *, variant: str, is_dark: bool
) -> None:
    key = variant if variant in ("privacy", "policy") else "policy"
    card.setObjectName("DiscoveryInfoCard")
    card.setProperty("discovery_info_variant", key)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_CARD, variant=key))


def style_discovery_info_title(label: QLabel, *, variant: str, is_dark: bool) -> None:
    key = variant if variant in ("privacy", "policy") else "policy"
    label.setObjectName("DiscoveryInfoCardTitle")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_TITLE, variant=key))


def style_discovery_info_highlight(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoHighlight")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_HIGHLIGHT))


def style_discovery_info_kv_key(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoKvKey")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_KV_KEY))


def style_discovery_info_kv_value(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoKvValue")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_KV_VALUE))


def style_discovery_info_bullet(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoBullet")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_BULLET))


def style_discovery_info_status(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoStatus")
    label.setStyleSheet(_theme(is_dark=is_dark).style(DISCOVERY_INFO_STATUS))
