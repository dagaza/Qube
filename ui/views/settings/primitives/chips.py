"""Status, role, and tag chips for Settings rich sections."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy

from core.theme.widget_styles import (
    DISCOVERY_PRIVACY_CHIP,
    DISCOVERY_ROLE_CHIP,
    KNOWLEDGE_ACCESS_BADGE,
)
from ui.views.settings.primitives.actions import ACTION_CONTROL_HEIGHT_PX
from ui.views.settings.primitives.theme import repolish_widget, settings_theme

_NESTED_CARD_ROLES = {
    "primary": "primary",
    "fallback": "fallback",
    "optional": "optional",
}


def _normalize_nested_role(role: str) -> str:
    return _NESTED_CARD_ROLES.get(role.strip().lower(), "fallback")


def style_settings_status_chip(
    label: QLabel,
    status: str,
    *,
    is_dark: bool,
    host=None,
) -> None:
    """Semantic status pill (connected, free, coming_soon, etc.)."""
    theme = settings_theme(is_dark=is_dark, host=host)
    label.setObjectName("KnowledgeAccessBadge")
    label.setProperty("access", status)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(KNOWLEDGE_ACCESS_BADGE, access=status))
    label.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    repolish_widget(label)


def style_settings_role_chip(
    label: QLabel,
    role: str,
    *,
    is_dark: bool,
) -> None:
    """Accent role chip for nested settings cards (primary / fallback / optional)."""
    normalized = _normalize_nested_role(role)
    theme = settings_theme(is_dark=is_dark)
    label.setObjectName("DiscoveryCardRoleChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(DISCOVERY_ROLE_CHIP, discovery_role=normalized))
    repolish_widget(label)


def style_settings_tag_chip(label: QLabel, *, is_dark: bool) -> None:
    """Muted inline tag chip (privacy notes, metadata tags)."""
    theme = settings_theme(is_dark=is_dark)
    label.setObjectName("DiscoveryCardPrivacyChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(DISCOVERY_PRIVACY_CHIP))
    repolish_widget(label)
