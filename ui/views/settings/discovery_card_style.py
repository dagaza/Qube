"""Backward-compatible styling aliases for Web search discovery cards.

Prefer ``ui.views.settings.primitives`` for new Settings UI work.
"""

from __future__ import annotations

from ui.views.settings.primitives import (
    apply_settings_info_card_theme as apply_discovery_info_card_theme,
    apply_settings_nested_card_theme as apply_discovery_provider_card_theme,
    build_settings_divider as build_discovery_divider,
    style_settings_info_bullet as style_discovery_info_bullet,
    style_settings_info_card_title as style_discovery_info_title,
    style_settings_info_highlight as style_discovery_info_highlight,
    style_settings_info_kv_key as style_discovery_info_kv_key,
    style_settings_info_kv_value as style_discovery_info_kv_value,
    style_settings_info_status as style_discovery_info_status,
    style_settings_nested_card_body as style_discovery_body_text,
    style_settings_nested_card_title as style_discovery_provider_name,
    style_settings_role_chip as style_discovery_role_chip,
    style_settings_tag_chip as style_discovery_privacy_chip,
)

__all__ = [
    "apply_discovery_info_card_theme",
    "apply_discovery_provider_card_theme",
    "build_discovery_divider",
    "style_discovery_body_text",
    "style_discovery_info_bullet",
    "style_discovery_info_highlight",
    "style_discovery_info_kv_key",
    "style_discovery_info_kv_value",
    "style_discovery_info_status",
    "style_discovery_info_title",
    "style_discovery_privacy_chip",
    "style_discovery_provider_name",
    "style_discovery_role_chip",
]
