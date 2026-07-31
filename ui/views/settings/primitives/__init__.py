"""Reusable Settings UI primitives (typography, cards, chips, callouts).

Hierarchy (see Phase 1 typography refresh):
  L2 — ``make_settings_card_title``       card section title
  L3 — ``make_settings_group_header``     in-card group header
  L5 — ``make_settings_hint``             muted explanatory copy

Rich sections should prefer these helpers over one-off styling modules.
"""

from ui.views.settings.primitives.actions import (
    ACTION_COLUMN_WIDTH_PX,
    ACTION_CONTROL_HEIGHT_PX,
    ACTION_ROW_BOTTOM_INSET_PX,
    STATUS_COLUMN_WIDTH_PX,
    make_settings_action_row,
    style_settings_action_button,
    style_settings_configure_button,
    style_settings_free_button,
)
from ui.views.settings.primitives.callouts import (
    SettingsCallout,
    apply_settings_callout_theme,
    style_settings_access_hint,
)
from ui.views.settings.primitives.cards import (
    DEFAULT_POLICY_KV_KEYS,
    SettingsInfoCard,
    apply_settings_info_card_theme,
    apply_settings_nested_card_theme,
    build_settings_divider,
    refresh_settings_divider,
    style_settings_info_bullet,
    style_settings_info_card_title,
    style_settings_info_highlight,
    style_settings_info_kv_key,
    style_settings_info_kv_value,
    style_settings_info_status,
    style_settings_nested_card_body,
    style_settings_nested_card_title,
)
from ui.views.settings.primitives.chips import (
    style_settings_role_chip,
    style_settings_status_chip,
    style_settings_tag_chip,
)
from ui.views.settings.primitives.theme import (
    coalesce_settings_is_dark,
    repolish_widget,
    resolve_settings_is_dark,
    settings_theme,
)
from ui.views.settings.primitives.typography import (
    make_settings_card_title,
    make_settings_group_header,
    make_settings_group_label,
    make_settings_hint,
    make_subsection_label,
)

__all__ = [
    "ACTION_COLUMN_WIDTH_PX",
    "ACTION_CONTROL_HEIGHT_PX",
    "ACTION_ROW_BOTTOM_INSET_PX",
    "DEFAULT_POLICY_KV_KEYS",
    "STATUS_COLUMN_WIDTH_PX",
    "SettingsCallout",
    "SettingsInfoCard",
    "apply_settings_callout_theme",
    "apply_settings_info_card_theme",
    "apply_settings_nested_card_theme",
    "build_settings_divider",
    "coalesce_settings_is_dark",
    "make_settings_action_row",
    "make_settings_card_title",
    "make_settings_group_header",
    "make_settings_group_label",
    "make_settings_hint",
    "make_subsection_label",
    "refresh_settings_divider",
    "repolish_widget",
    "resolve_settings_is_dark",
    "settings_theme",
    "style_settings_access_hint",
    "style_settings_action_button",
    "style_settings_configure_button",
    "style_settings_free_button",
    "style_settings_info_bullet",
    "style_settings_info_card_title",
    "style_settings_info_highlight",
    "style_settings_info_kv_key",
    "style_settings_info_kv_value",
    "style_settings_info_status",
    "style_settings_nested_card_body",
    "style_settings_nested_card_title",
    "style_settings_role_chip",
    "style_settings_status_chip",
    "style_settings_tag_chip",
]
