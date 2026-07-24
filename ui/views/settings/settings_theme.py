"""Theme resolution and icon tint helpers for Settings UI."""

from __future__ import annotations

from core.theme.accessors import theme_for
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    MUTED_ICON,
    PLACEHOLDER_MUTED,
    SETTINGS_CHEVRON_DISABLED,
    SETTINGS_CHEVRON_ENABLED,
    SETTINGS_DIVIDER,
    SETTINGS_NAV_ICON,
    SETTINGS_WARNING_LABEL,
    WARNING_STATUS,
)


def resolve_settings_theme(host=None, *, is_dark: bool | None = None):
    """Resolve the active theme for Settings widgets."""
    if host is not None:
        return view_resolved_theme(host, is_dark=is_dark)
    return theme_for(is_dark=bool(is_dark if is_dark is not None else True))


def settings_nav_icon_color(theme) -> str:
    return theme.color(SETTINGS_NAV_ICON)


def settings_chevron_color(theme, *, enabled: bool) -> str:
    role = SETTINGS_CHEVRON_ENABLED if enabled else SETTINGS_CHEVRON_DISABLED
    return theme.color(role)


def settings_info_icon_color(theme) -> str:
    return theme.color(MUTED_ICON)


def settings_hint_icon_color(theme) -> str:
    return theme.color(WARNING_STATUS)


def settings_preview_icon_color(theme) -> str:
    return theme.color(MUTED_ICON)


def settings_divider_color(theme) -> str:
    return theme.color(SETTINGS_DIVIDER)


def settings_placeholder_color(theme) -> str:
    return theme.color(PLACEHOLDER_MUTED)


def style_bootstrap_warning_label(label, theme) -> None:
    label.setStyleSheet(theme.style(SETTINGS_WARNING_LABEL))
