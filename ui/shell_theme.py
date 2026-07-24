"""Theme helpers for MainWindow shell chrome and frameless dialogs."""

from __future__ import annotations

from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import (
    MUTED_ICON,
    NAV_ICON_ACTIVE,
    NAV_ICON_INACTIVE,
    RAG_INDICATOR_STANDBY,
    RETRIEVAL_INDICATOR_ACTIVE,
    RETRIEVAL_INDICATOR_OFF,
    SETTINGS_CHEVRON_DISABLED,
    SETTINGS_CHEVRON_ENABLED,
    SETTINGS_NAV_ICON,
    SETTINGS_PRESTIGE_MENU,
    TELEMETRY_CPU,
    TELEMETRY_GPU,
    TELEMETRY_RAM,
    THEME_TOGGLE_MOON,
    THEME_TOGGLE_SUN,
    WEB_INDICATOR_STANDBY,
    settings_prestige_menu_palette,
)


def resolve_shell_theme(host=None, *, is_dark: bool | None = None) -> ResolvedTheme:
    """Resolve the active theme from a MainWindow, view, or explicit mode."""
    if host is not None and hasattr(host, "theme_manager"):
        if is_dark is None:
            return host.theme_manager.current
        return theme_for(is_dark=is_dark)
    window = host.window() if host is not None and hasattr(host, "window") else None
    if window is not None and hasattr(window, "theme_manager"):
        if is_dark is None:
            return window.theme_manager.current
        return theme_for(is_dark=is_dark)
    return theme_for(is_dark=bool(is_dark if is_dark is not None else True))


def retrieval_indicator_colors(theme: ResolvedTheme) -> dict[str, str]:
    return {
        "off": theme.color(RETRIEVAL_INDICATOR_OFF),
        "active": theme.color(RETRIEVAL_INDICATOR_ACTIVE),
        "rag_standby": theme.color(RAG_INDICATOR_STANDBY),
        "web_standby": theme.color(WEB_INDICATOR_STANDBY),
        "ddg_backoff": theme.warning,
    }


def nav_icon_colors(theme: ResolvedTheme) -> tuple[str, str]:
    return theme.color(NAV_ICON_ACTIVE), theme.color(NAV_ICON_INACTIVE)


def telemetry_metric_colors(theme: ResolvedTheme) -> tuple[str, str, str]:
    return (
        theme.color(TELEMETRY_CPU),
        theme.color(TELEMETRY_RAM),
        theme.color(TELEMETRY_GPU),
    )


def theme_toggle_icon_colors(theme: ResolvedTheme) -> tuple[str, str]:
    return theme.color(THEME_TOGGLE_MOON), theme.color(THEME_TOGGLE_SUN)


def chevron_colors(theme: ResolvedTheme, *, enabled: bool) -> str:
    role = SETTINGS_CHEVRON_ENABLED if enabled else SETTINGS_CHEVRON_DISABLED
    return theme.color(role)


def muted_icon_color(theme: ResolvedTheme) -> str:
    return theme.color(MUTED_ICON)


def sidebar_row_action_icon_color(
    theme: ResolvedTheme,
    *,
    highlighted: bool = False,
) -> str:
    """Chevron/ellipsis on sidebar list rows; brighter when folder/session is highlighted."""
    if highlighted:
        return theme.list_row_title_selected
    return muted_icon_color(theme)


def accent_icon_color(theme: ResolvedTheme) -> str:
    return theme.color(SETTINGS_NAV_ICON)


def apply_prestige_menu_theme(menu, theme: ResolvedTheme) -> None:
    from PyQt6.QtGui import QPalette

    colors = settings_prestige_menu_palette(theme)
    bg = theme.qcolor(colors["bg"])
    fg = theme.qcolor(colors["fg"])
    sel_bg = theme.qcolor(colors["sel_bg"])
    sel_fg = theme.qcolor(colors["sel_fg"])

    palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
        palette.setColor(role, bg)
    palette.setColor(QPalette.ColorRole.WindowText, fg)
    palette.setColor(QPalette.ColorRole.Text, fg)
    palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
    palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)
    menu.setPalette(palette)
    menu.setStyleSheet(theme.style(SETTINGS_PRESTIGE_MENU))


def vu_meter_palette(theme: ResolvedTheme) -> dict[str, str]:
    return {
        "track_idle": theme.surface_elevated if theme.is_dark else theme.surface_pressed,
        "track_pulse": theme.surface_pressed,
        "accent": theme.accent,
        "accent_hover": theme.accent_hover,
        "accent_muted": theme.accent_secondary,
        "gradient_start": theme.success,
        "gradient_mid": theme.warning,
        "gradient_end": theme.error,
        "progress_track": with_alpha(theme.text_primary, 0.08 if theme.is_dark else 0.06),
    }
