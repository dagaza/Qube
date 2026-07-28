"""Theme helpers for the desktop companion subsystem."""

from __future__ import annotations

from PyQt6.QtGui import QColor

from core.assistant_activity import AssistantActivity
from core.companion_idle_color import CompanionIdleColor, normalize_companion_idle_color
from core.theme.accessors import theme_for
from core.theme.color_utils import adjust_lightness, parse_color, with_alpha
from core.theme.tokens import ResolvedTheme
from ui.companion.persona_context import CompanionPaintContext

_ACTIVE_GLOW_ALPHA = 90
_ACTIVE_TEXT_ALPHA = 210
_IDLE_TEXT_ALPHA = 55


def resolve_companion_theme(
    *,
    is_dark: bool = True,
    theme: ResolvedTheme | None = None,
) -> ResolvedTheme:
    return theme if theme is not None else theme_for(is_dark=is_dark)


def companion_idle_color_pair(
    idle_color: CompanionIdleColor | str | None,
    theme: ResolvedTheme,
) -> tuple[str, str]:
    preset = normalize_companion_idle_color(idle_color)
    if preset == CompanionIdleColor.BLUE:
        secondary = adjust_lightness(theme.link, 0.08 if theme.is_dark else -0.05)
        return theme.link, secondary
    return theme.accent, theme.accent_hover


def activity_color_pair(
    activity: AssistantActivity,
    idle_color: CompanionIdleColor | str | None = None,
    *,
    is_dark: bool = True,
    theme: ResolvedTheme | None = None,
) -> tuple[str, str]:
    """Resolve primary/secondary colors for a companion activity."""
    resolved = resolve_companion_theme(is_dark=is_dark, theme=theme)
    if activity == AssistantActivity.IDLE_LISTEN:
        return companion_idle_color_pair(idle_color, resolved)

    pairs: dict[AssistantActivity, tuple[str, str]] = {
        AssistantActivity.ASSISTANT_OFF: (resolved.text_muted, resolved.text_secondary),
        AssistantActivity.CAPTURING: (resolved.error, adjust_lightness(resolved.warning, 0.05)),
        AssistantActivity.WORKING: (resolved.info, resolved.link),
        AssistantActivity.SPEAKING: (
            resolved.success,
            adjust_lightness(resolved.info, 0.05 if resolved.is_dark else -0.04),
        ),
        AssistantActivity.NEEDS_ATTENTION: (
            resolved.warning,
            adjust_lightness(resolved.warning, -0.08 if resolved.is_dark else 0.06),
        ),
        AssistantActivity.ERROR: (
            resolved.error,
            adjust_lightness(resolved.error, 0.12 if resolved.is_dark else -0.06),
        ),
        AssistantActivity.BACKGROUND_BUSY: (resolved.accent, resolved.accent_hover),
    }
    return pairs.get(activity, companion_idle_color_pair(idle_color, resolved))


def companion_caption_stylesheet(theme: ResolvedTheme) -> str:
    border = theme.border_subtle if theme.is_dark else theme.border
    return (
        f"QFrame#CompanionCaptionFrame {{ background-color: {theme.background};"
        f" border: 1px solid {border}; border-radius: 8px; }}"
        f"QLabel#CompanionCaptionLabel {{ background: transparent; color: {theme.text_primary}; }}"
    )


def companion_dock_strip_background(theme: ResolvedTheme, *, idle_faded: bool) -> QColor:
    rgba = parse_color(theme.background)
    bg = QColor(rgba.r, rgba.g, rgba.b)
    bg.setAlphaF(0.35 if idle_faded else 0.85)
    return bg


def companion_snap_compass_stylesheet(theme: ResolvedTheme) -> str:
    surface = theme.surface_elevated if theme.is_dark else theme.surface
    hover = theme.surface_pressed if theme.is_dark else theme.surface_elevated
    border = with_alpha(theme.text_primary, 0.12 if theme.is_dark else 0.18)
    hover_border = with_alpha(theme.text_primary, 0.22 if theme.is_dark else 0.28)
    return f"""
            QToolButton#CompanionSnapCompassButton {{
                background: {with_alpha(surface, 0.85)};
                border: 1px solid {border};
                border-radius: 8px;
                color: {theme.text_secondary};
                font-size: 10px;
                font-weight: 600;
            }}
            QToolButton#CompanionSnapCompassButton:hover {{
                background: {with_alpha(hover, 0.95)};
                border-color: {hover_border};
            }}
            QToolButton#CompanionSnapCompassButton:checked {{
                background: {with_alpha(theme.accent, 0.35)};
                border-color: {with_alpha(theme.accent_hover, 0.85)};
                color: {theme.text_primary};
            }}
            """


def _qcolor_alpha(hex_or_rgba: str, alpha: int) -> QColor:
    rgba = parse_color(hex_or_rgba)
    return QColor(rgba.r, rgba.g, rgba.b, alpha)


def companion_snap_overlay_pen(theme: ResolvedTheme, *, active: bool) -> QColor:
    alpha = _ACTIVE_TEXT_ALPHA if active else _IDLE_TEXT_ALPHA
    return _qcolor_alpha(theme.text_secondary, alpha)


def companion_snap_overlay_glow(theme: ResolvedTheme) -> QColor:
    return _qcolor_alpha(theme.accent, _ACTIVE_GLOW_ALPHA)


def persona_wire_qcolor(ctx: CompanionPaintContext, *, layer: int, alpha: float) -> QColor:
    theme = theme_for(is_dark=ctx.is_dark)
    wire = QColor(theme.text_on_accent)
    mult = 0.55 if layer == 2 else 0.35
    wire.setAlphaF(min(1.0, alpha * ctx.opacity * mult))
    return wire


def persona_highlight_qcolor(ctx: CompanionPaintContext, *, alpha: float) -> QColor:
    theme = theme_for(is_dark=ctx.is_dark)
    color = QColor(theme.text_on_accent)
    color.setAlphaF(alpha * ctx.opacity)
    return color


def persona_shine_qcolor(ctx: CompanionPaintContext, *, alpha: int) -> QColor:
    theme = theme_for(is_dark=ctx.is_dark)
    rgba = parse_color(theme.text_on_accent)
    return QColor(rgba.r, rgba.g, rgba.b, int(alpha * ctx.opacity))
