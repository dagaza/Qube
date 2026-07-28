"""Readability overlay mapping — lazy scrim computed at render time."""

from __future__ import annotations

from dataclasses import dataclass

from core.surface_fill.constants import OverlayStrength
from core.surface_fill.models import OverlaySpec
from core.theme.color_utils import parse_color, with_alpha
from core.theme.tokens import ResolvedTheme

# UI: Original / Balanced / Muted. Original = artwork shows through brightest;
# Muted = strongest wash for readability. Reader focus steps toward Muted.
OVERLAY_STRENGTH_BY_INCREASING_INTENSITY: tuple[OverlayStrength, ...] = (
    "vivid",
    "balanced",
    "subtle",
)


@dataclass(frozen=True)
class OverlayRenderParams:
    scrim_opacity: float
    saturation_scale: float
    blur_px: float = 0.0


_STRENGTH_PARAMS: dict[OverlayStrength, tuple[tuple[float, float], float]] = {
    "subtle": ((0.55, 0.40), 0.55),
    "balanced": ((0.35, 0.25), 0.70),
    "vivid": ((0.15, 0.10), 0.85),
}


def overlay_strength_with_boost(
    strength: OverlayStrength,
    boost: int = 0,
) -> OverlayStrength:
    """Bump readability overlay one step (reader focus). Muted stays at Muted (max)."""
    if boost <= 0:
        return strength
    order = OVERLAY_STRENGTH_BY_INCREASING_INTENSITY
    try:
        index = order.index(strength)
    except ValueError:
        return "balanced"
    index = min(index + boost, len(order) - 1)
    return order[index]


def overlay_render_params(
    overlay: OverlaySpec,
    theme: ResolvedTheme,
    *,
    boost: int = 0,
) -> OverlayRenderParams:
    strength = overlay_strength_with_boost(overlay.strength, boost=boost)
    (dark_opacity, light_opacity), saturation = _STRENGTH_PARAMS[strength]
    opacity = dark_opacity if theme.is_dark else light_opacity
    return OverlayRenderParams(
        scrim_opacity=opacity,
        saturation_scale=saturation,
        blur_px=0.0,
    )


def overlay_scrim_rgba(
    overlay: OverlaySpec,
    theme: ResolvedTheme,
    *,
    boost: int = 0,
) -> str:
    """Compute scrim color from overlay strength and active theme (paint-time only)."""
    params = overlay_render_params(overlay, theme, boost=boost)
    base = theme.background
    rgba = parse_color(base)
    alpha = int(round(max(0.0, min(1.0, params.scrim_opacity)) * 255))
    return with_alpha(rgba.to_hex(), alpha / 255.0)
