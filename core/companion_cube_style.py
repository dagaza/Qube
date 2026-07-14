"""Qube cube companion visual style (classic holographic vs splash wireframe)."""

from __future__ import annotations

from enum import Enum


class CompanionCubeStyle(str, Enum):
    CLASSIC = "classic"
    EXPERIMENTAL = "experimental"


CUBE_STYLE_LABELS: dict[CompanionCubeStyle, str] = {
    CompanionCubeStyle.CLASSIC: "Classic",
    CompanionCubeStyle.EXPERIMENTAL: "Experimental",
}

CUBE_STYLE_DESCRIPTIONS: dict[CompanionCubeStyle, str] = {
    CompanionCubeStyle.CLASSIC: (
        "Dan's holographic layered cube — soft faces, particles, and premium glow."
    ),
    CompanionCubeStyle.EXPERIMENTAL: (
        "Splash-style wireframe cube with the fixed Q tail (matches download/processing)."
    ),
}

DEFAULT_COMPANION_CUBE_STYLE = CompanionCubeStyle.CLASSIC


def normalize_companion_cube_style(
    value: str | CompanionCubeStyle | None,
) -> CompanionCubeStyle:
    if isinstance(value, CompanionCubeStyle):
        return value
    raw = str(value or "").strip().lower()
    for style in CompanionCubeStyle:
        if style.value == raw:
            return style
    return DEFAULT_COMPANION_CUBE_STYLE
