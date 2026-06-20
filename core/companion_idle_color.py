"""Companion idle (base) color presets — distinct from activity status colors."""

from __future__ import annotations

from enum import Enum


class CompanionIdleColor(str, Enum):
    PURPLE = "purple"
    BLUE = "blue"


IDLE_COLOR_LABELS: dict[CompanionIdleColor, str] = {
    CompanionIdleColor.PURPLE: "Qube purple",
    CompanionIdleColor.BLUE: "Soft blue",
}

IDLE_COLOR_DESCRIPTIONS: dict[CompanionIdleColor, str] = {
    CompanionIdleColor.PURPLE: "Brand violet — default idle glow for the companion.",
    CompanionIdleColor.BLUE: "Calm blue — the original companion idle colour.",
}

# Primary / secondary pairs for IDLE_LISTEN rendering.
# Kept separate from BACKGROUND_BUSY (#cba6f7) and WORKING (#74c7ec) status hues.
IDLE_COLOR_PAIRS: dict[CompanionIdleColor, tuple[str, str]] = {
    CompanionIdleColor.PURPLE: ("#8b5cf6", "#a78bfa"),
    CompanionIdleColor.BLUE: ("#89b4fa", "#b4befe"),
}

IDLE_COLOR_PULSE: dict[CompanionIdleColor, str] = {
    CompanionIdleColor.PURPLE: "#c4b5fd",
    CompanionIdleColor.BLUE: "#b4d0fb",
}

DEFAULT_COMPANION_IDLE_COLOR = CompanionIdleColor.PURPLE


def normalize_companion_idle_color(
    value: str | CompanionIdleColor | None,
) -> CompanionIdleColor:
    if isinstance(value, CompanionIdleColor):
        return value
    raw = str(value or "").strip().lower()
    for preset in CompanionIdleColor:
        if preset.value == raw:
            return preset
    return DEFAULT_COMPANION_IDLE_COLOR


def idle_color_pair(value: str | CompanionIdleColor | None = None) -> tuple[str, str]:
    preset = normalize_companion_idle_color(value)
    return IDLE_COLOR_PAIRS[preset]


def idle_color_primary_hex(value: str | CompanionIdleColor | None = None) -> str:
    return idle_color_pair(value)[0]


def idle_color_pulse_hex(value: str | CompanionIdleColor | None = None) -> str:
    preset = normalize_companion_idle_color(value)
    return IDLE_COLOR_PULSE[preset]
