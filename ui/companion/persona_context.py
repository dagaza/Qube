"""Shared paint context passed to companion persona renderers."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor

from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPhase


@dataclass(frozen=True)
class CompanionPaintContext:
    """Immutable snapshot of animation + presence state for one paint frame."""

    activity: AssistantActivity
    phase: AssistantPhase | None
    primary: QColor
    secondary: QColor
    center_x: float
    center_y: float
    body_radius: float
    breathe: float
    float_offset_y: float
    opacity: float
    anim_time: float
    rotation: float
    reduced_motion: bool
    is_dark: bool
    input_level: float
    speech_level_smooth: float
    wave_bars: tuple[float, ...]
    ripple_rings: tuple[tuple[float, float], ...]
    notify_pulse: float
    persona_blend: float = 1.0
