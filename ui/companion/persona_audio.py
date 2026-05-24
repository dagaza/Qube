"""Reusable audio-reactive paint helpers for companion personas."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen

from core.assistant_activity import AssistantActivity
from ui.companion.anim_engine import WAVE_BAR_COUNT
from ui.companion.persona_context import CompanionPaintContext


def paint_ripples(painter: QPainter, ctx: CompanionPaintContext) -> None:
    center = QPointF(ctx.center_x, ctx.center_y + ctx.float_offset_y)
    for age, strength in ctx.ripple_rings:
        ring = QColor(ctx.primary)
        ring.setAlphaF(max(0.0, (1.0 - age / 1.2) * 0.45 * strength * ctx.opacity))
        painter.setPen(QPen(ring, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        r = ctx.body_radius * (0.9 + age * 0.55)
        painter.drawEllipse(center, r, r)


def paint_waveform_ring(
    painter: QPainter,
    ctx: CompanionPaintContext,
    *,
    inner_radius: float | None = None,
) -> None:
    if ctx.activity not in (AssistantActivity.SPEAKING, AssistantActivity.CAPTURING):
        return

    cx = ctx.center_x
    cy = ctx.center_y + ctx.float_offset_y
    radius = ctx.body_radius * ctx.breathe
    inner_r = radius * 1.02 if inner_radius is None else inner_radius
    for i, level in enumerate(ctx.wave_bars):
        angle = (2 * math.pi * i / WAVE_BAR_COUNT) + ctx.rotation * 0.5
        bar_len = radius * (0.12 + level * 0.38)
        x1 = cx + math.cos(angle) * inner_r
        y1 = cy + math.sin(angle) * inner_r
        x2 = cx + math.cos(angle) * (inner_r + bar_len)
        y2 = cy + math.sin(angle) * (inner_r + bar_len)
        t = i / WAVE_BAR_COUNT
        bar_color = QColor(
            int(ctx.primary.red() * (1 - t) + ctx.secondary.red() * t),
            int(ctx.primary.green() * (1 - t) + ctx.secondary.green() * t),
            int(ctx.primary.blue() * (1 - t) + ctx.secondary.blue() * t),
        )
        bar_color.setAlphaF(0.35 + level * 0.55 * ctx.opacity)
        painter.setPen(QPen(bar_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
