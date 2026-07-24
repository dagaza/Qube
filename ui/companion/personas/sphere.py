"""Organic sphere companion persona."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPhase
from core.companion_personas import CompanionPersonaId
from ui.companion.persona_audio import paint_ripples, paint_waveform_ring
from ui.companion.companion_theme import persona_shine_qcolor
from ui.companion.persona_context import CompanionPaintContext
from ui.companion.personas.base import CompanionPersonaRenderer

_ORBIT_DOT_COUNT = 5


class SpherePersonaRenderer(CompanionPersonaRenderer):
    persona_id = CompanionPersonaId.SPHERE

    def halo_extra_px(self, body_radius: float) -> int:
        return int(body_radius * 0.45)

    def paint(self, painter: QPainter, ctx: CompanionPaintContext) -> None:
        center = QPointF(ctx.center_x, ctx.center_y + ctx.float_offset_y)
        radius = ctx.body_radius
        breathe = ctx.breathe
        primary, secondary = ctx.primary, ctx.secondary
        opacity = ctx.opacity * ctx.persona_blend

        self._paint_aurora(painter, center, radius, breathe, primary, secondary, opacity, ctx)
        paint_ripples(painter, ctx)
        paint_waveform_ring(painter, ctx)
        self._paint_working_energy(painter, center, radius * breathe, primary, secondary, opacity, ctx)
        self._paint_core(painter, center, radius, breathe, primary, secondary, opacity, ctx)

    def _paint_aurora(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        breathe: float,
        primary: QColor,
        secondary: QColor,
        opacity: float,
        ctx: CompanionPaintContext,
    ) -> None:
        if not ctx.reduced_motion:
            shift = 0.5 + 0.5 * math.sin(ctx.anim_time * 1.1)
            mix = QColor(
                int(primary.red() * (1 - shift) + secondary.red() * shift),
                int(primary.green() * (1 - shift) + secondary.green() * shift),
                int(primary.blue() * (1 - shift) + secondary.blue() * shift),
            )
        else:
            mix = secondary

        glow_r = radius * (1.55 + 0.12 * math.sin(ctx.anim_time * 2.0)) * breathe
        gradient = QRadialGradient(center, glow_r)
        inner = QColor(mix)
        inner.setAlphaF(0.55 * opacity)
        mid = QColor(primary)
        mid.setAlphaF(0.28 * opacity)
        outer = QColor(primary)
        outer.setAlphaF(0.0)
        gradient.setColorAt(0.0, inner)
        gradient.setColorAt(0.45, mid)
        gradient.setColorAt(1.0, outer)
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, glow_r, glow_r)

    def _paint_working_energy(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        primary: QColor,
        secondary: QColor,
        opacity: float,
        ctx: CompanionPaintContext,
    ) -> None:
        if ctx.activity != AssistantActivity.WORKING or ctx.reduced_motion:
            return

        phase = ctx.phase
        for i in range(3):
            spin = ctx.rotation * (1.0 + i * 0.35) + i * (2 * math.pi / 3)
            arc_r = radius * (0.62 + i * 0.12)
            span = 90 if phase != AssistantPhase.STT else 55
            color = secondary if i % 2 else primary
            color.setAlphaF(0.35 * opacity)
            painter.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(center.x() - arc_r, center.y() - arc_r, arc_r * 2, arc_r * 2)
            painter.drawArc(rect, int(math.degrees(spin) * 16), span * 16)

        for i in range(_ORBIT_DOT_COUNT):
            angle = ctx.rotation * 1.4 + i * (2 * math.pi / _ORBIT_DOT_COUNT)
            dist = radius * (1.08 + 0.06 * math.sin(ctx.anim_time * 2.5 + i))
            dot = QPointF(center.x() + math.cos(angle) * dist, center.y() + math.sin(angle) * dist)
            dot_color = QColor(secondary)
            dot_color.setAlphaF(0.75 * opacity)
            painter.setBrush(dot_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(dot, 2.8, 2.8)

    def _paint_core(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        breathe: float,
        primary: QColor,
        secondary: QColor,
        opacity: float,
        ctx: CompanionPaintContext,
    ) -> None:
        core_r = radius * 0.74 * breathe

        if ctx.activity == AssistantActivity.WORKING and not ctx.reduced_motion:
            spin_grad = QConicalGradient(center, math.degrees(ctx.rotation) + 90)
            c1 = QColor(primary)
            c1.setAlphaF(0.95 * opacity)
            c2 = QColor(secondary)
            c2.setAlphaF(0.85 * opacity)
            spin_grad.setColorAt(0.0, c1)
            spin_grad.setColorAt(0.45, c2)
            spin_grad.setColorAt(1.0, c1)
            painter.setBrush(QBrush(spin_grad))
        else:
            body = QRadialGradient(center - QPointF(core_r * 0.22, core_r * 0.28), core_r * 1.2)
            highlight = persona_shine_qcolor(ctx, alpha=90)
            mid = QColor(primary)
            mid.setAlphaF(0.92 * opacity)
            deep = QColor(primary.darker(125))
            deep.setAlphaF(0.98 * opacity)
            body.setColorAt(0.0, highlight)
            body.setColorAt(0.35, mid)
            body.setColorAt(1.0, deep)
            painter.setBrush(QBrush(body))

        painter.setPen(QPen(persona_shine_qcolor(ctx, alpha=50), 1.2))
        painter.drawEllipse(center, core_r, core_r)

        if ctx.activity == AssistantActivity.SPEAKING and ctx.speech_level_smooth > 0.08:
            wobble = core_r * (0.08 + ctx.speech_level_smooth * 0.14)
            path = QPainterPath()
            steps = 24
            for i in range(steps + 1):
                angle = (2 * math.pi * i) / steps
                noise = 1.0 + 0.18 * ctx.speech_level_smooth * math.sin(
                    angle * 5 + ctx.anim_time * 8
                )
                r = core_r + wobble * (noise - 1.0)
                pt = QPointF(center.x() + math.cos(angle) * r, center.y() + math.sin(angle) * r)
                if i == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            path.closeSubpath()
            distort = QColor(secondary)
            distort.setAlphaF(0.22 * ctx.speech_level_smooth * opacity)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(distort, 1.5))
            painter.drawPath(path)

        if ctx.activity == AssistantActivity.NEEDS_ATTENTION:
            pulse = 0.5 + 0.5 * math.sin(ctx.anim_time * 4.5)
            ring = QColor(primary)
            ring.setAlphaF(0.35 + pulse * 0.35)
            painter.setPen(QPen(ring, 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                center,
                core_r * (1.08 + pulse * 0.06),
                core_r * (1.08 + pulse * 0.06),
            )
