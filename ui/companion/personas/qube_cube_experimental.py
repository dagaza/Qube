"""Qube wireframe cube companion persona (matches splash processing cube)."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient

from core.assistant_activity import AssistantActivity
from core.companion_personas import CompanionPersonaId
from ui.companion.persona_audio import paint_ripples, paint_waveform_ring
from ui.companion.persona_context import CompanionPaintContext
from ui.companion.personas.base import CompanionPersonaRenderer
from ui.qube_wireframe_cube import (
    BRAND_STROKE_COLOR,
    build_qube_wireframe_segments,
    paint_qube_wireframe_cube,
    qube_wireframe_visual_extent,
)

_CUBE_BODY_SCALE = 0.88
_PARTICLE_COUNT = 10
_WAVEFORM_OUTER_INSET = 1.38
_COMPANION_STROKE_WIDTH_RATIO = 0.19


class QubeCubeExperimentalPersonaRenderer(CompanionPersonaRenderer):
    persona_id = CompanionPersonaId.QUBE

    def halo_extra_px(self, body_radius: float) -> int:
        return int(body_radius * 0.95)

    def visual_extent_px(self, body_radius: float) -> float:
        fit_radius = body_radius * _CUBE_BODY_SCALE
        return qube_wireframe_visual_extent(fit_radius) + 8.0

    def paint(self, painter: QPainter, ctx: CompanionPaintContext) -> None:
        cx = ctx.center_x
        cy = ctx.center_y + ctx.float_offset_y
        breathe = self._cube_breathe(ctx)
        fit_radius = ctx.body_radius * _CUBE_BODY_SCALE * breathe
        opacity = ctx.opacity * ctx.persona_blend
        if opacity <= 0.001:
            return

        self._paint_holo_aura(painter, cx, cy, fit_radius, ctx, opacity)
        paint_ripples(painter, ctx)
        if ctx.activity == AssistantActivity.CAPTURING:
            paint_waveform_ring(painter, ctx)
        self._paint_particles(painter, cx, cy, fit_radius, ctx, opacity)

        spin = 0.0 if ctx.reduced_motion else ctx.rotation
        stroke = QColor(ctx.primary if ctx.activity != AssistantActivity.IDLE_LISTEN else BRAND_STROKE_COLOR)
        stroke.setAlphaF(opacity)
        stroke_width = max(2.8, fit_radius * _COMPANION_STROKE_WIDTH_RATIO)
        paint_qube_wireframe_cube(
            painter,
            center_x=cx,
            center_y=cy,
            fit_radius=fit_radius,
            spin_angle=spin,
            stroke_color=stroke,
            stroke_width=stroke_width,
        )

        if ctx.activity == AssistantActivity.CAPTURING and ctx.input_level > 0.05:
            self._paint_edge_glow(
                painter,
                cx,
                cy,
                fit_radius,
                spin,
                QColor(ctx.primary),
                opacity * 0.35 * ctx.input_level,
                stroke_width=stroke_width + 1.2,
            )
        if ctx.activity == AssistantActivity.NEEDS_ATTENTION:
            self._paint_attention_ring(painter, cx, cy, fit_radius, spin, ctx, opacity, stroke_width)
        if ctx.activity == AssistantActivity.SPEAKING:
            paint_waveform_ring(
                painter,
                ctx,
                inner_radius=fit_radius * _WAVEFORM_OUTER_INSET,
            )

    def _cube_breathe(self, ctx: CompanionPaintContext) -> float:
        """During TTS the cube keeps idle-style breathing; only the waveform reacts."""
        if ctx.activity == AssistantActivity.SPEAKING:
            if ctx.reduced_motion:
                return 1.0
            return 1.0 + 0.065 * math.sin(ctx.anim_time * 1.8)
        return ctx.breathe

    def _paint_holo_aura(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        fit_radius: float,
        ctx: CompanionPaintContext,
        opacity: float,
    ) -> None:
        grad = QRadialGradient(QPointF(cx, cy), fit_radius * 1.35)
        core = QColor(ctx.secondary)
        core.setAlphaF(0.16 * opacity)
        mid = QColor(ctx.primary)
        mid.setAlphaF(0.08 * opacity)
        outer = QColor(ctx.primary)
        outer.setAlphaF(0.0)
        grad.setColorAt(0.0, core)
        grad.setColorAt(0.55, mid)
        grad.setColorAt(1.0, outer)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), fit_radius * 1.25, fit_radius * 1.25)

    def _paint_particles(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        fit_radius: float,
        ctx: CompanionPaintContext,
        opacity: float,
    ) -> None:
        if ctx.reduced_motion:
            return
        rotation = ctx.rotation
        for i in range(_PARTICLE_COUNT):
            angle = rotation * 1.2 + i * (2 * math.pi / _PARTICLE_COUNT)
            dist = fit_radius * (1.18 + 0.1 * math.sin(ctx.anim_time * 1.8 + i))
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist * 0.85 + math.sin(ctx.anim_time * 2 + i) * 2
            particle = QColor(ctx.secondary)
            particle.setAlphaF(0.35 * opacity)
            painter.setBrush(particle)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px, py), 2.0, 2.0)

    def _paint_edge_glow(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        fit_radius: float,
        spin: float,
        color: QColor,
        alpha: float,
        *,
        stroke_width: float,
    ) -> None:
        glow = QColor(color)
        glow.setAlphaF(alpha)
        segments = build_qube_wireframe_segments(
            center_x=cx,
            center_y=cy,
            fit_radius=fit_radius,
            spin_angle=spin,
            stroke_width=stroke_width,
        )
        painter.setPen(QPen(glow, 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for start, end in segments:
            painter.drawLine(start, end)

    def _paint_attention_ring(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        fit_radius: float,
        spin: float,
        ctx: CompanionPaintContext,
        opacity: float,
        stroke_width: float,
    ) -> None:
        pulse = 0.5 + 0.5 * math.sin(ctx.anim_time * 4.5)
        ring = QColor(ctx.primary)
        ring.setAlphaF((0.35 + pulse * 0.35) * opacity)
        segments = build_qube_wireframe_segments(
            center_x=cx,
            center_y=cy,
            fit_radius=fit_radius,
            spin_angle=spin,
            stroke_width=stroke_width,
        )
        xs = [pt.x() for seg in segments for pt in seg]
        ys = [pt.y() for seg in segments for pt in seg]
        if not xs:
            return
        painter.setPen(QPen(ring, 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cx_ring = (min(xs) + max(xs)) / 2
        cy_ring = (min(ys) + max(ys)) / 2
        rx = (max(xs) - min(xs)) / 2 * (1.08 + pulse * 0.06)
        ry = (max(ys) - min(ys)) / 2 * (1.08 + pulse * 0.06)
        painter.drawEllipse(QPointF(cx_ring, cy_ring), rx, ry)
