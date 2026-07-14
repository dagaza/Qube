"""Holographic Qube cube companion persona."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

from core.assistant_activity import AssistantActivity
from core.companion_personas import CompanionPersonaId
from ui.companion.persona_audio import paint_ripples, paint_waveform_ring
from ui.companion.persona_context import CompanionPaintContext
from ui.companion.personas.base import CompanionPersonaRenderer

_CUBE_VERTS = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
]

_CUBE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_CUBE_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)

_PARTICLE_COUNT = 10
_ISO_X = 0.866  # cos(30°)
_ISO_Y = 0.5    # sin(30°)
_CUBE_SIZE_SCALE = 0.88  # ~12% smaller than the original 1.05× body factor
_WAVEFORM_OUTER_INSET = 1.38  # waveform ring starts outside the drawn cube silhouette


class QubeCubeClassicPersonaRenderer(CompanionPersonaRenderer):
    persona_id = CompanionPersonaId.QUBE

    def halo_extra_px(self, body_radius: float) -> int:
        return int(body_radius * 0.95)

    def visual_extent_px(self, body_radius: float) -> float:
        # Rotated cube corners (~2.5×r), float drift, waveform, and AA/pen margin.
        return body_radius * 2.65 + 10.0

    def paint(self, painter: QPainter, ctx: CompanionPaintContext) -> None:
        cx = ctx.center_x
        cy = ctx.center_y + ctx.float_offset_y
        breathe = self._cube_breathe(ctx)
        size = ctx.body_radius * 1.05 * _CUBE_SIZE_SCALE * breathe
        opacity = ctx.opacity * ctx.persona_blend
        if opacity <= 0.001:
            return

        self._paint_holo_aura(painter, cx, cy, size, ctx, opacity)
        paint_ripples(painter, ctx)
        if ctx.activity == AssistantActivity.CAPTURING:
            paint_waveform_ring(painter, ctx)
        self._paint_particles(painter, cx, cy, size, ctx, opacity)
        self._paint_layered_cube(painter, cx, cy, size, ctx, opacity, layer=0, scale=0.52, alpha=0.92)
        self._paint_layered_cube(painter, cx, cy, size, ctx, opacity, layer=1, scale=0.76, alpha=0.72)
        self._paint_layered_cube(painter, cx, cy, size, ctx, opacity, layer=2, scale=1.0, alpha=0.48)
        if ctx.activity == AssistantActivity.SPEAKING:
            paint_waveform_ring(
                painter,
                ctx,
                inner_radius=size * _WAVEFORM_OUTER_INSET,
            )

    def _cube_breathe(self, ctx: CompanionPaintContext) -> float:
        """During TTS the cube keeps idle-style breathing; only the waveform reacts."""
        if ctx.activity == AssistantActivity.SPEAKING:
            if ctx.reduced_motion:
                return 1.0
            return 1.0 + 0.065 * math.sin(ctx.anim_time * 1.8)
        return ctx.breathe

    def _project(
        self,
        x: float,
        y: float,
        z: float,
        cx: float,
        cy: float,
        scale: float,
        yaw: float,
        pitch: float,
    ) -> QPointF:
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        rx = x * cos_y - z * sin_y
        rz = x * sin_y + z * cos_y
        ry = y * cos_p - rz * sin_p
        rz2 = y * sin_p + rz * cos_p
        sx = cx + (rx - rz2) * scale * _ISO_X
        sy = cy + (rx + rz2) * scale * _ISO_Y - ry * scale * 0.82
        return QPointF(sx, sy)

    def _rotated_vertex(
        self,
        vx: float,
        vy: float,
        vz: float,
        morph: float,
        layer: int,
        t: float,
    ) -> tuple[float, float, float]:
        wobble = morph * 0.1 * math.sin(t * 2.3 + layer + vx * 3.1 + vy * 2.7)
        return vx + wobble, vy + wobble * 0.6, vz - wobble * 0.4

    def _cube_points(
        self,
        cx: float,
        cy: float,
        size: float,
        ctx: CompanionPaintContext,
        *,
        layer: int,
        scale: float,
    ) -> tuple[list[QPointF], float, float]:
        activity = ctx.activity
        morph = 0.0
        if activity == AssistantActivity.CAPTURING:
            morph = ctx.input_level

        rotation = ctx.rotation
        yaw = rotation * (1.0 + layer * 0.25) + layer * 0.4
        if activity == AssistantActivity.WORKING:
            yaw += ctx.anim_time * (0.45 + layer * 0.12)
        pitch = 0.42 + 0.14 * math.sin(ctx.anim_time * 1.6 + layer)

        points_2d: list[QPointF] = []
        for vx, vy, vz in _CUBE_VERTS:
            rx, ry, rz = self._rotated_vertex(vx, vy, vz, morph, layer, ctx.anim_time)
            points_2d.append(
                self._project(rx, ry, rz, cx, cy, size * scale, yaw, pitch)
            )
        return points_2d, yaw, pitch

    def _paint_holo_aura(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        size: float,
        ctx: CompanionPaintContext,
        opacity: float,
    ) -> None:
        grad = QRadialGradient(QPointF(cx, cy), size * 1.35)
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
        painter.drawEllipse(QPointF(cx, cy), size * 1.25, size * 1.25)

    def _paint_particles(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        size: float,
        ctx: CompanionPaintContext,
        opacity: float,
    ) -> None:
        if ctx.reduced_motion:
            return
        rotation = ctx.rotation
        for i in range(_PARTICLE_COUNT):
            angle = rotation * 1.2 + i * (2 * math.pi / _PARTICLE_COUNT)
            dist = size * (1.18 + 0.1 * math.sin(ctx.anim_time * 1.8 + i))
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist * 0.85 + math.sin(ctx.anim_time * 2 + i) * 2
            particle = QColor(ctx.secondary)
            particle.setAlphaF(0.35 * opacity)
            painter.setBrush(particle)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px, py), 2.0, 2.0)

    def _paint_layered_cube(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        size: float,
        ctx: CompanionPaintContext,
        opacity: float,
        *,
        layer: int,
        scale: float,
        alpha: float,
    ) -> None:
        activity = ctx.activity
        points_2d, yaw, pitch = self._cube_points(cx, cy, size, ctx, layer=layer, scale=scale)

        face_depths: list[tuple[float, tuple[int, ...]]] = []
        for face in _CUBE_FACES:
            avg_z = sum(_CUBE_VERTS[i][2] for i in face) / 4
            face_depths.append((avg_z, face))
        face_depths.sort(key=lambda item: item[0])

        for _depth, face in face_depths:
            path = QPainterPath()
            for idx, vi in enumerate(face):
                pt = points_2d[vi]
                if idx == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            path.closeSubpath()

            face_grad = QLinearGradient(points_2d[face[0]], points_2d[face[2]])
            top = QColor(ctx.secondary)
            top.setAlphaF(alpha * opacity * 0.62)
            bottom = QColor(ctx.primary)
            bottom.setAlphaF(alpha * opacity * 0.34)
            face_grad.setColorAt(0.0, top)
            face_grad.setColorAt(1.0, bottom)
            painter.setBrush(QBrush(face_grad))
            edge = QColor(ctx.secondary)
            edge.setAlphaF(min(1.0, alpha * opacity * 0.95))
            painter.setPen(QPen(edge, 1.6 if layer == 2 else 1.2))
            painter.drawPath(path)

        wire = QColor("#ffffff")
        wire.setAlphaF(min(1.0, alpha * opacity * (0.55 if layer == 2 else 0.35)))
        painter.setPen(QPen(wire, 1.1 if layer == 2 else 0.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i, j in _CUBE_EDGES:
            painter.drawLine(points_2d[i], points_2d[j])

        if layer == 0:
            core_center = self._project(0, 0, 0, cx, cy, size * scale * 0.35, yaw, pitch)
            core_r = size * scale * 0.16
            if activity == AssistantActivity.WORKING and not ctx.reduced_motion:
                core_grad: QConicalGradient | QRadialGradient = QConicalGradient(
                    core_center, math.degrees(ctx.anim_time * 40) % 360
                )
            else:
                core_grad = QRadialGradient(core_center, core_r)
            c1 = QColor(ctx.secondary)
            c1.setAlphaF(0.95 * opacity)
            c2 = QColor("#ffffff")
            c2.setAlphaF(0.85 * opacity)
            if isinstance(core_grad, QConicalGradient):
                core_grad.setColorAt(0.0, c1)
                core_grad.setColorAt(0.5, c2)
                core_grad.setColorAt(1.0, c1)
            else:
                core_grad.setColorAt(0.0, c2)
                core_grad.setColorAt(0.55, c1)
                core_grad.setColorAt(1.0, QColor(ctx.primary))
            painter.setBrush(QBrush(core_grad))
            painter.setPen(Qt.PenStyle.NoPen)
            pulse = 1.0 + 0.04 * math.sin(ctx.anim_time * 2)
            painter.drawRect(
                int(core_center.x() - core_r * pulse),
                int(core_center.y() - core_r * pulse),
                int(core_r * 2 * pulse),
                int(core_r * 2 * pulse),
            )

        if layer == 2 and activity == AssistantActivity.CAPTURING and ctx.input_level > 0.05:
            glow = QColor(ctx.primary)
            glow.setAlphaF(0.35 * ctx.input_level * opacity)
            painter.setPen(QPen(glow, 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i, j in _CUBE_EDGES:
                painter.drawLine(points_2d[i], points_2d[j])

        if layer == 2 and activity == AssistantActivity.NEEDS_ATTENTION:
            pulse = 0.5 + 0.5 * math.sin(ctx.anim_time * 4.5)
            ring = QColor(ctx.primary)
            ring.setAlphaF(0.35 + pulse * 0.35)
            painter.setPen(QPen(ring, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            xs = [p.x() for p in points_2d]
            ys = [p.y() for p in points_2d]
            cx_ring = (min(xs) + max(xs)) / 2
            cy_ring = (min(ys) + max(ys)) / 2
            rx = (max(xs) - min(xs)) / 2 * (1.08 + pulse * 0.06)
            ry = (max(ys) - min(ys)) / 2 * (1.08 + pulse * 0.06)
            painter.drawEllipse(QPointF(cx_ring, cy_ring), rx, ry)
