"""Shared Qube logo wireframe cube (splash + desktop companion)."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath

BRAND_STROKE_COLOR = QColor("#8b5cf6")
DEFAULT_STROKE_WIDTH = 7.6

_CUBE_VERTICES: tuple[tuple[float, float, float], ...] = (
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
)

_CUBE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
)

_LOGO_REST_TILT_X = math.atan(1.0 / math.sqrt(2.0))
_LOGO_REST_TILT_Y = math.pi / 4.0
_ROTATION_AXIS = (3.0, 1.0, 1.0)
_CANVAS_INSET_RATIO = 0.06
_CUBE_SCALE_RATIO = 1.0
_Q_TAIL_3D_LENGTH = 0.82
_SPIN_EXTENT_SAMPLES = 120


def _rotate_xyz(
    x: float,
    y: float,
    z: float,
    ax: float,
    ay: float,
    az: float,
) -> tuple[float, float, float]:
    cosy, siny = math.cos(ay), math.sin(ay)
    x, z = x * cosy + z * siny, -x * siny + z * cosy
    cosx, sinx = math.cos(ax), math.sin(ax)
    y, z = y * cosx - z * sinx, y * sinx + z * cosx
    cosz, sinz = math.cos(az), math.sin(az)
    x, y = x * cosz - y * sinz, x * sinz + y * cosz
    return x, y, z


def _rotate_axis(
    x: float,
    y: float,
    z: float,
    ax: float,
    ay: float,
    az: float,
    angle_rad: float,
) -> tuple[float, float, float]:
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length <= 0.0:
        return x, y, z
    ux, uy, uz = ax / length, ay / length, az / length
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dot = ux * x + uy * y + uz * z
    cross_x = uy * z - uz * y
    cross_y = uz * x - ux * z
    cross_z = ux * y - uy * x
    return (
        x * cos_a + cross_x * sin_a + ux * dot * (1.0 - cos_a),
        y * cos_a + cross_y * sin_a + uy * dot * (1.0 - cos_a),
        z * cos_a + cross_z * sin_a + uz * dot * (1.0 - cos_a),
    )


def _project_rest_point(x: float, y: float, z: float, *, scale: float = 1.0) -> QPointF:
    rx, ry, rz = _rotate_xyz(x, y, z, _LOGO_REST_TILT_X, _LOGO_REST_TILT_Y, 0.0)
    _ = rz
    return QPointF(rx * scale, -ry * scale)


def _resolve_q_tail_attachment() -> tuple[tuple[int, int], tuple[float, float, float]]:
    """Lock the Q tail to the logo's south-eastern edge in object space."""
    best_edge = (0, 1)
    best_mid_score = float("-inf")
    for i, j in _CUBE_EDGES:
        v1 = _CUBE_VERTICES[i]
        v2 = _CUBE_VERTICES[j]
        mid = (
            (v1[0] + v2[0]) * 0.5,
            (v1[1] + v2[1]) * 0.5,
            (v1[2] + v2[2]) * 0.5,
        )
        mid_screen = _project_rest_point(*mid)
        score = mid_screen.x() + mid_screen.y()
        if score > best_mid_score:
            best_mid_score = score
            best_edge = (i, j)

    i, j = best_edge
    v1 = _CUBE_VERTICES[i]
    v2 = _CUBE_VERTICES[j]
    edge_vec = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    edge_len = math.hypot(*edge_vec)
    edge_unit = tuple(component / edge_len for component in edge_vec)
    mid = (
        (v1[0] + v2[0]) * 0.5,
        (v1[1] + v2[1]) * 0.5,
        (v1[2] + v2[2]) * 0.5,
    )
    mid_screen = _project_rest_point(*mid)

    best_outward = (0.0, 0.0, 1.0)
    best_out_score = float("-inf")
    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        cross = (
            edge_unit[1] * axis[2] - edge_unit[2] * axis[1],
            edge_unit[2] * axis[0] - edge_unit[0] * axis[2],
            edge_unit[0] * axis[1] - edge_unit[1] * axis[0],
        )
        cross_len = math.hypot(*cross)
        if cross_len <= 1e-6:
            continue
        outward = tuple(component / cross_len for component in cross)
        tip_screen = _project_rest_point(
            mid[0] + outward[0] * _Q_TAIL_3D_LENGTH,
            mid[1] + outward[1] * _Q_TAIL_3D_LENGTH,
            mid[2] + outward[2] * _Q_TAIL_3D_LENGTH,
        )
        score = (tip_screen.x() - mid_screen.x()) + (tip_screen.y() - mid_screen.y())
        if score > best_out_score:
            best_out_score = score
            best_outward = outward

    return best_edge, best_outward


_Q_TAIL_EDGE, _Q_TAIL_OUTWARD_3D = _resolve_q_tail_attachment()


def _transform_point(
    x: float,
    y: float,
    z: float,
    *,
    spin_angle: float,
) -> tuple[float, float, float]:
    rx, ry, rz = _rotate_xyz(x, y, z, _LOGO_REST_TILT_X, _LOGO_REST_TILT_Y, 0.0)
    if spin_angle != 0.0:
        ax, ay, az = _ROTATION_AXIS
        rx, ry, rz = _rotate_axis(rx, ry, rz, ax, ay, az, spin_angle)
    return rx, ry, rz


def _q_tail_tip_at_angle(
    rotated: list[tuple[float, float, float]],
    *,
    spin_angle: float,
) -> tuple[float, float, float]:
    i, j = _Q_TAIL_EDGE
    v1 = rotated[i]
    v2 = rotated[j]
    mid = (
        (v1[0] + v2[0]) * 0.5,
        (v1[1] + v2[1]) * 0.5,
        (v1[2] + v2[2]) * 0.5,
    )
    ox, oy, oz = _transform_point(*_Q_TAIL_OUTWARD_3D, spin_angle=spin_angle)
    return (
        mid[0] + ox * _Q_TAIL_3D_LENGTH,
        mid[1] + oy * _Q_TAIL_3D_LENGTH,
        mid[2] + oz * _Q_TAIL_3D_LENGTH,
    )


def _normalized_xy_points_at_angle(spin_angle: float) -> list[tuple[float, float]]:
    rotated = [
        _transform_point(x, y, z, spin_angle=spin_angle)
        for x, y, z in _CUBE_VERTICES
    ]
    points = [(x, y) for x, y, _z in rotated]
    tip = _q_tail_tip_at_angle(rotated, spin_angle=spin_angle)
    points.append((tip[0], tip[1]))
    return points


def _compute_max_spin_extent_norm(*, samples: int = _SPIN_EXTENT_SAMPLES) -> float:
    max_extent = 0.0
    for i in range(samples):
        angle = (2.0 * math.pi * i) / samples
        for x, y in _normalized_xy_points_at_angle(angle):
            max_extent = max(max_extent, abs(x), abs(y))
    return max_extent


_MAX_SPIN_EXTENT_NORM = _compute_max_spin_extent_norm()


def _to_screen(
    x: float,
    y: float,
    z: float,
    *,
    center_x: float,
    center_y: float,
    scale: float,
) -> QPointF:
    _ = z
    return QPointF(center_x + x * scale, center_y - y * scale)


def _filled_capsule_path(start: QPointF, end: QPointF, radius: float) -> QPainterPath:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    length = math.hypot(dx, dy)
    if length < 1e-6:
        path = QPainterPath()
        path.addEllipse(start, radius, radius)
        return path
    ux, uy = dx / length, dy / length
    px, py = -uy * radius, ux * radius

    path = QPainterPath()
    path.addEllipse(start, radius, radius)
    path.addEllipse(end, radius, radius)

    body = QPainterPath()
    body.moveTo(QPointF(start.x() + px, start.y() + py))
    body.lineTo(QPointF(end.x() + px, end.y() + py))
    body.lineTo(QPointF(end.x() - px, end.y() - py))
    body.lineTo(QPointF(start.x() - px, start.y() - py))
    body.closeSubpath()
    return path.united(body)


def _paint_stroke_union(
    painter: QPainter,
    segments: list[tuple[QPointF, QPointF]],
    *,
    stroke_color: QColor,
    joint_radius: float,
) -> None:
    if not segments:
        return
    merged = QPainterPath()
    for start, end in segments:
        merged = merged.united(_filled_capsule_path(start, end, joint_radius))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(stroke_color)
    painter.drawPath(merged)


def _scale_for_fit_radius(fit_radius: float, *, joint_radius: float) -> float:
    stroke_pad = joint_radius + 0.5
    available_radius = max(fit_radius - stroke_pad, fit_radius * 0.2)
    return (available_radius / _MAX_SPIN_EXTENT_NORM) * _CUBE_SCALE_RATIO


def qube_wireframe_visual_extent(fit_radius: float, *, stroke_width: float | None = None) -> float:
    """Max distance from center to furthest painted pixel for layout sizing."""
    width = DEFAULT_STROKE_WIDTH if stroke_width is None else stroke_width
    joint_radius = width * 0.62
    return fit_radius + joint_radius + 2.0


def build_qube_wireframe_segments(
    *,
    center_x: float,
    center_y: float,
    fit_radius: float,
    spin_angle: float = 0.0,
    stroke_width: float | None = None,
) -> list[tuple[QPointF, QPointF]]:
    """Return screen-space edge segments for the wireframe cube + Q tail."""
    width = DEFAULT_STROKE_WIDTH if stroke_width is None else stroke_width
    joint_radius = width * 0.62
    scale = _scale_for_fit_radius(fit_radius, joint_radius=joint_radius)
    rotated = [
        _transform_point(x, y, z, spin_angle=spin_angle)
        for x, y, z in _CUBE_VERTICES
    ]
    screen = [
        _to_screen(x, y, z, center_x=center_x, center_y=center_y, scale=scale)
        for x, y, z in rotated
    ]
    segments = [(screen[i], screen[j]) for i, j in _CUBE_EDGES]
    i, j = _Q_TAIL_EDGE
    v1 = rotated[i]
    v2 = rotated[j]
    mid = (
        (v1[0] + v2[0]) * 0.5,
        (v1[1] + v2[1]) * 0.5,
        (v1[2] + v2[2]) * 0.5,
    )
    tip = _q_tail_tip_at_angle(rotated, spin_angle=spin_angle)
    anchor = _to_screen(*mid, center_x=center_x, center_y=center_y, scale=scale)
    tip_pt = _to_screen(*tip, center_x=center_x, center_y=center_y, scale=scale)
    segments.append((anchor, tip_pt))
    return segments


def paint_qube_wireframe_cube(
    painter: QPainter,
    *,
    center_x: float,
    center_y: float,
    fit_radius: float,
    spin_angle: float = 0.0,
    stroke_color: QColor | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw the isometric Qube wireframe cube with fixed SE Q tail."""
    color = BRAND_STROKE_COLOR if stroke_color is None else stroke_color
    width = DEFAULT_STROKE_WIDTH if stroke_width is None else stroke_width
    joint_radius = width * 0.62
    segments = build_qube_wireframe_segments(
        center_x=center_x,
        center_y=center_y,
        fit_radius=fit_radius,
        spin_angle=spin_angle,
        stroke_width=width,
    )
    _paint_stroke_union(
        painter,
        segments,
        stroke_color=color,
        joint_radius=joint_radius,
    )


def fit_radius_for_widget_side(side: float, *, inset_ratio: float = _CANVAS_INSET_RATIO) -> float:
    """Match splash cube sizing: largest fit radius inside a square widget."""
    inset = side * inset_ratio
    drawable = max(side - 2.0 * inset, side * 0.5)
    return max(drawable / 2.0, drawable * 0.2)
