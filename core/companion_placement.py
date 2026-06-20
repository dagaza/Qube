"""Companion screen placement, snap zones, and restore helpers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QScreen
from PyQt6.QtWidgets import QApplication, QWidget

from core.platform.work_area import workspace_bounds_for_screen

if TYPE_CHECKING:
    pass

_DEFAULT_MARGIN_PX = 24
_EDGE_INSET_PX = 4
_SNAP_LABEL_INSET_PX = 52
_SNAP_CAPTURE_RADIUS_PX = 96
# Approximate label metrics for snap hit-testing (overlay uses real font metrics).
_SNAP_LABEL_REF_HEIGHT_PX = 48


class CompanionSnapZone(str, Enum):
    NONE = "none"
    CENTER = "center"
    N = "n"
    NE = "ne"
    E = "e"
    SE = "se"
    S = "s"
    SW = "sw"
    W = "w"
    NW = "nw"


COMPANION_SNAP_ZONE_LABELS: dict[CompanionSnapZone, str] = {
    CompanionSnapZone.CENTER: "Centre",
    CompanionSnapZone.N: "N",
    CompanionSnapZone.NE: "NE",
    CompanionSnapZone.E: "E",
    CompanionSnapZone.SE: "SE",
    CompanionSnapZone.S: "S",
    CompanionSnapZone.SW: "SW",
    CompanionSnapZone.W: "W",
    CompanionSnapZone.NW: "NW",
}

COMPASS_SNAP_ZONES: tuple[CompanionSnapZone, ...] = (
    CompanionSnapZone.N,
    CompanionSnapZone.NE,
    CompanionSnapZone.E,
    CompanionSnapZone.SE,
    CompanionSnapZone.S,
    CompanionSnapZone.SW,
    CompanionSnapZone.W,
    CompanionSnapZone.NW,
    CompanionSnapZone.CENTER,
)

_SNAP_LABEL_REF_WIDTHS: dict[CompanionSnapZone, int] = {
    CompanionSnapZone.CENTER: 88,
    CompanionSnapZone.N: 28,
    CompanionSnapZone.S: 28,
    CompanionSnapZone.E: 28,
    CompanionSnapZone.W: 28,
    CompanionSnapZone.NE: 52,
    CompanionSnapZone.NW: 52,
    CompanionSnapZone.SE: 52,
    CompanionSnapZone.SW: 52,
}


def _geo_origin(geo: QRect, *, local: bool) -> tuple[int, int]:
    return (0, 0) if local else (geo.left(), geo.top())


def snap_zone_label_box(
    zone: CompanionSnapZone,
    geo: QRect,
    *,
    text_width: int,
    text_height: int,
    margin: int = _SNAP_LABEL_INSET_PX,
    local: bool = False,
) -> tuple[int, int, int, int]:
    """Label bounding box inset uniformly from the work-area edges."""
    if zone == CompanionSnapZone.NONE:
        return (0, 0, text_width, text_height)

    ox, oy = _geo_origin(geo, local=local)
    width = geo.width()
    height = geo.height()
    cx = ox + width // 2
    cy = oy + height // 2

    if zone == CompanionSnapZone.CENTER:
        x = cx - text_width // 2
        y = cy - text_height // 2
    elif zone == CompanionSnapZone.N:
        x = cx - text_width // 2
        y = oy + margin
    elif zone == CompanionSnapZone.S:
        x = cx - text_width // 2
        y = oy + height - margin - text_height
    elif zone == CompanionSnapZone.E:
        x = ox + width - margin - text_width
        y = cy - text_height // 2
    elif zone == CompanionSnapZone.W:
        x = ox + margin
        y = cy - text_height // 2
    elif zone == CompanionSnapZone.NE:
        x = ox + width - margin - text_width
        y = oy + margin
    elif zone == CompanionSnapZone.NW:
        x = ox + margin
        y = oy + margin
    elif zone == CompanionSnapZone.SE:
        x = ox + width - margin - text_width
        y = oy + height - margin - text_height
    elif zone == CompanionSnapZone.SW:
        x = ox + margin
        y = oy + height - margin - text_height
    else:
        x = cx - text_width // 2
        y = cy - text_height // 2

    return (int(x), int(y), int(text_width), int(text_height))


def snap_zone_label_center(
    zone: CompanionSnapZone,
    geo: QRect,
    *,
    text_width: int | None = None,
    text_height: int | None = None,
    margin: int = _SNAP_LABEL_INSET_PX,
    local: bool = False,
) -> tuple[int, int]:
    tw = _SNAP_LABEL_REF_WIDTHS.get(zone, 40) if text_width is None else text_width
    th = _SNAP_LABEL_REF_HEIGHT_PX if text_height is None else text_height
    x, y, w, h = snap_zone_label_box(
        zone,
        geo,
        text_width=tw,
        text_height=th,
        margin=margin,
        local=local,
    )
    return (x + w // 2, y + h // 2)


def snap_zone_label_anchors(geo: QRect, *, local: bool = False) -> dict[CompanionSnapZone, tuple[int, int]]:
    """Centre of each label box — uniform inset from work-area edges."""
    anchors: dict[CompanionSnapZone, tuple[int, int]] = {}
    for zone in COMPASS_SNAP_ZONES:
        anchors[zone] = snap_zone_label_center(zone, geo, local=local)
    return anchors


def normalize_companion_snap_zone(value: str | CompanionSnapZone | None) -> CompanionSnapZone:
    if isinstance(value, CompanionSnapZone):
        return value
    raw = str(value or "").strip().lower()
    for zone in CompanionSnapZone:
        if zone.value == raw:
            return zone
    return CompanionSnapZone.NONE


def screen_for_widget(widget: QWidget | None) -> QScreen | None:
    """Best-effort screen for a top-level window (prefers geometry centre)."""
    if widget is not None:
        try:
            centre = widget.frameGeometry().center()
            at = QApplication.screenAt(centre)
            if at is not None:
                return at
        except RuntimeError:
            pass
        screen = widget.screen()
        if screen is not None:
            return screen
    return QApplication.primaryScreen()


def resolve_companion_screen(
    *,
    saved_screen_name: str,
    anchor_widget: QWidget | None = None,
) -> QScreen | None:
    """Resolve saved screen name, falling back to the main window's screen."""
    name = str(saved_screen_name or "").strip()
    if name:
        for screen in QApplication.screens():
            if screen.name() == name:
                return screen
    return screen_for_widget(anchor_widget)


def workspace_for_screen(screen: QScreen | None) -> QRect | None:
    if screen is None:
        return None
    return workspace_bounds_for_screen(screen)


def default_companion_position(
    geo: QRect,
    *,
    width: int,
    height: int,
    margin: int = _DEFAULT_MARGIN_PX,
) -> tuple[int, int]:
    """Bottom-right of the work area (default when position is unknown)."""
    x = geo.right() - int(width) - margin
    y = geo.bottom() - int(height) - margin
    return clamp_position(x, y, width, height, geo)


def compute_snap_position(
    zone: CompanionSnapZone,
    geo: QRect,
    *,
    width: int,
    height: int,
    margin: int = _DEFAULT_MARGIN_PX,
) -> tuple[int, int]:
    """Place companion window in a compass snap zone within ``geo``."""
    if zone in (CompanionSnapZone.NONE, CompanionSnapZone.CENTER):
        x = geo.left() + max(0, (geo.width() - int(width)) // 2)
        y = geo.top() + max(0, (geo.height() - int(height)) // 2)
        return clamp_position(x, y, width, height, geo)

    mid_x = geo.left() + max(0, (geo.width() - int(width)) // 2)
    mid_y = geo.top() + max(0, (geo.height() - int(height)) // 2)
    left = geo.left() + margin
    right = geo.right() - int(width) - margin
    top = geo.top() + margin
    bottom = geo.bottom() - int(height) - margin

    zone_xy: dict[CompanionSnapZone, tuple[int, int]] = {
        CompanionSnapZone.N: (mid_x, top),
        CompanionSnapZone.S: (mid_x, bottom),
        CompanionSnapZone.E: (right, mid_y),
        CompanionSnapZone.W: (left, mid_y),
        CompanionSnapZone.NE: (right, top),
        CompanionSnapZone.NW: (left, top),
        CompanionSnapZone.SE: (right, bottom),
        CompanionSnapZone.SW: (left, bottom),
    }
    x, y = zone_xy.get(zone, (right, bottom))
    return clamp_position(x, y, width, height, geo)


def nearest_snap_zone(
    companion_x: int,
    companion_y: int,
    *,
    width: int,
    height: int,
    geo: QRect,
    radius: int = _SNAP_CAPTURE_RADIUS_PX,
) -> CompanionSnapZone:
    """Return the compass zone nearest the companion centre within ``radius`` px."""
    orb_cx = int(companion_x) + int(width) // 2
    orb_cy = int(companion_y) + int(height) // 2
    best = CompanionSnapZone.NONE
    best_dist = float(radius) + 1.0
    for zone, (ax, ay) in snap_zone_label_anchors(geo).items():
        dx = orb_cx - ax
        dy = orb_cy - ay
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = zone
    if best_dist <= float(radius):
        return best
    return CompanionSnapZone.NONE


def clamp_position(x: int, y: int, width: int, height: int, geo: QRect) -> tuple[int, int]:
    max_x = geo.right() - int(width)
    max_y = geo.bottom() - int(height)
    return (
        max(geo.left(), min(int(x), max_x)),
        max(geo.top(), min(int(y), max_y)),
    )


def is_position_on_screen(
    x: int,
    y: int,
    *,
    width: int,
    height: int,
    geo: QRect,
    slack: int = 12,
) -> bool:
    return (
        geo.left() - slack <= x <= geo.right() - int(width) + slack
        and geo.top() - slack <= y <= geo.bottom() - int(height) + slack
    )


def normalized_position(x: int, y: int, geo: QRect) -> tuple[float, float]:
    return (
        (x - geo.left()) / max(1, geo.width()),
        (y - geo.top()) / max(1, geo.height()),
    )


def position_from_saved(
    pos: dict,
    geo: QRect,
    *,
    width: int,
    height: int,
    anchor_widget: QWidget | None = None,
) -> tuple[int, int, CompanionSnapZone, bool]:
    """Resolve companion coordinates from persisted settings.

    Returns ``(x, y, snap_zone, used_fallback)``.
    """
    x = pos.get("x")
    y = pos.get("y")
    if x is not None and y is not None:
        ix, iy = int(x), int(y)
        if is_position_on_screen(ix, iy, width=width, height=height, geo=geo):
            edge = str(pos.get("dock_edge") or "none")
            if edge == "left":
                ix = geo.left() + _EDGE_INSET_PX
            elif edge == "right":
                ix = geo.right() - int(width) - _EDGE_INSET_PX
            elif edge == "bottom":
                iy = geo.bottom() - int(height) - _EDGE_INSET_PX
            ix, iy = clamp_position(ix, iy, width, height, geo)
            return ix, iy, CompanionSnapZone.NONE, False

    norm_x = pos.get("norm_x")
    norm_y = pos.get("norm_y")
    if norm_x is not None and norm_y is not None:
        ix = int(geo.left() + float(norm_x) * geo.width())
        iy = int(geo.top() + float(norm_y) * geo.height())
        ix, iy = clamp_position(ix, iy, width, height, geo)
        if is_position_on_screen(ix, iy, width=width, height=height, geo=geo):
            return ix, iy, CompanionSnapZone.NONE, False

    snap_zone = normalize_companion_snap_zone(pos.get("snap_zone"))
    if snap_zone != CompanionSnapZone.NONE:
        x, y = compute_snap_position(snap_zone, geo, width=width, height=height)
        return x, y, snap_zone, False

    x, y = default_companion_position(geo, width=width, height=height)
    return x, y, CompanionSnapZone.SE, True
