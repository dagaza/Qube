"""Tests for companion screen placement helpers."""

from __future__ import annotations

from PyQt6.QtCore import QRect

from core.companion_placement import (
    COMPASS_SNAP_ZONES,
    CompanionSnapZone,
    _SNAP_LABEL_INSET_PX,
    _SNAP_LABEL_REF_HEIGHT_PX,
    _SNAP_LABEL_REF_WIDTHS,
    compute_snap_position,
    default_companion_position,
    nearest_snap_zone,
    normalize_companion_snap_zone,
    position_from_saved,
    snap_zone_label_box,
    snap_zone_label_center,
)


def test_normalize_companion_snap_zone_defaults_unknown():
    assert normalize_companion_snap_zone(None) == CompanionSnapZone.NONE
    assert normalize_companion_snap_zone("invalid") == CompanionSnapZone.NONE
    assert normalize_companion_snap_zone("ne") == CompanionSnapZone.NE


def test_default_companion_position_is_bottom_right():
    geo = QRect(100, 50, 1000, 800)
    x, y = default_companion_position(geo, width=120, height=120)
    assert x == geo.right() - 120 - 24
    assert y == geo.bottom() - 120 - 24


def test_compute_snap_positions_cover_compass():
    geo = QRect(0, 0, 1000, 800)
    width, height = 100, 100
    margin = 24
    right = geo.right() - width - margin
    bottom = geo.bottom() - height - margin
    mid_x = geo.left() + (geo.width() - width) // 2
    mid_y = geo.top() + (geo.height() - height) // 2
    cases = {
        CompanionSnapZone.N: (mid_x, margin),
        CompanionSnapZone.S: (mid_x, bottom),
        CompanionSnapZone.E: (right, mid_y),
        CompanionSnapZone.W: (margin, mid_y),
        CompanionSnapZone.NE: (right, margin),
        CompanionSnapZone.NW: (margin, margin),
        CompanionSnapZone.SE: (right, bottom),
        CompanionSnapZone.SW: (margin, bottom),
        CompanionSnapZone.CENTER: (mid_x, mid_y),
    }
    for zone, expected in cases.items():
        assert compute_snap_position(zone, geo, width=width, height=height) == expected


def test_position_from_saved_prefers_last_coordinates():
    geo = QRect(0, 0, 1000, 800)
    pos = {
        "x": 200,
        "y": 300,
        "snap_zone": "se",
        "dock_edge": "none",
    }
    x, y, zone, fallback = position_from_saved(pos, geo, width=100, height=100)
    assert (x, y) == (200, 300)
    assert zone == CompanionSnapZone.NONE
    assert fallback is False


def test_position_from_saved_uses_snap_when_coordinates_missing():
    geo = QRect(0, 0, 1000, 800)
    pos = {"snap_zone": "n", "dock_edge": "none"}
    x, y, zone, fallback = position_from_saved(pos, geo, width=100, height=100)
    assert (x, y) == compute_snap_position(CompanionSnapZone.N, geo, width=100, height=100)
    assert zone == CompanionSnapZone.N
    assert fallback is False


def test_position_from_saved_falls_back_to_bottom_right():
    geo = QRect(0, 0, 1000, 800)
    x, y, zone, fallback = position_from_saved({}, geo, width=100, height=100)
    assert (x, y) == default_companion_position(geo, width=100, height=100)
    assert zone == CompanionSnapZone.SE
    assert fallback is True


def test_snap_zone_label_boxes_use_uniform_edge_inset():
    geo = QRect(0, 0, 1000, 800)
    margin = _SNAP_LABEL_INSET_PX
    top_zones = {CompanionSnapZone.N, CompanionSnapZone.NE, CompanionSnapZone.NW}
    bottom_zones = {CompanionSnapZone.S, CompanionSnapZone.SE, CompanionSnapZone.SW}
    left_zones = {CompanionSnapZone.W, CompanionSnapZone.NW, CompanionSnapZone.SW}
    right_zones = {CompanionSnapZone.E, CompanionSnapZone.NE, CompanionSnapZone.SE}

    for zone in COMPASS_SNAP_ZONES:
        tw = _SNAP_LABEL_REF_WIDTHS[zone]
        th = _SNAP_LABEL_REF_HEIGHT_PX
        x, y, w, h = snap_zone_label_box(
            zone, geo, text_width=tw, text_height=th, margin=margin, local=True
        )
        if zone in top_zones:
            assert y == margin
        if zone in bottom_zones:
            assert y + h == geo.height() - margin
        if zone in left_zones:
            assert x == margin
        if zone in right_zones:
            assert x + w == geo.width() - margin


def test_nearest_snap_zone_finds_closest_anchor():
    geo = QRect(0, 0, 1000, 800)
    cx, cy = snap_zone_label_center(CompanionSnapZone.NE, geo, local=True)
    zone = nearest_snap_zone(
        cx - 50,
        cy - 50,
        width=100,
        height=100,
        geo=geo,
    )
    assert zone == CompanionSnapZone.NE


def test_nearest_snap_zone_returns_none_when_far():
    geo = QRect(0, 0, 1000, 800)
    zone = nearest_snap_zone(400, 400, width=100, height=100, geo=geo, radius=20)
    assert zone == CompanionSnapZone.NONE
