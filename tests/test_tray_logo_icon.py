"""Tests for Qube tray logo icon resolution."""

from __future__ import annotations

from ui.tray_controller import build_tray_logo_icon, resolve_qube_logo_path


def test_resolve_qube_logo_path_finds_repo_asset():
    path = resolve_qube_logo_path()
    assert path is not None
    assert path.name == "qube_logo_256.png"
    assert path.is_file()


def test_build_tray_logo_icon_from_repo_asset():
    icon = build_tray_logo_icon()
    assert not icon.isNull()
    assert icon.availableSizes()
