"""Tests for Qube window/taskbar icon resolution."""

from __future__ import annotations

import sys

from ui.app_icon import qube_window_icon, resolve_qube_window_icon_path


def test_resolve_qube_window_icon_path_finds_asset():
    path = resolve_qube_window_icon_path()
    assert path is not None
    assert path.is_file()
    if sys.platform == "win32":
        assert path.suffix.lower() == ".ico"
    else:
        assert path.suffix.lower() == ".png"


def test_qube_window_icon_is_valid(_qube_app):
    icon = qube_window_icon()
    assert not icon.isNull()
