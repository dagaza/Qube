"""Tests for Qube window/taskbar icon resolution."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ui.app_icon import (
    apply_linux_desktop_integration,
    qube_window_icon,
    resolve_qube_window_icon_path,
)


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


def test_apply_linux_desktop_integration_sets_app_name(_qube_app):
    app = QApplication.instance()
    assert app is not None
    apply_linux_desktop_integration(app)
    if sys.platform == "linux":
        assert app.applicationName() == "Qube"
        assert app.applicationDisplayName() == "Qube"
