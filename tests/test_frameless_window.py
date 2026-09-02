"""Tests for frameless translucent window helpers."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from core.platform.frameless_window import (
    apply_frameless_dialog_chrome,
    configure_frameless_dialog_host,
    frameless_dialog_window_flags,
)


def test_frameless_dialog_window_flags_include_no_drop_shadow():
    flags = frameless_dialog_window_flags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.Dialog
    assert flags & Qt.WindowType.NoDropShadowWindowHint


def test_configure_frameless_dialog_host_sets_translucent_attrs(qapp):
    widget = QWidget()
    configure_frameless_dialog_host(widget)
    assert widget.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert not widget.autoFillBackground()


def test_apply_frameless_dialog_chrome_does_not_raise(qapp):
    widget = QWidget()
    configure_frameless_dialog_host(widget)
    apply_frameless_dialog_chrome(widget)
