"""Tests for core/platform/window_activation.py."""

from __future__ import annotations

from core.platform import window_activation as mod


def test_splash_window_flags_include_splash_screen_and_topmost(qapp_cls) -> None:
    from PyQt6.QtCore import Qt

    flags = mod.splash_window_flags()
    assert flags & Qt.WindowType.SplashScreen
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint


def test_screen_for_new_window_falls_back_to_primary(qapp_cls) -> None:
    app = qapp_cls.instance() or qapp_cls([])
    primary = app.primaryScreen()
    assert mod.screen_for_new_window() is primary


def test_early_splash_present_activates_window(qapp_cls, monkeypatch) -> None:
    from ui import early_splash as early_mod
    from ui.early_splash import EarlySplashController

    calls: list[object] = []
    monkeypatch.setattr(
        early_mod,
        "activate_toplevel_window",
        lambda widget: calls.append(widget),
    )

    app = qapp_cls.instance() or qapp_cls([])
    splash = EarlySplashController()
    splash.present()
    for _ in range(5):
        app.processEvents()
    splash.dismiss()
    for _ in range(5):
        app.processEvents()

    assert calls
    assert calls[0] is splash._shell
