"""Regression tests for the lightweight process entry point."""

from __future__ import annotations

from core.qube_tooltip import QubeToolTipController
from ui.branded_theme import SPLASH_SURFACE_BG, early_splash_card_qss
from ui.early_splash import EarlySplashController


def test_tooltip_controller_construction_is_reentrant_safe(qapp_cls) -> None:
    app = qapp_cls.instance() or qapp_cls([])
    ctrl = QubeToolTipController.instance()
    app.processEvents()
    assert ctrl is QubeToolTipController.instance()


def test_early_splash_present_does_not_recurse(qapp_cls) -> None:
    """Early splash must not recurse through QubeToolTipController construction."""
    app = qapp_cls.instance() or qapp_cls([])
    splash = EarlySplashController()
    splash.present()
    for _ in range(5):
        app.processEvents()
    splash.dismiss()
    for _ in range(5):
        app.processEvents()


def test_early_splash_is_static_opaque_branded_card(qapp_cls) -> None:
    """Pre-import splash must not rely on timers (GUI thread blocks on import)."""
    app = qapp_cls.instance() or qapp_cls([])
    splash = EarlySplashController()
    splash.present()
    app.processEvents()

    assert splash._shell.windowOpacity() == 1.0  # noqa: SLF001
    assert splash._status.text() == "Loading…"  # noqa: SLF001
    assert not hasattr(splash, "_spinner")
    assert not hasattr(splash, "_spinner_timer")
    qss = early_splash_card_qss()
    assert "QubeEarlySplashCard" in qss
    assert SPLASH_SURFACE_BG in qss

    splash.dismiss()
    app.processEvents()
