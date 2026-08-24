"""Regression tests for the lightweight process entry point."""

from __future__ import annotations

from core.qube_tooltip import QubeToolTipController
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
