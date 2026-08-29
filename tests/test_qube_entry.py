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
    from ui.branded_theme import SPLASH_DETAIL_COLOR, SPLASH_TITLE_COLOR

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
    assert "rgba(" not in qss
    assert SPLASH_TITLE_COLOR in splash._status.styleSheet() or SPLASH_DETAIL_COLOR in splash._status.styleSheet()  # noqa: SLF001

    splash.dismiss()
    app.processEvents()


def test_qss_color_produces_hex_for_cross_platform_splash_qss() -> None:
    from ui.branded_theme import qss_color, splash_split_card_qss

    assert qss_color(148, 163, 184, 1.0) == "#94a3b8"
    assert qss_color(148, 163, 184, 0.9).startswith("#94a3b8")
    assert len(qss_color(148, 163, 184, 0.9)) == 9
    split_qss = splash_split_card_qss()
    assert "rgba(" not in split_qss
    assert "#f8fafc" in split_qss
