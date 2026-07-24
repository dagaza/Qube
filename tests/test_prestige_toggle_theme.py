"""Tests for PrestigeToggle theme-aware track colors."""

from __future__ import annotations

from core.theme.accessors import theme_for
from ui.components.toggle import PrestigeToggle


def test_apply_theme_dark_track(_qube_app):
    toggle = PrestigeToggle(is_dark=True)
    theme = theme_for(is_dark=True)
    assert toggle._bg_color.name() == theme.qcolor(theme.surface_pressed).name()


def test_apply_theme_light_track(_qube_app):
    toggle = PrestigeToggle(is_dark=False)
    theme = theme_for(is_dark=False)
    assert toggle._bg_color.name() == theme.qcolor(theme.border).name()


def test_apply_theme_switches_with_mode(_qube_app):
    toggle = PrestigeToggle(is_dark=True)
    toggle.apply_theme(is_dark=False)
    assert toggle._bg_color.name() == theme_for(is_dark=False).qcolor(
        theme_for(is_dark=False).border
    ).name()
    toggle.apply_theme(is_dark=True)
    assert toggle._bg_color.name() == theme_for(is_dark=True).qcolor(
        theme_for(is_dark=True).surface_pressed
    ).name()
