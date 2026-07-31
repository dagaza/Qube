"""Tests for PrestigeToggle theme-aware track colors."""

from __future__ import annotations

from core.theme.accessors import theme_for
from core.theme.color_utils import adjust_lightness
from core.theme.widget_styles import prestige_toggle_palette
from ui.components.toggle import PrestigeToggle


def _expected_focus_border(theme) -> str:
    return (
        theme.accent
        if theme.is_dark
        else adjust_lightness(theme.border, -0.15)
    )


def test_prestige_toggle_palette_uses_checkbox_focus_border_when_off(_qube_app):
    theme = theme_for(is_dark=True)
    palette = prestige_toggle_palette(theme)
    assert palette["track_unchecked_border"] == _expected_focus_border(theme)


def test_apply_theme_dark_unchecked_border(_qube_app):
    toggle = PrestigeToggle(is_dark=True)
    theme = theme_for(is_dark=True)
    assert (
        toggle._track_unchecked_border.name()
        == theme.qcolor(_expected_focus_border(theme)).name()
    )


def test_apply_theme_light_unchecked_border(_qube_app):
    toggle = PrestigeToggle(is_dark=False)
    theme = theme_for(is_dark=False)
    assert (
        toggle._track_unchecked_border.name()
        == theme.qcolor(_expected_focus_border(theme)).name()
    )


def test_apply_theme_switches_with_mode(_qube_app):
    toggle = PrestigeToggle(is_dark=True)
    toggle.apply_theme(is_dark=False)
    light_theme = theme_for(is_dark=False)
    assert toggle._track_unchecked_border.name() == light_theme.qcolor(
        _expected_focus_border(light_theme)
    ).name()
    toggle.apply_theme(is_dark=True)
    dark_theme = theme_for(is_dark=True)
    assert toggle._track_unchecked_border.name() == dark_theme.qcolor(
        _expected_focus_border(dark_theme)
    ).name()


def test_apply_theme_checked_fill_uses_success(_qube_app):
    toggle = PrestigeToggle(is_dark=True)
    theme = theme_for(is_dark=True)
    assert toggle._track_checked_fill.name() == theme.qcolor(theme.success).name()
