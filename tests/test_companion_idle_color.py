"""Tests for companion idle color presets and settings."""

from __future__ import annotations

from core.assistant_activity import AssistantActivity
from core.companion_idle_color import (
    CompanionIdleColor,
    DEFAULT_COMPANION_IDLE_COLOR,
    idle_color_pair,
    normalize_companion_idle_color,
)
from core.theme.accessors import theme_for
from ui.companion.companion_theme import activity_color_pair, companion_idle_color_pair


def test_normalize_companion_idle_color_defaults_unknown():
    assert normalize_companion_idle_color(None) == DEFAULT_COMPANION_IDLE_COLOR
    assert normalize_companion_idle_color("") == DEFAULT_COMPANION_IDLE_COLOR
    assert normalize_companion_idle_color("invalid") == DEFAULT_COMPANION_IDLE_COLOR


def test_normalize_companion_idle_color_accepts_values():
    assert normalize_companion_idle_color("purple") == CompanionIdleColor.PURPLE
    assert normalize_companion_idle_color("BLUE") == CompanionIdleColor.BLUE
    assert normalize_companion_idle_color(CompanionIdleColor.PURPLE) == CompanionIdleColor.PURPLE


def test_idle_color_pair_presets():
    assert idle_color_pair(CompanionIdleColor.PURPLE) == ("#8b5cf6", "#a78bfa")
    assert idle_color_pair(CompanionIdleColor.BLUE) == ("#89b4fa", "#b4befe")


def test_companion_idle_color_pair_uses_theme_tokens():
    theme = theme_for(is_dark=True)
    purple = companion_idle_color_pair(CompanionIdleColor.PURPLE, theme)
    blue = companion_idle_color_pair(CompanionIdleColor.BLUE, theme)
    assert purple == (theme.accent, theme.accent_hover)
    assert blue[0] == theme.link


def test_activity_color_pair_idle_respects_preset():
    theme = theme_for(is_dark=True)
    purple = activity_color_pair(
        AssistantActivity.IDLE_LISTEN,
        CompanionIdleColor.PURPLE,
        is_dark=True,
    )
    blue = activity_color_pair(
        AssistantActivity.IDLE_LISTEN,
        CompanionIdleColor.BLUE,
        is_dark=True,
    )
    assert purple == companion_idle_color_pair(CompanionIdleColor.PURPLE, theme)
    assert blue == companion_idle_color_pair(CompanionIdleColor.BLUE, theme)


def test_activity_color_pair_non_idle_uses_semantic_tokens():
    theme = theme_for(is_dark=True)
    working = activity_color_pair(
        AssistantActivity.WORKING,
        CompanionIdleColor.PURPLE,
        is_dark=True,
    )
    assert working == (theme.info, theme.link)


def test_companion_idle_color_settings_round_trip(monkeypatch):
    store: dict[str, object] = {}

    class FakeStore:
        def get(self, key, default=None):
            return store.get(key, default)

        def set(self, key, value):
            store[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: FakeStore())

    from core import app_settings

    assert app_settings.get_companion_idle_color() == CompanionIdleColor.PURPLE
    app_settings.set_companion_idle_color("blue")
    assert app_settings.get_companion_idle_color() == CompanionIdleColor.BLUE
