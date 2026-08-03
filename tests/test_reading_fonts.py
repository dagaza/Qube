"""Tests for curated reading fonts and app settings persistence."""

from __future__ import annotations

import pytest

from core import app_settings
from core.reading_fonts import (
    DEFAULT_READING_FONT_ID,
    READING_FONT_IBM_PLEX_SANS,
    READING_FONT_INTER,
    READING_FONT_LITERATA,
    READING_FONT_SOURCE_SANS_3,
    make_system_reading_font_id,
    normalize_reading_font_id,
    parse_system_reading_font_family,
    reading_font_display_label,
    reading_font_label,
    reading_font_qt_family,
    reset_reading_font_cache_for_tests,
    system_reading_font_families,
)
from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import (
    AGENT_MESSAGE_SHELL,
    TRANSPARENT_TEXT_PREVIEW,
    USER_BUBBLE_LABEL,
    theme_style,
)


@pytest.fixture(autouse=True)
def _reset_reading_font_loader():
    reset_reading_font_cache_for_tests()
    yield
    reset_reading_font_cache_for_tests()


def test_normalize_reading_font_id_defaults_unknown():
    assert normalize_reading_font_id(None) == DEFAULT_READING_FONT_ID
    assert normalize_reading_font_id("unknown-font") == DEFAULT_READING_FONT_ID
    assert normalize_reading_font_id(READING_FONT_LITERATA) == READING_FONT_LITERATA


def test_reading_font_label_for_each_choice():
    assert reading_font_label(READING_FONT_INTER) == "Inter"
    assert reading_font_label(READING_FONT_SOURCE_SANS_3) == "Source Sans 3"
    assert reading_font_label(READING_FONT_IBM_PLEX_SANS) == "IBM Plex Sans"
    assert reading_font_label(READING_FONT_LITERATA) == "Literata"


def test_system_reading_font_id_round_trip(monkeypatch):
    monkeypatch.setattr(
        "core.reading_fonts.QFontDatabase.families",
        lambda: ["Inter", "DejaVu Sans", "Noto Emoji"],
    )
    reset_reading_font_cache_for_tests()

    font_id = make_system_reading_font_id("dejavu sans")
    assert font_id == "system:DejaVu Sans"
    assert normalize_reading_font_id(font_id) == font_id
    assert parse_system_reading_font_family(font_id) == "DejaVu Sans"
    assert reading_font_display_label(font_id) == "DejaVu Sans (system)"
    assert reading_font_qt_family(font_id) == "DejaVu Sans"


def test_normalize_unknown_system_font_falls_back(monkeypatch):
    monkeypatch.setattr(
        "core.reading_fonts.QFontDatabase.families",
        lambda: ["Arial", "Courier New"],
    )
    assert normalize_reading_font_id("system:Not Installed Here") == DEFAULT_READING_FONT_ID


def test_system_reading_font_families_filters_icon_fonts(monkeypatch):
    monkeypatch.setattr(
        "core.reading_fonts.QFontDatabase.families",
        lambda: ["Arial", "Font Awesome", "Segoe UI Emoji", "Literata"],
    )
    reset_reading_font_cache_for_tests()
    families = system_reading_font_families(refresh=True)
    assert "Arial" in families
    assert "Literata" in families
    assert "Font Awesome" not in families
    assert "Segoe UI Emoji" not in families


def test_app_settings_persists_system_font(monkeypatch):
    backing: dict[str, object] = {}

    class _FakeStore:
        def get(self, key, default=None):
            return backing.get(key, default)

        def set(self, key, value):
            backing[key] = value

    monkeypatch.setattr(app_settings, "_store", lambda: _FakeStore())
    monkeypatch.setattr(
        "core.reading_fonts.QFontDatabase.families",
        lambda: ["Courier New"],
    )
    reset_reading_font_cache_for_tests()

    app_settings.set_ui_reading_font("system:courier new")
    assert app_settings.get_ui_reading_font() == "system:Courier New"


def test_reading_font_qt_family_falls_back_to_inter():
    family = reading_font_qt_family(READING_FONT_LITERATA)
    assert isinstance(family, str)
    assert family


def test_app_settings_reading_font_round_trip(monkeypatch):
    backing: dict[str, object] = {}

    class _FakeStore:
        def get(self, key, default=None):
            return backing.get(key, default)

        def set(self, key, value):
            backing[key] = value

    monkeypatch.setattr(app_settings, "_store", lambda: _FakeStore())

    assert app_settings.get_ui_reading_font() == DEFAULT_READING_FONT_ID
    app_settings.set_ui_reading_font(READING_FONT_SOURCE_SANS_3)
    assert app_settings.get_ui_reading_font() == READING_FONT_SOURCE_SANS_3
    app_settings.set_ui_reading_font("not-a-font")
    assert app_settings.get_ui_reading_font() == DEFAULT_READING_FONT_ID


def test_widget_styles_include_font_family_when_requested():
    from core.theme.manager import ThemeResolver
    from core.theme.schemes import BUILTIN_SCHEMES, DEFAULT_SCHEME_ID_DARK
    from core.theme.tokens import ThemeMode

    theme = ThemeResolver(BUILTIN_SCHEMES).resolve(
        mode=ThemeMode.DARK,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
    )
    css = theme_style(
        theme,
        USER_BUBBLE_LABEL,
        font_pt=11.0,
        font_family="Literata",
    )
    assert 'font-family: "Literata";' in css
    css = theme_style(
        theme,
        AGENT_MESSAGE_SHELL,
        font_pt=11.0,
        font_family="Source Sans 3",
    )
    assert 'font-family: "Source Sans 3";' in css
    css = theme_style(
        theme,
        TRANSPARENT_TEXT_PREVIEW,
        font_pt=11.0,
        font_family="IBM Plex Sans",
    )
    assert 'font-family: "IBM Plex Sans";' in css
