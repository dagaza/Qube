"""Application UI language variant (British vs American English spelling)."""

from __future__ import annotations

import re
from enum import Enum


class UiLanguage(str, Enum):
    BRITISH = "british"
    AMERICAN = "american"


UI_LANGUAGE_LABELS: dict[UiLanguage, str] = {
    UiLanguage.BRITISH: "British English",
    UiLanguage.AMERICAN: "American English",
}

UI_LANGUAGE_DESCRIPTIONS: dict[UiLanguage, str] = {
    UiLanguage.BRITISH: (
        "Standard Qube spelling — colour, behaviour, minimise, and similar forms."
    ),
    UiLanguage.AMERICAN: (
        "American spelling — color, behavior, minimize, and similar forms."
    ),
}

DEFAULT_UI_LANGUAGE = UiLanguage.BRITISH

# Canonical UI copy is British English; apply these pairs when American is selected.
_BRITISH_AMERICAN_SPELLINGS: tuple[tuple[str, str], ...] = (
    ("behaviour", "behavior"),
    ("coloured", "colored"),
    ("colour", "color"),
    ("customise", "customize"),
    ("favourite", "favorite"),
    ("grey", "gray"),
    ("initialise", "initialize"),
    ("labelled", "labeled"),
    ("maximise", "maximize"),
    ("minimise", "minimize"),
    ("modelling", "modeling"),
    ("optimised", "optimized"),
    ("organise", "organize"),
    ("recognise", "recognize"),
    ("visualisation", "visualization"),
)


def normalize_ui_language(value: str | UiLanguage | None) -> UiLanguage:
    if isinstance(value, UiLanguage):
        return value
    raw = str(value or "").strip().lower()
    for language in UiLanguage:
        if language.value == raw:
            return language
    return DEFAULT_UI_LANGUAGE


def _replace_preserve_case(match: re.Match[str], american: str) -> str:
    word = match.group(0)
    if word.isupper():
        return american.upper()
    if word[0].isupper():
        return american.capitalize()
    return american


def localize_text(text: str, language: UiLanguage | None = None) -> str:
    """Return UI text localised for the selected language variant."""
    if not text:
        return text
    if language is None:
        from core import app_settings

        language = app_settings.get_ui_language()
    if language == UiLanguage.BRITISH:
        return text
    result = text
    for british, american in _BRITISH_AMERICAN_SPELLINGS:
        result = re.sub(
            rf"\b{re.escape(british)}\b",
            lambda m, a=american: _replace_preserve_case(m, a),
            result,
            flags=re.IGNORECASE,
        )
    return result


def tr(text: str, language: UiLanguage | None = None) -> str:
    """Short alias for :func:`localize_text`."""
    return localize_text(text, language=language)
