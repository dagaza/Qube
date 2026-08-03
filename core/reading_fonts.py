"""Curated and system reading fonts for Conversations and Library content."""

from __future__ import annotations

import logging
from typing import NamedTuple

from PyQt6.QtGui import QFontDatabase

from core.paths import resource_path

logger = logging.getLogger("Qube.ReadingFonts")

READING_FONT_INTER = "inter"
READING_FONT_SOURCE_SANS_3 = "source_sans_3"
READING_FONT_IBM_PLEX_SANS = "ibm_plex_sans"
READING_FONT_LITERATA = "literata"

READING_FONT_SYSTEM_PREFIX = "system:"
READING_FONT_BROWSE_SYSTEM = "__browse_system_fonts__"

DEFAULT_READING_FONT_ID = READING_FONT_INTER

_FALLBACK_QT_FAMILY = "Inter"

_SYSTEM_FAMILY_SKIP_FRAGMENTS = (
    "awesome",  # Font Awesome icon families
    "icon",
    "emoji",
    "symbol",
    "dingbat",
    "webdings",
    "wingdings",
    "marlett",
)


class ReadingFontChoice(NamedTuple):
    font_id: str
    label: str
    filename: str | None


READING_FONT_CHOICES: tuple[ReadingFontChoice, ...] = (
    ReadingFontChoice(READING_FONT_INTER, "Inter", None),
    ReadingFontChoice(
        READING_FONT_SOURCE_SANS_3,
        "Source Sans 3",
        "SourceSans3-Regular.ttf",
    ),
    ReadingFontChoice(
        READING_FONT_IBM_PLEX_SANS,
        "IBM Plex Sans",
        "IBMPlexSans-Regular.ttf",
    ),
    ReadingFontChoice(READING_FONT_LITERATA, "Literata", "Literata-Regular.ttf"),
)

READING_FONT_BROWSE_SYSTEM_LABEL = "Browse system fonts…"

_CHOICE_BY_ID = {choice.font_id: choice for choice in READING_FONT_CHOICES}
_VALID_IDS = frozenset(_CHOICE_BY_ID)

_resolved_families: dict[str, str] = {}
_loaded = False
_system_families_cache: tuple[str, ...] | None = None


def is_system_reading_font_id(font_id: str | None) -> bool:
    return bool(font_id) and str(font_id).startswith(READING_FONT_SYSTEM_PREFIX)


def parse_system_reading_font_family(font_id: str | None) -> str | None:
    if not is_system_reading_font_id(font_id):
        return None
    family = str(font_id)[len(READING_FONT_SYSTEM_PREFIX) :].strip()
    return family or None


def _canonical_system_family(name: str | None) -> str | None:
    target = str(name or "").strip()
    if not target:
        return None
    for family in QFontDatabase.families():
        if family.casefold() == target.casefold():
            return family
    return None


def make_system_reading_font_id(family: str | None) -> str | None:
    canonical = _canonical_system_family(family)
    if canonical is None:
        return None
    return f"{READING_FONT_SYSTEM_PREFIX}{canonical}"


def normalize_reading_font_id(raw: str | None) -> str:
    if raw in _VALID_IDS:
        return raw
    if is_system_reading_font_id(raw):
        canonical = _canonical_system_family(parse_system_reading_font_family(raw))
        if canonical is not None:
            return f"{READING_FONT_SYSTEM_PREFIX}{canonical}"
    return DEFAULT_READING_FONT_ID


def reading_font_display_label(font_id: str | None) -> str:
    if not font_id:
        return _CHOICE_BY_ID[DEFAULT_READING_FONT_ID].label
    if font_id in _VALID_IDS:
        return _CHOICE_BY_ID[font_id].label
    if is_system_reading_font_id(font_id):
        family = parse_system_reading_font_family(font_id)
        if family:
            return f"{family} (system)"
    return _CHOICE_BY_ID[DEFAULT_READING_FONT_ID].label


def reading_font_label(font_id: str | None) -> str:
    return reading_font_display_label(font_id)


def _is_usable_system_family(name: str) -> bool:
    cleaned = str(name or "").strip()
    if not cleaned or cleaned.startswith("."):
        return False
    lower = cleaned.casefold()
    return not any(fragment in lower for fragment in _SYSTEM_FAMILY_SKIP_FRAGMENTS)


def system_reading_font_families(*, refresh: bool = False) -> tuple[str, ...]:
    global _system_families_cache
    if _system_families_cache is None or refresh:
        families = sorted(
            {family for family in QFontDatabase.families() if _is_usable_system_family(family)},
            key=str.casefold,
        )
        _system_families_cache = tuple(families)
    return _system_families_cache


def ensure_reading_fonts_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    inter_family = _FALLBACK_QT_FAMILY
    for family in QFontDatabase.families():
        if family.lower() == "inter":
            inter_family = family
            break
    _resolved_families[READING_FONT_INTER] = inter_family

    for choice in READING_FONT_CHOICES:
        if choice.font_id == READING_FONT_INTER:
            continue
        path = resource_path("assets", "fonts", choice.filename or "")
        family = _load_font_file(path)
        if family:
            _resolved_families[choice.font_id] = family
            continue
        logger.warning(
            "Reading font %s failed to load from %s; falling back to %s",
            choice.label,
            path,
            inter_family,
        )
        _resolved_families[choice.font_id] = inter_family


def _load_font_file(path) -> str | None:
    if not path.is_file():
        return None
    font_id = QFontDatabase.addApplicationFont(str(path))
    if font_id == -1:
        return None
    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else None


def reading_font_qt_family(font_id: str | None = None) -> str:
    ensure_reading_fonts_loaded()
    if font_id is None:
        from core.app_settings import get_ui_reading_font

        font_id = get_ui_reading_font()
    normalized = normalize_reading_font_id(font_id)
    if is_system_reading_font_id(normalized):
        family = parse_system_reading_font_family(normalized)
        if family is not None:
            return family
    return (
        _resolved_families.get(normalized)
        or _resolved_families.get(DEFAULT_READING_FONT_ID)
        or _FALLBACK_QT_FAMILY
    )


def reset_reading_font_cache_for_tests() -> None:
    """Clear loader state so tests can re-resolve families."""
    global _loaded, _system_families_cache
    _loaded = False
    _system_families_cache = None
    _resolved_families.clear()
