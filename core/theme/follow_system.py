"""Follow-system theme preference helpers (§14 Phase 7 prep — UI in Phase 9).

When ``ThemeAppearancePreference.FOLLOW_SYSTEM`` is enabled (future), effective
polarity comes from the OS and the active scheme is the last-used theme for that
polarity, falling back to built-in defaults.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from core.theme.definition import ColorSchemeDefinition
from core.theme.schemes import default_scheme_id_for_mode
from core.theme.tokens import ThemeMode

KEY_THEME_APPEARANCE = "qube.ui.theme.appearance"
KEY_LAST_SCHEME_DARK = "qube.ui.color_scheme.last.dark"
KEY_LAST_SCHEME_LIGHT = "qube.ui.color_scheme.last.light"


class ThemeAppearancePreference(str, Enum):
    """User-facing theme appearance preference (follow-system ships in Phase 9)."""

    DARK = "dark"
    LIGHT = "light"
    FOLLOW_SYSTEM = "follow_system"


def parse_appearance_preference(raw: str | None) -> ThemeAppearancePreference:
    value = str(raw or ThemeAppearancePreference.DARK.value).strip().lower()
    try:
        return ThemeAppearancePreference(value)
    except ValueError:
        return ThemeAppearancePreference.DARK


def detect_system_polarity() -> ThemeMode:
    """Read OS light/dark preference via ``QStyleHints.colorScheme()`` when available."""
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QGuiApplication

        app = QGuiApplication.instance()
        if app is None:
            return ThemeMode.DARK
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Light:
            return ThemeMode.LIGHT
        if scheme == Qt.ColorScheme.Dark:
            return ThemeMode.DARK
    except Exception:
        pass
    return ThemeMode.DARK


def effective_mode_for_preference(preference: ThemeAppearancePreference) -> ThemeMode:
    if preference is ThemeAppearancePreference.FOLLOW_SYSTEM:
        return detect_system_polarity()
    return ThemeMode.DARK if preference is ThemeAppearancePreference.DARK else ThemeMode.LIGHT


def resolve_scheme_for_polarity(
    *,
    polarity: ThemeMode,
    last_scheme_by_polarity: Mapping[str, str],
    schemes: Mapping[str, ColorSchemeDefinition],
) -> str:
    """Pick a scheme for ``polarity``: last-used match, else the mode default."""
    key = polarity.value
    last_id = str(last_scheme_by_polarity.get(key) or "").strip()
    if last_id and last_id in schemes:
        definition = schemes[last_id]
        if definition.base_mode == key:
            return last_id
    return default_scheme_id_for_mode(key)


def resolve_active_theme_choice(
    *,
    preference: ThemeAppearancePreference,
    current_scheme_id: str,
    last_scheme_by_polarity: Mapping[str, str],
    schemes: Mapping[str, ColorSchemeDefinition],
) -> tuple[ThemeMode, str]:
    """Resolve effective mode and scheme id for the current appearance preference."""
    if preference is ThemeAppearancePreference.FOLLOW_SYSTEM:
        mode = detect_system_polarity()
    elif preference is ThemeAppearancePreference.DARK:
        mode = ThemeMode.DARK
    else:
        mode = ThemeMode.LIGHT

    if current_scheme_id in schemes:
        definition = schemes[current_scheme_id]
        if definition.base_mode == mode.value:
            return mode, current_scheme_id

    return mode, resolve_scheme_for_polarity(
        polarity=mode,
        last_scheme_by_polarity=last_scheme_by_polarity,
        schemes=schemes,
    )


def appearance_preference_label(preference: ThemeAppearancePreference) -> str:
    labels = {
        ThemeAppearancePreference.DARK: "Dark",
        ThemeAppearancePreference.LIGHT: "Light",
        ThemeAppearancePreference.FOLLOW_SYSTEM: "Follow system",
    }
    return labels[preference]
