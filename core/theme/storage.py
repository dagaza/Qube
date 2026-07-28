"""Theme persistence — settings keys and custom scheme files."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

from core.paths import user_data_root
from core.theme.catalog import derived_mode_for_definition
from core.theme.follow_system import (
    KEY_LAST_SCHEME_DARK,
    KEY_LAST_SCHEME_LIGHT,
    KEY_THEME_APPEARANCE,
    ThemeAppearancePreference,
    parse_appearance_preference,
    resolve_active_theme_choice,
)
from core.theme.definition import ColorSchemeDefinition
from core.theme.io import import_color_scheme
from core.theme.schemes import (
    BUILTIN_SCHEMES,
    DEFAULT_SCHEME_ID_DARK,
    default_scheme_id_for_mode,
)
from core.theme.tokens import ThemeMode

logger = logging.getLogger("Qube.ThemeStorage")

KEY_THEME_MODE = "qube.ui.theme.mode"
KEY_COLOR_SCHEME_ID = "qube.ui.color_scheme.id"


def themes_directory() -> Path:
    path = user_data_root() / "themes"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ThemeStorage:
    """Loads/saves theme mode and scheme id; discovers custom scheme JSON files."""

    def __init__(
        self,
        *,
        settings_get: Callable[[str, object], object] | None = None,
        settings_set: Callable[[str, object], None] | None = None,
        settings_contains: Callable[[str], bool] | None = None,
    ) -> None:
        self._get = settings_get
        self._set = settings_set
        self._contains = settings_contains
        self._mode = ThemeMode.DARK
        self._scheme_id = DEFAULT_SCHEME_ID_DARK
        self._last_scheme_by_polarity: dict[str, str] = {}
        self._appearance_preference: ThemeAppearancePreference | None = None
        self._custom_schemes: dict[str, ColorSchemeDefinition] = {}
        self.reload_custom_schemes()

    def reload_custom_schemes(self) -> None:
        self._custom_schemes.clear()
        root = themes_directory()
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                definition = import_color_scheme(payload)
                self._custom_schemes[definition.id] = definition
            except Exception as exc:
                logger.warning("Skipping invalid theme file %s: %s", path, exc)

    @property
    def mode(self) -> ThemeMode:
        if self._get is not None:
            raw = str(self._get(KEY_THEME_MODE, ThemeMode.DARK.value))
            try:
                return ThemeMode(raw)
            except ValueError:
                logger.warning("Invalid theme mode %r; defaulting to dark", raw)
                return ThemeMode.DARK
        return self._mode

    @property
    def scheme_id(self) -> str:
        if self._get is not None:
            return str(self._get(KEY_COLOR_SCHEME_ID, DEFAULT_SCHEME_ID_DARK))
        return self._scheme_id

    def save(self, *, mode: ThemeMode, scheme_id: str) -> None:
        if self._set is not None:
            self._set(KEY_THEME_MODE, mode.value)
            self._set(KEY_COLOR_SCHEME_ID, scheme_id)
            if mode is ThemeMode.DARK:
                self._set(KEY_LAST_SCHEME_DARK, scheme_id)
            else:
                self._set(KEY_LAST_SCHEME_LIGHT, scheme_id)
        self._mode = mode
        self._scheme_id = scheme_id
        self._last_scheme_by_polarity[mode.value] = scheme_id

    def last_scheme_ids_by_polarity(self) -> dict[str, str]:
        """Last-applied scheme id per polarity (follow-system prep)."""
        result: dict[str, str] = dict(self._last_scheme_by_polarity)
        if self._get is not None:
            dark = str(self._get(KEY_LAST_SCHEME_DARK, "") or "").strip()
            light = str(self._get(KEY_LAST_SCHEME_LIGHT, "") or "").strip()
            if dark:
                result[ThemeMode.DARK.value] = dark
            if light:
                result[ThemeMode.LIGHT.value] = light
        return result

    @property
    def appearance_preference(self) -> ThemeAppearancePreference | None:
        """Persisted appearance preference, or ``None`` for legacy scheme-driven mode."""
        if self._get is not None:
            # Schema default is "dark", but unset means legacy scheme-driven load().
            if self._contains is not None and not self._contains(KEY_THEME_APPEARANCE):
                return None
            raw = self._get(KEY_THEME_APPEARANCE, None)
            if raw is None or str(raw).strip() == "":
                return None
            return parse_appearance_preference(str(raw))
        return self._appearance_preference

    def save_appearance_preference(
        self,
        preference: ThemeAppearancePreference,
        *,
        persist: bool = True,
    ) -> None:
        if persist and self._set is not None:
            self._set(KEY_THEME_APPEARANCE, preference.value)
        self._appearance_preference = preference

    def load(self) -> tuple[ThemeMode, str]:
        preference = self.appearance_preference
        scheme_id = self.scheme_id
        schemes = self.all_schemes()

        if preference is not None:
            if scheme_id not in schemes:
                logger.warning(
                    "Unknown stored color scheme %r; resolving via appearance %s",
                    scheme_id,
                    preference.value,
                )
                scheme_id = ""
            mode, resolved_scheme_id = resolve_active_theme_choice(
                preference=preference,
                current_scheme_id=scheme_id,
                last_scheme_by_polarity=self.last_scheme_ids_by_polarity(),
                schemes=schemes,
            )
            if resolved_scheme_id not in schemes:
                resolved_scheme_id = default_scheme_id_for_mode(mode.value)
            definition = schemes[resolved_scheme_id]
            derived_mode = derived_mode_for_definition(definition)
            if derived_mode != mode or resolved_scheme_id != self.scheme_id:
                logger.info(
                    "Resolved appearance %s -> mode %s scheme %s",
                    preference.value,
                    derived_mode.value,
                    resolved_scheme_id,
                )
            self.save(mode=derived_mode, scheme_id=resolved_scheme_id)
            return derived_mode, resolved_scheme_id

        stored_mode = self.mode
        if scheme_id not in schemes:
            logger.warning(
                "Unknown stored color scheme %r; falling back for mode %s",
                scheme_id,
                stored_mode.value,
            )
            scheme_id = default_scheme_id_for_mode(stored_mode.value)

        definition = schemes[scheme_id]
        derived_mode = derived_mode_for_definition(definition)

        original_scheme_id = self.scheme_id
        if derived_mode != stored_mode or scheme_id != original_scheme_id:
            if derived_mode != stored_mode:
                logger.info(
                    "Repaired stored theme mode %s -> %s for scheme %s",
                    stored_mode.value,
                    derived_mode.value,
                    scheme_id,
                )
            self.save(mode=derived_mode, scheme_id=scheme_id)

        return derived_mode, scheme_id

    def all_schemes(self) -> dict[str, ColorSchemeDefinition]:
        return {**BUILTIN_SCHEMES, **self._custom_schemes}

    def save_custom_scheme(self, definition: ColorSchemeDefinition) -> Path:
        from core.theme.io import export_color_scheme

        path = themes_directory() / f"{definition.id.replace('/', '_')}.json"
        path.write_text(
            json.dumps(export_color_scheme(definition), indent=2) + "\n",
            encoding="utf-8",
        )
        self._custom_schemes[definition.id] = definition
        return path

    def get_custom_scheme(self, scheme_id: str) -> ColorSchemeDefinition | None:
        return self._custom_schemes.get(scheme_id)

    def with_scheme(self, definition: ColorSchemeDefinition) -> ColorSchemeDefinition:
        """Return definition registered in custom storage (for tests)."""
        self._custom_schemes[definition.id] = definition
        return definition


def theme_storage_from_app_settings() -> ThemeStorage:
    """``ThemeStorage`` backed by ``core.app_settings`` / ``settings.json``."""
    from core import app_settings

    store = app_settings._store()

    def _get(key: str, default: object) -> object:
        return store.get(key, default)

    def _set(key: str, value: object) -> None:
        store.set(key, value)

    return ThemeStorage(
        settings_get=_get,
        settings_set=_set,
        settings_contains=store.contains,
    )
