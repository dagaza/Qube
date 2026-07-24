"""Theme coordinator — resolves, validates, applies, and persists themes."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from core.theme.applicator import ThemeApplicator
from core.theme.catalog import ThemeCatalog, derived_mode_for_definition, family_display_name
from core.theme.definition import ColorSchemeDefinition
from core.theme.families_policy import nav_fallback_primary_action_label
from core.theme.follow_system import (
    ThemeAppearancePreference,
    resolve_active_theme_choice,
)
from core.theme.io import export_color_scheme, import_color_scheme
from core.theme.overrides import sparse_core_overrides
from core.theme.polarity_toggle import PolarityToggleAction, PolarityToggleRequest
from core.theme.resolver import ThemeResolver
from core.theme.scheme_utils import (
    ensure_user_scheme_id,
    slugify_scheme_name,
    uniquify_scheme_id,
)
from core.theme.schemes import BUILTIN_SCHEMES, default_scheme_id_for_mode
from core.theme.storage import ThemeStorage
from core.theme.tokens import ResolvedTheme, ThemeMode
from core.theme.validation import ThemeValidationResult, ThemeValidator

# Ensure built-in derivation strategies are registered.
import core.theme.strategies  # noqa: F401

logger = logging.getLogger("Qube.ThemeManager")

if TYPE_CHECKING:
    from core.theme_toggle_profile import ThemeToggleProfiler


class ThemeManager:
    """Coordinates theme components. One instance per application — not a singleton."""

    def __init__(
        self,
        *,
        storage: ThemeStorage | None = None,
        resolver: ThemeResolver | None = None,
        applicator: ThemeApplicator | None = None,
        validator: ThemeValidator | None = None,
    ) -> None:
        self._storage = storage or ThemeStorage()
        self._resolver = resolver or ThemeResolver(BUILTIN_SCHEMES)
        self._applicator = applicator or ThemeApplicator()
        self._validator = validator or ThemeValidator()
        self._subscribers: list[Callable[[ResolvedTheme], None]] = []

        mode, scheme_id = self._storage.load()
        self._current = self._resolve(mode=mode, scheme_id=scheme_id)

    @property
    def current(self) -> ResolvedTheme:
        return self._current

    @property
    def mode(self) -> ThemeMode:
        return self._current.mode

    @property
    def is_dark(self) -> bool:
        return self._current.is_dark

    @property
    def scheme_id(self) -> str:
        return self._current.scheme_id

    @property
    def appearance_preference(self) -> ThemeAppearancePreference | None:
        return self._storage.appearance_preference

    def set_appearance_preference(
        self,
        preference: ThemeAppearancePreference,
        *,
        persist: bool = True,
    ) -> None:
        self._storage.save_appearance_preference(preference, persist=persist)

    def apply_from_appearance_preference(
        self,
        *,
        persist: bool = True,
        profiler: ThemeToggleProfiler | None = None,
    ) -> ResolvedTheme | None:
        preference = self.appearance_preference
        if preference is None:
            return None
        mode, scheme_id = resolve_active_theme_choice(
            preference=preference,
            current_scheme_id=self.scheme_id,
            last_scheme_by_polarity=self._storage.last_scheme_ids_by_polarity(),
            schemes=self.list_schemes(),
        )
        return self.apply(scheme_id=scheme_id, persist=persist, profiler=profiler)

    def sync_with_system_appearance(
        self,
        *,
        persist: bool = True,
        profiler: ThemeToggleProfiler | None = None,
    ) -> ResolvedTheme | None:
        """Re-apply theme when OS polarity changes under follow-system mode."""
        if self.appearance_preference is not ThemeAppearancePreference.FOLLOW_SYSTEM:
            return None
        return self.apply_from_appearance_preference(persist=persist, profiler=profiler)

    def subscribe(self, callback: Callable[[ResolvedTheme], None]) -> None:
        self._subscribers.append(callback)

    def preview_resolve(
        self,
        *,
        mode: ThemeMode | None = None,
        scheme_id: str | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> ResolvedTheme:
        resolved = self._resolve(
            mode=mode,
            scheme_id=scheme_id or self.scheme_id,
            overrides=overrides,
        )
        self._validator.validate(resolved)
        return resolved

    def validate(
        self,
        theme: ResolvedTheme | None = None,
    ) -> ThemeValidationResult:
        return self._validator.validate(theme or self._current)

    def apply(
        self,
        *,
        mode: ThemeMode | None = None,
        scheme_id: str | None = None,
        overrides: Mapping[str, str] | None = None,
        persist: bool = True,
        profiler: ThemeToggleProfiler | None = None,
    ) -> ResolvedTheme:
        resolved = self.preview_resolve(
            mode=mode,
            scheme_id=scheme_id,
            overrides=overrides,
        )
        self._current = resolved
        self._applicator.apply(resolved, profiler=profiler)
        if persist:
            self._storage.save(mode=resolved.mode, scheme_id=resolved.scheme_id)
        self._notify(resolved)
        return resolved

    def toggle_polarity(
        self,
        *,
        on_no_sibling: Callable[[PolarityToggleRequest], PolarityToggleAction] | None = None,
        prepare_apply: Callable[[], None] | None = None,
        persist: bool = True,
        profiler: ThemeToggleProfiler | None = None,
    ) -> ResolvedTheme | None:
        """Switch to the opposite polarity, preserving theme family when possible."""
        schemes = self.list_schemes()
        catalog = ThemeCatalog(schemes)
        current_id = self.scheme_id
        target_mode = ThemeMode.LIGHT if self.is_dark else ThemeMode.DARK
        sibling = catalog.sibling_for_polarity(current_id, target_mode)

        if sibling:
            if prepare_apply is not None:
                prepare_apply()
            return self.apply(scheme_id=sibling, persist=persist, profiler=profiler)

        family = catalog.family_of(current_id)
        fallback_id = catalog.fallback_for_family(family, target_mode)
        polarity = "light" if target_mode is ThemeMode.LIGHT else "dark"
        request = PolarityToggleRequest(
            family=family,
            family_display_name=family_display_name(family),
            current_scheme_id=current_id,
            current_display_name=catalog.display_name(current_id),
            target_mode=target_mode,
            fallback_scheme_id=fallback_id,
            fallback_display_name=catalog.display_name(fallback_id),
            primary_action_label=nav_fallback_primary_action_label(polarity=polarity),
        )

        if on_no_sibling is None:
            logger.warning(
                "No %s variant for scheme %s and no fallback callback provided",
                polarity,
                current_id,
            )
            return None

        action = on_no_sibling(request)
        if action is PolarityToggleAction.CANCEL:
            return None
        if action is PolarityToggleAction.CHOOSE_THEME:
            return None
        if action is PolarityToggleAction.APPLY_FALLBACK:
            if prepare_apply is not None:
                prepare_apply()
            return self.apply(scheme_id=fallback_id, persist=persist, profiler=profiler)

        return None

    def reload_custom_schemes(self) -> None:
        self._storage.reload_custom_schemes()
        self._resolver.register_many(self._storage.all_schemes())

    def list_schemes(self) -> dict:
        """All registered scheme definitions (built-in + custom)."""
        self._storage.reload_custom_schemes()
        self._resolver.register_many(self._storage.all_schemes())
        return self._storage.all_schemes()

    def list_scheme_ids(self) -> list[str]:
        schemes = self.list_schemes()

        def sort_key(sid: str) -> tuple[int, str, str]:
            tier = 0 if sid.startswith("builtin.") else 1
            return (tier, schemes[sid].name.lower(), sid)

        return sorted(schemes.keys(), key=sort_key)

    def get_scheme_definition(self, scheme_id: str) -> ColorSchemeDefinition:
        schemes = self.list_schemes()
        if scheme_id not in schemes:
            raise KeyError(f"Unknown color scheme: {scheme_id!r}")
        return schemes[scheme_id]

    def export_scheme_payload(self, scheme_id: str) -> dict[str, Any]:
        return export_color_scheme(self.get_scheme_definition(scheme_id))

    def export_scheme_to_path(self, scheme_id: str, path: Path) -> None:
        payload = self.export_scheme_payload(scheme_id)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def import_scheme_from_path(self, path: Path) -> ColorSchemeDefinition:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.import_scheme_payload(payload)

    def import_scheme_payload(self, payload: Mapping[str, Any]) -> ColorSchemeDefinition:
        definition = import_color_scheme(dict(payload), registry=self.list_schemes())
        scheme_id = ensure_user_scheme_id(definition.id, fallback_name=definition.name)
        existing = set(self.list_schemes())
        scheme_id = uniquify_scheme_id(scheme_id, existing)
        definition = replace(definition, id=scheme_id)
        self._storage.save_custom_scheme(definition)
        self.reload_custom_schemes()
        return definition

    def sparse_overrides_for_draft(
        self,
        *,
        mode: ThemeMode | None = None,
        scheme_id: str,
        overrides: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        base = self.preview_resolve(scheme_id=scheme_id)
        draft = self.preview_resolve(scheme_id=scheme_id, overrides=overrides)
        return sparse_core_overrides(base.core_tokens(), draft.core_tokens())

    def save_draft_as_custom_scheme(
        self,
        *,
        name: str,
        mode: ThemeMode | None = None,
        scheme_id: str,
        overrides: Mapping[str, str] | None = None,
    ) -> ColorSchemeDefinition:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Scheme name is required")
        resolved = self.preview_resolve(
            scheme_id=scheme_id,
            overrides=overrides,
        )
        validation = self._validator.validate(resolved)
        if not validation.can_save:
            message = validation.errors[0] if validation.errors else "Contrast is too low to save"
            raise ValueError(message)

        sparse = self.sparse_overrides_for_draft(
            scheme_id=scheme_id,
            overrides=overrides,
        )
        if not sparse:
            raise ValueError("No color customizations to save — adjust at least one token first")

        parent = self.get_scheme_definition(scheme_id)
        from core.theme.catalog import resolve_scheme_family

        base_id = f"user.{slugify_scheme_name(cleaned)}"
        new_id = uniquify_scheme_id(base_id, set(self.list_schemes()))
        parent_family = parent.family or resolve_scheme_family(parent, self.list_schemes())
        scheme_mode = resolved.mode
        definition = ColorSchemeDefinition(
            id=new_id,
            name=cleaned,
            base_mode=scheme_mode.value,  # type: ignore[arg-type]
            family=parent_family,
            variant=None,
            extends=scheme_id,
            algorithm=parent.algorithm,
            overrides=sparse,
        )
        self._storage.save_custom_scheme(definition)
        self.reload_custom_schemes()
        return definition

    def _resolve(
        self,
        *,
        mode: ThemeMode | None,
        scheme_id: str,
        overrides: Mapping[str, str] | None = None,
    ) -> ResolvedTheme:
        self._resolver.register_many(self._storage.all_schemes())
        schemes = self._storage.all_schemes()

        if scheme_id not in schemes:
            hint_mode = mode or ThemeMode.DARK
            logger.warning(
                "Unknown color scheme %r; falling back using mode hint %s",
                scheme_id,
                hint_mode.value,
            )
            scheme_id = default_scheme_id_for_mode(hint_mode.value)

        definition = schemes[scheme_id]
        derived_mode = derived_mode_for_definition(definition)
        if mode is not None and mode != derived_mode:
            logger.warning(
                "Ignoring requested theme mode %s; scheme %s requires %s",
                mode.value,
                scheme_id,
                derived_mode.value,
            )

        return self._resolver.resolve(
            mode=derived_mode,
            scheme_id=scheme_id,
            runtime_overrides=overrides,
        )

    def _notify(self, resolved: ResolvedTheme) -> None:
        for callback in list(self._subscribers):
            callback(resolved)
