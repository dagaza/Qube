"""Theme family catalog — display names, grouping, and sibling lookup (§14 Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from core.theme.definition import ColorSchemeDefinition, merge_scheme_chain
from core.theme.families_policy import fallback_scheme_id_for_polarity
from core.theme.tokens import ThemeMode

Polarity = Literal["dark", "light"]


def mode_for_base_mode(base_mode: Polarity) -> ThemeMode:
    return ThemeMode.DARK if base_mode == "dark" else ThemeMode.LIGHT


def derived_mode_for_definition(definition: ColorSchemeDefinition) -> ThemeMode:
    return mode_for_base_mode(definition.base_mode)


_FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "catppuccin": "Catppuccin",
    "github": "GitHub",
    "solarized": "Solarized",
    "gruvbox": "Gruvbox",
    "nord": "Nord",
    "dracula": "Dracula",
    "slate": "Slate",
    "custom": "Custom",
}


def family_display_name(family: str) -> str:
    """User-facing family title."""
    if not family:
        return "Custom"
    return _FAMILY_DISPLAY_NAMES.get(family, family.replace("-", " ").title())


def variant_display_name(variant: str | None) -> str | None:
    if not variant:
        return None
    return variant.replace("-", " ").title()


def infer_family_from_scheme_id(scheme_id: str) -> str:
    """Best-effort family id from a scheme id slug."""
    slug = scheme_id.rsplit(".", 1)[-1]
    if slug.startswith("catppuccin-"):
        return "catppuccin"
    if slug.endswith("-dark"):
        return slug[: -len("-dark")]
    if slug.endswith("-light"):
        return slug[: -len("-light")]
    return slug


def infer_variant_from_scheme_id(scheme_id: str) -> str | None:
    slug = scheme_id.rsplit(".", 1)[-1]
    if slug.startswith("catppuccin-"):
        return slug.split("-", 1)[1]
    if slug.endswith("-dark"):
        return "dark"
    if slug.endswith("-light"):
        return "light"
    return None


def resolve_scheme_family(
    definition: ColorSchemeDefinition,
    registry: Mapping[str, ColorSchemeDefinition],
) -> str:
    if definition.family:
        return definition.family
    if definition.extends:
        try:
            parent = registry[definition.extends]
        except KeyError:
            pass
        else:
            return resolve_scheme_family(parent, registry)
    return infer_family_from_scheme_id(definition.id)


def resolve_scheme_variant(definition: ColorSchemeDefinition) -> str | None:
    if definition.variant is not None:
        return definition.variant
    return infer_variant_from_scheme_id(definition.id)


@dataclass(frozen=True)
class ThemePickerEntry:
    scheme_id: str
    display_name: str
    family: str
    family_display_name: str
    variant: str | None
    variant_display_name: str | None
    base_mode: Polarity
    search_text: str
    swatch_color: str = "#64748b"


@dataclass(frozen=True)
class ThemePickerModel:
    entries: tuple[ThemePickerEntry, ...]
    families: tuple[str, ...]


class ThemeCatalog:
    """Query layer over a color-scheme registry."""

    def __init__(self, registry: Mapping[str, ColorSchemeDefinition] | None = None) -> None:
        self._registry: dict[str, ColorSchemeDefinition] = dict(registry or {})
        self._family_members: dict[str, list[str]] = {}
        self._rebuild_indexes()

    @property
    def registry(self) -> dict[str, ColorSchemeDefinition]:
        return dict(self._registry)

    def register_many(self, schemes: Mapping[str, ColorSchemeDefinition]) -> None:
        self._registry.update(schemes)
        self._rebuild_indexes()

    def get_definition(self, scheme_id: str) -> ColorSchemeDefinition:
        return self._registry[scheme_id]

    def family_of(self, scheme_id: str) -> str:
        definition = self.get_definition(scheme_id)
        return resolve_scheme_family(definition, self._registry)

    def members_of_family(self, family: str) -> list[str]:
        members = list(self._family_members.get(family, []))
        members.sort(
            key=lambda sid: (
                0 if self._registry[sid].base_mode == "dark" else 1,
                self._registry[sid].name.lower(),
                sid,
            )
        )
        return members

    def _family_polarities(self, family: str) -> set[Polarity]:
        return {self._registry[sid].base_mode for sid in self.members_of_family(family)}

    def _is_custom_theme(self, scheme_id: str) -> bool:
        return scheme_id.startswith("user.")

    def display_name(self, scheme_id: str) -> str:
        definition = self.get_definition(scheme_id)
        if self._is_custom_theme(scheme_id):
            return definition.name

        family = self.family_of(scheme_id)
        polarities = self._family_polarities(family)
        if len(polarities) > 1:
            polarity_label = "Dark" if definition.base_mode == "dark" else "Light"
            return f"{family_display_name(family)} {polarity_label}"
        return definition.name

    def variant_label(self, scheme_id: str) -> str | None:
        definition = self.get_definition(scheme_id)
        variant = resolve_scheme_variant(definition)
        return variant_display_name(variant)

    def sibling_for_polarity(self, scheme_id: str, mode: ThemeMode) -> str | None:
        target: Polarity = "dark" if mode is ThemeMode.DARK else "light"
        definition = self.get_definition(scheme_id)
        if definition.base_mode == target:
            return scheme_id

        family = self.family_of(scheme_id)
        for member_id in self.members_of_family(family):
            member = self._registry[member_id]
            if member.base_mode == target:
                return member_id
        return None

    def fallback_for_family(self, family: str, mode: ThemeMode) -> str:
        polarity: Polarity = "dark" if mode is ThemeMode.DARK else "light"
        return fallback_scheme_id_for_polarity(family=family, polarity=polarity)

    def resolve_theme_choice(self, scheme_id: str) -> tuple[ThemeMode, str]:
        definition = self.get_definition(scheme_id)
        return derived_mode_for_definition(definition), scheme_id

    def themes_for_picker(self) -> ThemePickerModel:
        entries: list[ThemePickerEntry] = []
        families_seen: list[str] = []

        for scheme_id in sorted(
            self._registry,
            key=lambda sid: (self.family_of(sid), self.display_name(sid).lower()),
        ):
            definition = self._registry[scheme_id]
            family = self.family_of(scheme_id)
            if family not in families_seen:
                families_seen.append(family)

            variant = resolve_scheme_variant(definition)
            variant_label = variant_display_name(variant)
            display = self.display_name(scheme_id)
            family_label = family_display_name(family)
            search_parts = [
                scheme_id,
                display,
                family,
                family_label,
                definition.name,
                definition.base_mode,
            ]
            if variant:
                search_parts.extend([variant, variant_label or ""])
            search_text = " ".join(part.lower() for part in search_parts if part)
            merged = merge_scheme_chain(scheme_id, self._registry)
            swatch_color = merged.merged_overrides().get("background", "#64748b")

            entries.append(
                ThemePickerEntry(
                    scheme_id=scheme_id,
                    display_name=display,
                    family=family,
                    family_display_name=family_label,
                    variant=variant,
                    variant_display_name=variant_label,
                    base_mode=definition.base_mode,
                    search_text=search_text,
                    swatch_color=swatch_color,
                )
            )

        return ThemePickerModel(entries=tuple(entries), families=tuple(families_seen))

    def filter_picker_entries(
        self,
        query: str,
        *,
        model: ThemePickerModel | None = None,
    ) -> tuple[ThemePickerEntry, ...]:
        picker = model or self.themes_for_picker()
        needle = query.strip().lower()
        if not needle:
            return picker.entries
        return tuple(entry for entry in picker.entries if needle in entry.search_text)

    def _rebuild_indexes(self) -> None:
        self._family_members.clear()
        for scheme_id, definition in self._registry.items():
            family = resolve_scheme_family(definition, self._registry)
            self._family_members.setdefault(family, []).append(scheme_id)


def catalog_for_registry(
    registry: Mapping[str, ColorSchemeDefinition],
) -> ThemeCatalog:
    """Construct a catalog for a registry snapshot."""
    return ThemeCatalog(registry)
