"""Resolve color schemes and derive ``ResolvedTheme`` instances."""

from __future__ import annotations

from typing import Mapping

from core.theme.definition import ColorSchemeDefinition, merge_scheme_chain
from core.theme.schemes import validate_primitive_keys
from core.theme.strategies import get_strategy
from core.theme.tokens import CoreTokenSet, ResolvedTheme, ThemeMode


class ThemeResolver:
    def __init__(
        self,
        registry: Mapping[str, ColorSchemeDefinition] | None = None,
    ) -> None:
        self._registry: dict[str, ColorSchemeDefinition] = dict(registry or {})

    def register(self, definition: ColorSchemeDefinition) -> None:
        self._registry[definition.id] = definition

    def register_many(self, definitions: Mapping[str, ColorSchemeDefinition]) -> None:
        self._registry.update(definitions)

    def get_definition(self, scheme_id: str) -> ColorSchemeDefinition:
        return merge_scheme_chain(scheme_id, self._registry)

    def resolve(
        self,
        *,
        mode: ThemeMode,
        scheme_id: str,
        runtime_overrides: Mapping[str, str] | None = None,
    ) -> ResolvedTheme:
        merged = self.get_definition(scheme_id)
        primitive_values = dict(merged.overrides or {})
        if runtime_overrides:
            validate_primitive_keys(dict(runtime_overrides))
            primitive_values.update(runtime_overrides)

        core = CoreTokenSet.from_dict(primitive_values)
        strategy = get_strategy(merged.algorithm)
        return strategy.derive(
            core,
            scheme_id=merged.id,
            scheme_name=merged.name,
            mode=mode,
            algorithm=merged.algorithm,
        )
