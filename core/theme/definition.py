"""Color scheme definition and inheritance merge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping


@dataclass(frozen=True)
class ColorSchemeDefinition:
    id: str
    name: str
    base_mode: Literal["dark", "light"]
    family: str = ""
    variant: str | None = None
    extends: str | None = None
    algorithm: str = "default"
    overrides: Mapping[str, str] | None = None
    author: str = ""
    description: str = ""
    supports: tuple[str, ...] = ()

    def merged_overrides(self) -> dict[str, str]:
        return dict(self.overrides or {})


def _infer_family_from_scheme_id(scheme_id: str) -> str:
    slug = scheme_id.rsplit(".", 1)[-1]
    if slug.startswith("catppuccin-"):
        return "catppuccin"
    if slug.endswith("-dark"):
        return slug[: -len("-dark")]
    if slug.endswith("-light"):
        return slug[: -len("-light")]
    return slug


def merge_scheme_chain(
    scheme_id: str,
    registry: Mapping[str, ColorSchemeDefinition],
) -> ColorSchemeDefinition:
    """Resolve ``extends`` chain; child overrides win. Detect cycles."""
    visited: list[str] = []
    current_id: str | None = scheme_id
    chain: list[ColorSchemeDefinition] = []

    while current_id is not None:
        if current_id in visited:
            raise ValueError(f"Color scheme inheritance cycle detected at {current_id!r}")
        visited.append(current_id)
        try:
            definition = registry[current_id]
        except KeyError as exc:
            raise KeyError(f"Unknown color scheme: {current_id!r}") from exc
        chain.append(definition)
        current_id = definition.extends

    if not chain:
        raise KeyError(f"Unknown color scheme: {scheme_id!r}")

    merged: dict[str, str] = {}
    algorithm = "default"
    base_mode = chain[-1].base_mode
    name = chain[0].name
    resolved_id = chain[0].id
    family = chain[0].family
    variant = chain[0].variant
    if not family:
        for definition in chain[1:]:
            if definition.family:
                family = definition.family
                break
    if not family:
        family = _infer_family_from_scheme_id(resolved_id)

    for definition in reversed(chain):
        merged.update(definition.merged_overrides())
        if definition.algorithm:
            algorithm = definition.algorithm
        base_mode = definition.base_mode

    return ColorSchemeDefinition(
        id=resolved_id,
        name=name,
        base_mode=base_mode,
        family=family,
        variant=variant,
        extends=None,
        algorithm=algorithm,
        overrides=merged,
    )
