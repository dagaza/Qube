"""Color scheme JSON import/export."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from core.theme.catalog import resolve_scheme_family
from core.theme.definition import ColorSchemeDefinition
from core.theme.families_policy import (
    EXPORT_SCHEMA_VERSION,
    IMPORT_SCHEMA_VERSION_MAX,
    IMPORT_SCHEMA_VERSION_MIN,
)
from core.theme.schemes import BUILTIN_SCHEMES, validate_primitive_keys

SCHEMA_VERSION = EXPORT_SCHEMA_VERSION

_VALID_SUPPORTS = frozenset({"dark", "light"})


def export_color_scheme(definition: ColorSchemeDefinition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "id": definition.id,
        "name": definition.name,
        "base_mode": definition.base_mode,
        "algorithm": definition.algorithm,
    }
    if definition.extends:
        payload["extends"] = definition.extends
    if definition.family:
        payload["family"] = definition.family
    if definition.variant:
        payload["variant"] = definition.variant
    if definition.author:
        payload["author"] = definition.author
    if definition.description:
        payload["description"] = definition.description
    if definition.supports:
        payload["supports"] = list(definition.supports)
    overrides = definition.merged_overrides()
    if overrides:
        payload["overrides"] = dict(overrides)
    return payload


def _parse_supports(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("supports must be an array")
    supports: list[str] = []
    for item in raw:
        value = str(item).strip().lower()
        if value not in _VALID_SUPPORTS:
            raise ValueError(
                f"Invalid supports entry: {item!r} (expected 'dark' and/or 'light')"
            )
        if value not in supports:
            supports.append(value)
    return tuple(supports)


def import_color_scheme(
    payload: dict[str, Any],
    *,
    registry: Mapping[str, ColorSchemeDefinition] | None = None,
) -> ColorSchemeDefinition:
    schema = payload.get("schema")
    if schema is None:
        raise ValueError("Color scheme schema version is required")
    if not isinstance(schema, int) or not (
        IMPORT_SCHEMA_VERSION_MIN <= schema <= IMPORT_SCHEMA_VERSION_MAX
    ):
        raise ValueError(
            f"Unsupported color scheme schema version: {schema!r} "
            f"(expected {IMPORT_SCHEMA_VERSION_MIN}–{IMPORT_SCHEMA_VERSION_MAX})"
        )

    scheme_id = str(payload.get("id") or "").strip()
    name = str(payload.get("name") or scheme_id).strip()
    base_mode = str(payload.get("base_mode") or "").strip().lower()
    if base_mode not in ("dark", "light"):
        raise ValueError(f"Invalid base_mode: {base_mode!r}")
    if not scheme_id:
        raise ValueError("Color scheme id is required")

    overrides_raw = payload.get("overrides") or {}
    if not isinstance(overrides_raw, dict):
        raise ValueError("overrides must be an object")
    overrides = {str(k): str(v) for k, v in overrides_raw.items()}
    validate_primitive_keys(overrides)

    extends = payload.get("extends")
    extends_id = str(extends).strip() if extends else None
    algorithm = str(payload.get("algorithm") or "default").strip() or "default"
    family_raw = payload.get("family")
    family = str(family_raw).strip() if family_raw else ""
    variant_raw = payload.get("variant")
    variant = str(variant_raw).strip() if variant_raw else None
    author = str(payload.get("author") or "").strip()
    description = str(payload.get("description") or "").strip()
    supports = _parse_supports(payload.get("supports"))

    definition = ColorSchemeDefinition(
        id=scheme_id,
        name=name or scheme_id,
        base_mode=base_mode,  # type: ignore[arg-type]
        family=family,
        variant=variant,
        extends=extends_id,
        algorithm=algorithm,
        overrides=overrides,
        author=author,
        description=description,
        supports=supports,
    )

    if not definition.family:
        lookup_registry = registry or BUILTIN_SCHEMES
        inferred_family = resolve_scheme_family(definition, lookup_registry)
        definition = replace(definition, family=inferred_family)

    return definition
