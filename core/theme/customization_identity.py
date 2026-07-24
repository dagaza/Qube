"""Customization identity copy for Settings → Themes (§14 Phase 5)."""

from __future__ import annotations

from typing import Mapping

from core.theme.catalog import ThemeCatalog


def lineage_root_scheme_id(scheme_id: str, catalog: ThemeCatalog) -> str:
    """Walk ``extends`` to the root preset id."""
    visited: set[str] = set()
    current = scheme_id
    while current not in visited:
        visited.add(current)
        definition = catalog.get_definition(current)
        if not definition.extends:
            return current
        current = definition.extends
    return scheme_id


def suggested_custom_theme_name(scheme_id: str, catalog: ThemeCatalog) -> str:
    root_id = lineage_root_scheme_id(scheme_id, catalog)
    return f"My {catalog.display_name(root_id)}"


def customization_identity_text(
    *,
    scheme_id: str,
    overrides: Mapping[str, str] | None,
    catalog: ThemeCatalog,
) -> str:
    """User-facing Customize card identity line."""
    has_overrides = bool(overrides)
    is_user_theme = scheme_id.startswith("user.")
    display = catalog.display_name(scheme_id)

    if has_overrides:
        if is_user_theme:
            return f"Custom · {display} (unsaved changes)"
        return f"Custom · based on {display}"

    if is_user_theme:
        root_id = lineage_root_scheme_id(scheme_id, catalog)
        if root_id != scheme_id:
            return f"{display} · based on {catalog.display_name(root_id)}"
        return display

    return f"Based on: {display}"


def customization_is_active(overrides: Mapping[str, str] | None) -> bool:
    return bool(overrides)
