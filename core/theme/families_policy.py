"""Locked UX policy for theme families (design doc §14 Phase 0).

Encodes decisions locked before Phase 1 implementation. Later phases (catalog,
nav toggle, Settings UI, import/export) must read from here rather than
re-deciding behavior ad hoc.

See ``docs/theme_customization_design.md`` §14.10.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal

from core.theme.schemes import DEFAULT_SCHEME_ID_DARK, DEFAULT_SCHEME_ID_LIGHT

Polarity = Literal["dark", "light"]


class NavPolarityFallbackStyle(str, Enum):
    """How to prompt when nav polarity toggle has no family sibling."""

    MODAL = "modal"


class RuntimeOverridesPolicy(str, Enum):
    """How unsaved color edits behave relative to the active preset."""

    PERSIST_WITH_SCHEME = "persist_with_scheme"


class DisplayNamePolicy(str, Enum):
    """Where user-facing theme titles come from."""

    CATALOG_COMPUTED = "catalog_computed"


# --- Locked decisions (Phase 0, 2026-07-24) ---------------------------------

NAV_POLARITY_FALLBACK_STYLE: Final = NavPolarityFallbackStyle.MODAL
"""Use a small ``PrestigeDialog`` — never silently swap scheme on nav toggle."""

RUNTIME_OVERRIDES_POLICY: Final = RuntimeOverridesPolicy.PERSIST_WITH_SCHEME
"""Keep sparse overrides with the active scheme id until Save as (current behavior)."""

EXPERIMENTAL_MODE_DECOUPLE_ENABLED: Final = False
"""Defer Settings → Advanced “mode ≠ palette” override until explicitly requested."""

DISPLAY_NAME_POLICY: Final = DisplayNamePolicy.CATALOG_COMPUTED
"""Compute display names in ``ThemeCatalog.display_name()``; do not rename registry ids."""

EXPORT_SCHEMA_VERSION: Final = 2
"""Current export schema; ``io.SCHEMA_VERSION`` matches this value."""

IMPORT_SCHEMA_VERSION_MIN: Final = 1
IMPORT_SCHEMA_VERSION_MAX: Final = 2
"""Import accepts v1 and v2 payloads once Phase 6 lands."""

GLOBAL_LIGHT_FALLBACK_SCHEME_ID: Final = DEFAULT_SCHEME_ID_LIGHT
GLOBAL_DARK_FALLBACK_SCHEME_ID: Final = DEFAULT_SCHEME_ID_DARK
"""Global defaults when a family has no sibling for the requested polarity."""

FAMILY_POLARITY_FALLBACK_SCHEME_IDS: Final[dict[str, dict[Polarity, str]]] = {
    "dracula": {"light": DEFAULT_SCHEME_ID_LIGHT},
    "slate": {"dark": DEFAULT_SCHEME_ID_DARK},
}
"""Per-family polarity fallbacks when no sibling exists (Phase 7)."""

# Copy hints for Phase 3 nav fallback modal (display names resolved at runtime).
NAV_FALLBACK_MODAL_TITLE: Final = "Light theme unavailable"
NAV_FALLBACK_MODAL_TITLE_DARK: Final = "Dark theme unavailable"
NAV_FALLBACK_CHOOSE_THEME_ACTION: Final = "Choose theme…"
NAV_FALLBACK_CANCEL_ACTION: Final = "Cancel"


def fallback_scheme_id_for_polarity(*, family: str, polarity: Polarity) -> str:
    """Scheme id to offer when ``family`` has no variant for ``polarity``."""
    family_map = FAMILY_POLARITY_FALLBACK_SCHEME_IDS.get(family)
    if family_map is not None:
        scheme_id = family_map.get(polarity)
        if scheme_id:
            return scheme_id
    if polarity == "light":
        return GLOBAL_LIGHT_FALLBACK_SCHEME_ID
    return GLOBAL_DARK_FALLBACK_SCHEME_ID


def nav_fallback_primary_action_label(*, polarity: Polarity) -> str:
    """Primary fallback button label for nav polarity prompt."""
    from core.theme.catalog import catalog_for_registry
    from core.theme.schemes import BUILTIN_SCHEMES

    scheme_id = fallback_scheme_id_for_polarity(family="", polarity=polarity)
    display = catalog_for_registry(BUILTIN_SCHEMES).display_name(scheme_id)
    return f"Switch to {display}"
