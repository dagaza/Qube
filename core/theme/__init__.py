"""Python-first theme system for Qube."""

from core.theme.accessors import theme_for
from core.theme.applicator import ThemeApplicator
from core.theme.catalog import ThemeCatalog, ThemePickerEntry, ThemePickerModel, catalog_for_registry, derived_mode_for_definition
from core.theme.definition import ColorSchemeDefinition, merge_scheme_chain
from core.theme.feature_flags import is_generated_theme_enabled
from core.theme.io import SCHEMA_VERSION, export_color_scheme, import_color_scheme
from core.theme.manager import ThemeManager
from core.theme.polarity_toggle import PolarityToggleAction, PolarityToggleRequest
from core.theme.resolver import ThemeResolver
from core.theme.schemes import (
    BUILTIN_SCHEMES,
    BUILTIN_CATPUCCIN_LATTE_ID,
    DEFAULT_SCHEME_ID_DARK,
    DEFAULT_SCHEME_ID_LIGHT,
)
from core.theme.storage import ThemeStorage, theme_storage_from_app_settings
from core.theme.stylesheet import render_stylesheet
from core.theme.tokens import CORE_TOKEN_KEYS, CoreTokenSet, ResolvedTheme, ThemeMode
from core.theme.validation import ThemeValidationResult, ThemeValidator
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import apply_theme_style, theme_color, theme_style

__all__ = [
    "BUILTIN_CATPUCCIN_LATTE_ID",
    "BUILTIN_SCHEMES",
    "CORE_TOKEN_KEYS",
    "ColorSchemeDefinition",
    "CoreTokenSet",
    "DEFAULT_SCHEME_ID_DARK",
    "DEFAULT_SCHEME_ID_LIGHT",
    "PolarityToggleAction",
    "PolarityToggleRequest",
    "ResolvedTheme",
    "SCHEMA_VERSION",
    "ThemeApplicator",
    "ThemeCatalog",
    "ThemeManager",
    "ThemeMode",
    "ThemePickerEntry",
    "ThemePickerModel",
    "ThemeResolver",
    "ThemeStorage",
    "ThemeValidationResult",
    "ThemeValidator",
    "export_color_scheme",
    "import_color_scheme",
    "is_generated_theme_enabled",
    "merge_scheme_chain",
    "render_stylesheet",
    "apply_theme_style",
    "theme_color",
    "theme_for",
    "theme_storage_from_app_settings",
    "theme_style",
    "view_resolved_theme",
    "catalog_for_registry",
    "derived_mode_for_definition",
]
