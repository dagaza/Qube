"""Built-in color scheme registry."""

from __future__ import annotations

from core.theme.definition import ColorSchemeDefinition
from core.theme.tokens import CORE_TOKEN_KEYS

CATPUCCIN_MOCHA_PRIMITIVES: dict[str, str] = {
    "background": "#1e1e2e",
    # Nav/tools chrome: rgba(0,0,0,0.15) over background (pre-themes-v1 NavSidebar).
    "surface": "#1a1a27",
    "sidebar_surface": "#232337",
    "surface_elevated": "#313244",
    "text_primary": "#cdd6f4",
    "text_secondary": "#a6adc8",
    "border": "rgba(255,255,255,0.1)",
    "accent": "#8b5cf6",
    "success": "#34d399",
    "warning": "#fbbf24",
    "error": "#f87171",
    "info": "#fb923c",
}

CATPUCCIN_LATTE_PRIMITIVES: dict[str, str] = {
    "background": "#eff1f5",
    "surface": "#e6e9ef",
    "sidebar_surface": "#dce0e8",
    "surface_elevated": "#ffffff",
    "text_primary": "#4c4f69",
    "text_secondary": "#5c5f77",
    "border": "#acb0be",
    "accent": "#8b5cf6",
    "success": "#40a02b",
    "warning": "#df8e1d",
    "error": "#d20f39",
    "info": "#fe640b",
}

SLATE_PRIMITIVES: dict[str, str] = {
    "background": "#f1f5f9",
    "surface": "#f8fafc",
    "sidebar_surface": "#E9EFF5",
    "surface_elevated": "#ffffff",
    "text_primary": "#1e293b",
    "text_secondary": "#475569",
    "border": "#cbd5e1",
    "accent": "#8b5cf6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "info": "#ea580c",
}

NORD_PRIMITIVES: dict[str, str] = {
    **CATPUCCIN_MOCHA_PRIMITIVES,
    "background": "#2e3440",
    "surface": "#272c36",
    "sidebar_surface": "#3b4252",
    "surface_elevated": "#434c5e",
    "text_primary": "#eceff4",
    "text_secondary": "#d8dee9",
    "border": "rgba(236,239,244,0.12)",
    "accent": "#88c0d0",
    "success": "#a3be8c",
    "warning": "#ebcb8b",
    "error": "#bf616a",
    "info": "#d08770",
}

NORD_LIGHT_PRIMITIVES: dict[str, str] = {
    "background": "#eceff4",
    "surface": "#e5e9f0",
    "sidebar_surface": "#d8dee9",
    "surface_elevated": "#d8dee9",
    "text_primary": "#2e3440",
    "text_secondary": "#4c566a",
    "border": "#d8dee9",
    "accent": "#5e81ac",
    "success": "#a3be8c",
    "warning": "#ebcb8b",
    "error": "#bf616a",
    "info": "#d08770",
}

DRACULA_PRIMITIVES: dict[str, str] = {
    "background": "#282a36",
    "surface": "#22242e",
    "sidebar_surface": "#21222c",
    "surface_elevated": "#343746",
    "text_primary": "#f8f8f2",
    "text_secondary": "#6272a4",
    "border": "rgba(255,255,255,0.1)",
    "accent": "#bd93f9",
    "success": "#50fa7b",
    "warning": "#f1fa8c",
    "error": "#ff5555",
    "info": "#8be9fd",
}

GRUVBOX_DARK_PRIMITIVES: dict[str, str] = {
    "background": "#282828",
    "surface": "#222222",
    "sidebar_surface": "#32302f",
    "surface_elevated": "#3c3836",
    "text_primary": "#ebdbb2",
    "text_secondary": "#a89984",
    "border": "rgba(235,219,178,0.12)",
    "accent": "#b16286",
    "success": "#b8bb26",
    "warning": "#fabd2f",
    "error": "#fb4934",
    "info": "#83a598",
}

GRUVBOX_LIGHT_PRIMITIVES: dict[str, str] = {
    "background": "#fbf1c7",
    "surface": "#f2e5bc",
    "sidebar_surface": "#ebdbb2",
    "surface_elevated": "#ebdbb2",
    "text_primary": "#3c3836",
    "text_secondary": "#7c6f64",
    "border": "#d5c4a1",
    "accent": "#b16286",
    "success": "#98971a",
    "warning": "#d79921",
    "error": "#cc241d",
    "info": "#458588",
}

SOLARIZED_DARK_PRIMITIVES: dict[str, str] = {
    "background": "#002b36",
    "surface": "#00252e",
    "sidebar_surface": "#073642",
    "surface_elevated": "#0a4a58",
    "text_primary": "#839496",
    "text_secondary": "#586e75",
    "border": "rgba(131,148,150,0.25)",
    "accent": "#268bd2",
    "success": "#859900",
    "warning": "#b58900",
    "error": "#dc322f",
    "info": "#2aa198",
}

SOLARIZED_LIGHT_PRIMITIVES: dict[str, str] = {
    "background": "#fdf6e3",
    "surface": "#d7d1c1",
    "sidebar_surface": "#eee8d5",
    "surface_elevated": "#fdf6e3",
    "text_primary": "#657b83",
    "text_secondary": "#586e75",
    "border": "#93a1a1",
    "accent": "#268bd2",
    "success": "#859900",
    "warning": "#b58900",
    "error": "#dc322f",
    "info": "#2aa198",
}

GITHUB_DARK_PRIMITIVES: dict[str, str] = {
    "background": "#0d1117",
    "surface": "#0b0e14",
    "sidebar_surface": "#161b22",
    "surface_elevated": "#21262d",
    "text_primary": "#c9d1d9",
    "text_secondary": "#8b949e",
    "border": "#30363d",
    "accent": "#58a6ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
    "info": "#79c0ff",
}

GITHUB_LIGHT_PRIMITIVES: dict[str, str] = {
    "background": "#ffffff",
    "surface": "#f6f8fa",
    "sidebar_surface": "#eaeef2",
    "surface_elevated": "#ffffff",
    "text_primary": "#24292f",
    "text_secondary": "#57606a",
    "border": "#d0d7de",
    "accent": "#0969da",
    "success": "#1a7f37",
    "warning": "#9a6700",
    "error": "#cf222e",
    "info": "#0969da",
}

DEFAULT_SCHEME_ID_DARK = "builtin.catppuccin-mocha"
DEFAULT_SCHEME_ID_LIGHT = "builtin.slate"
BUILTIN_CATPUCCIN_LATTE_ID = "builtin.catppuccin-latte"
BUILTIN_NORD_LIGHT_ID = "builtin.nord-light"
BUILTIN_GRUVBOX_LIGHT_ID = "builtin.gruvbox-light"

BUILTIN_SCHEME_IDS: tuple[str, ...] = (
    DEFAULT_SCHEME_ID_DARK,
    BUILTIN_CATPUCCIN_LATTE_ID,
    DEFAULT_SCHEME_ID_LIGHT,
    "builtin.nord",
    BUILTIN_NORD_LIGHT_ID,
    "builtin.dracula",
    "builtin.gruvbox-dark",
    BUILTIN_GRUVBOX_LIGHT_ID,
    "builtin.solarized-dark",
    "builtin.solarized-light",
    "builtin.github-dark",
    "builtin.github-light",
)

BUILTIN_SCHEMES: dict[str, ColorSchemeDefinition] = {
    DEFAULT_SCHEME_ID_DARK: ColorSchemeDefinition(
        id=DEFAULT_SCHEME_ID_DARK,
        name="Catppuccin Mocha",
        base_mode="dark",
        family="catppuccin",
        variant="mocha",
        algorithm="catppuccin",
        overrides=CATPUCCIN_MOCHA_PRIMITIVES,
    ),
    BUILTIN_CATPUCCIN_LATTE_ID: ColorSchemeDefinition(
        id=BUILTIN_CATPUCCIN_LATTE_ID,
        name="Catppuccin Latte",
        base_mode="light",
        family="catppuccin",
        variant="latte",
        algorithm="catppuccin",
        overrides=CATPUCCIN_LATTE_PRIMITIVES,
    ),
    DEFAULT_SCHEME_ID_LIGHT: ColorSchemeDefinition(
        id=DEFAULT_SCHEME_ID_LIGHT,
        name="Slate",
        base_mode="light",
        family="slate",
        variant=None,
        algorithm="default",
        overrides=SLATE_PRIMITIVES,
    ),
    "builtin.nord": ColorSchemeDefinition(
        id="builtin.nord",
        name="Nord",
        base_mode="dark",
        family="nord",
        variant="dark",
        algorithm="nord",
        overrides=NORD_PRIMITIVES,
    ),
    BUILTIN_NORD_LIGHT_ID: ColorSchemeDefinition(
        id=BUILTIN_NORD_LIGHT_ID,
        name="Nord Light",
        base_mode="light",
        family="nord",
        variant="light",
        algorithm="nord",
        overrides=NORD_LIGHT_PRIMITIVES,
    ),
    "builtin.dracula": ColorSchemeDefinition(
        id="builtin.dracula",
        name="Dracula",
        base_mode="dark",
        family="dracula",
        variant=None,
        algorithm="default",
        overrides=DRACULA_PRIMITIVES,
    ),
    "builtin.gruvbox-dark": ColorSchemeDefinition(
        id="builtin.gruvbox-dark",
        name="Gruvbox Dark",
        base_mode="dark",
        family="gruvbox",
        variant="dark",
        algorithm="default",
        overrides=GRUVBOX_DARK_PRIMITIVES,
    ),
    BUILTIN_GRUVBOX_LIGHT_ID: ColorSchemeDefinition(
        id=BUILTIN_GRUVBOX_LIGHT_ID,
        name="Gruvbox Light",
        base_mode="light",
        family="gruvbox",
        variant="light",
        algorithm="default",
        overrides=GRUVBOX_LIGHT_PRIMITIVES,
    ),
    "builtin.solarized-dark": ColorSchemeDefinition(
        id="builtin.solarized-dark",
        name="Solarized Dark",
        base_mode="dark",
        family="solarized",
        variant="dark",
        algorithm="default",
        overrides=SOLARIZED_DARK_PRIMITIVES,
    ),
    "builtin.solarized-light": ColorSchemeDefinition(
        id="builtin.solarized-light",
        name="Solarized Light",
        base_mode="light",
        family="solarized",
        variant="light",
        algorithm="default",
        overrides=SOLARIZED_LIGHT_PRIMITIVES,
    ),
    "builtin.github-dark": ColorSchemeDefinition(
        id="builtin.github-dark",
        name="GitHub Dark",
        base_mode="dark",
        family="github",
        variant="dark",
        algorithm="default",
        overrides=GITHUB_DARK_PRIMITIVES,
    ),
    "builtin.github-light": ColorSchemeDefinition(
        id="builtin.github-light",
        name="GitHub Light",
        base_mode="light",
        family="github",
        variant="light",
        algorithm="default",
        overrides=GITHUB_LIGHT_PRIMITIVES,
    ),
}


def default_scheme_id_for_mode(mode: str) -> str:
    return DEFAULT_SCHEME_ID_DARK if mode == "dark" else DEFAULT_SCHEME_ID_LIGHT


def validate_primitive_keys(overrides: dict[str, str]) -> None:
    unknown = sorted(set(overrides) - set(CORE_TOKEN_KEYS))
    if unknown:
        raise ValueError(
            "Overrides may only contain core primitive tokens; "
            f"unknown keys: {', '.join(unknown)}"
        )
