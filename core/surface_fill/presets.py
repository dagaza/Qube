"""Built-in wallpaper preset catalog (Phase 0 stub — assets ship in Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from core.paths import resource_path
from core.surface_fill.models import (
    GradientStop,
    Wallpaper,
    WallpaperGradient,
    WallpaperImage,
    WallpaperPreset,
    WallpaperSolid,
)

PresetKind = Literal["solid", "gradient", "image"]

BUILTIN_PRESET_PREFIX = "builtin."


@dataclass(frozen=True)
class PresetDefinition:
    id: str
    name: str
    kind: PresetKind
    wallpaper: Wallpaper
    asset_relpath: str | None = None

    @property
    def preset_id(self) -> str:
        return self.id


# Gradient / solid presets are defined inline; image presets reference future bundled assets.
_BUILTIN_PRESETS: dict[str, PresetDefinition] = {
    "builtin.paper": PresetDefinition(
        id="builtin.paper",
        name="Paper",
        kind="solid",
        wallpaper=WallpaperSolid(color="#e8e4dc"),
    ),
    "builtin.mist": PresetDefinition(
        id="builtin.mist",
        name="Mist",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="vertical",
            stops=(
                GradientStop(0.0, "#2e3440"),
                GradientStop(1.0, "#3b4252"),
            ),
        ),
    ),
    "builtin.aurora": PresetDefinition(
        id="builtin.aurora",
        name="Aurora",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#0f766e"),
                GradientStop(1.0, "#6d28d9"),
            ),
        ),
    ),
    "builtin.slate-gradient": PresetDefinition(
        id="builtin.slate-gradient",
        name="Slate gradient",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="vertical",
            stops=(
                GradientStop(0.0, "#f1f5f9"),
                GradientStop(1.0, "#cbd5e1"),
            ),
        ),
    ),
    "builtin.catppuccin-gradient": PresetDefinition(
        id="builtin.catppuccin-gradient",
        name="Catppuccin gradient",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="vertical",
            stops=(
                GradientStop(0.0, "#1e1e2e"),
                GradientStop(1.0, "#313244"),
            ),
        ),
    ),
    "builtin.catppuccin-latte-gradient": PresetDefinition(
        id="builtin.catppuccin-latte-gradient",
        name="Catppuccin Latte gradient",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="vertical",
            stops=(
                GradientStop(0.0, "#eff1f5"),
                GradientStop(1.0, "#e6e9ef"),
            ),
        ),
    ),
    "builtin.nebula": PresetDefinition(
        id="builtin.nebula",
        name="Nebula",
        kind="image",
        wallpaper=WallpaperImage(source="assets/wallpapers/nebula.jpg"),
        asset_relpath="assets/wallpapers/nebula.jpg",
    ),
    "builtin.forest": PresetDefinition(
        id="builtin.forest",
        name="Forest",
        kind="image",
        wallpaper=WallpaperImage(source="assets/wallpapers/forest.jpg"),
        asset_relpath="assets/wallpapers/forest.jpg",
    ),
    "builtin.ocean": PresetDefinition(
        id="builtin.ocean",
        name="Ocean",
        kind="image",
        wallpaper=WallpaperImage(source="assets/wallpapers/ocean.jpg"),
        asset_relpath="assets/wallpapers/ocean.jpg",
    ),
}

# Family + polarity → default preset id for theme_default resolution (§14).
_FAMILY_DEFAULT_PRESET_DARK: dict[str, str] = {
    "catppuccin": "builtin.nebula",
    "nord": "builtin.mist",
    "gruvbox": "builtin.paper",
    "dracula": "builtin.aurora",
    "github": "builtin.slate-gradient",
    "slate": "builtin.slate-gradient",
    "solarized": "builtin.mist",
}

_FAMILY_DEFAULT_PRESET_LIGHT: dict[str, str] = {
    "catppuccin": "builtin.catppuccin-latte-gradient",
    "nord": "builtin.slate-gradient",
    "gruvbox": "builtin.paper",
    "dracula": "builtin.slate-gradient",
    "github": "builtin.slate-gradient",
    "slate": "builtin.slate-gradient",
    "solarized": "builtin.paper",
}

_FALLBACK_DEFAULT_PRESET_DARK = "builtin.mist"
_FALLBACK_DEFAULT_PRESET_LIGHT = "builtin.slate-gradient"


def list_preset_ids() -> list[str]:
    return sorted(_BUILTIN_PRESETS.keys())


def get_preset(preset_id: str) -> PresetDefinition | None:
    return _BUILTIN_PRESETS.get(preset_id)


def preset_exists(preset_id: str) -> bool:
    return preset_id in _BUILTIN_PRESETS


def preset_wallpaper(preset_id: str) -> Wallpaper:
    definition = get_preset(preset_id)
    if definition is None:
        raise KeyError(f"Unknown wallpaper preset: {preset_id!r}")
    return definition.wallpaper


def preset_asset_path(preset_id: str) -> Path | None:
    definition = get_preset(preset_id)
    if definition is None or not definition.asset_relpath:
        return None
    return resource_path(definition.asset_relpath)


def theme_default_preset_id(*, family: str, base_mode: str) -> str:
    """Resolve scheme family and polarity to a bundled preset id."""
    key = (family or "").strip().lower()
    mode = (base_mode or "dark").strip().lower()
    if mode == "light":
        return _FAMILY_DEFAULT_PRESET_LIGHT.get(key, _FALLBACK_DEFAULT_PRESET_LIGHT)
    return _FAMILY_DEFAULT_PRESET_DARK.get(key, _FALLBACK_DEFAULT_PRESET_DARK)


def resolve_preset_reference(wallpaper: WallpaperPreset) -> Wallpaper:
    """Expand a preset reference to its concrete wallpaper definition."""
    return preset_wallpaper(wallpaper.preset_id)


def all_presets() -> Mapping[str, PresetDefinition]:
    return dict(_BUILTIN_PRESETS)
