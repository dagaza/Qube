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
    # Brand & marketing gradients (docs/private/WEBSITE_THEME_DESIGN_SPEC.md §2, §12.5–§12.7).
    "builtin.qube-signature": PresetDefinition(
        id="builtin.qube-signature",
        name="Qube signature",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="horizontal",
            stops=(
                GradientStop(0.0, "#3b82f6"),
                GradientStop(1.0, "#8b5cf6"),
            ),
        ),
    ),
    "builtin.indigo-trail": PresetDefinition(
        id="builtin.indigo-trail",
        name="Indigo trail",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="horizontal",
            stops=(
                GradientStop(0.0, "#3b82f6"),
                GradientStop(0.55, "#8b5cf6"),
                GradientStop(1.0, "#6366f1"),
            ),
        ),
    ),
    "builtin.prism-drift": PresetDefinition(
        id="builtin.prism-drift",
        name="Prism drift",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#3b82f6"),
                GradientStop(0.5, "#8b5cf6"),
                GradientStop(1.0, "#6366f1"),
            ),
        ),
    ),
    "builtin.conversation-glow": PresetDefinition(
        id="builtin.conversation-glow",
        name="Conversation glow",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#122649"),
                GradientStop(0.5, "#191432"),
                GradientStop(1.0, "#05070f"),
            ),
        ),
    ),
    "builtin.library-shimmer": PresetDefinition(
        id="builtin.library-shimmer",
        name="Library shimmer",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#052a36"),
                GradientStop(0.5, "#0d1932"),
                GradientStop(1.0, "#05070f"),
            ),
        ),
    ),
    "builtin.studio-hues": PresetDefinition(
        id="builtin.studio-hues",
        name="Studio hues",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#261c49"),
                GradientStop(0.5, "#281124"),
                GradientStop(1.0, "#0a1326"),
            ),
        ),
    ),
    "builtin.morning-glow": PresetDefinition(
        id="builtin.morning-glow",
        name="Morning glow",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#c9dcfa"),
                GradientStop(0.5, "#e8e2fb"),
                GradientStop(1.0, "#f8fafc"),
            ),
        ),
    ),
    "builtin.daybreak-tide": PresetDefinition(
        id="builtin.daybreak-tide",
        name="Daybreak tide",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#c8ecf4"),
                GradientStop(0.5, "#dce8fb"),
                GradientStop(1.0, "#f8fafc"),
            ),
        ),
    ),
    "builtin.midnight-ember": PresetDefinition(
        id="builtin.midnight-ember",
        name="Midnight ember",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#0c0a09"),
                GradientStop(0.45, "#431407"),
                GradientStop(1.0, "#c2410c"),
            ),
        ),
    ),
    "builtin.cyber-mint": PresetDefinition(
        id="builtin.cyber-mint",
        name="Cyber mint",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_up",
            stops=(
                GradientStop(0.0, "#042f2e"),
                GradientStop(0.5, "#0f766e"),
                GradientStop(1.0, "#115e59"),
            ),
        ),
    ),
    "builtin.obsidian-rose": PresetDefinition(
        id="builtin.obsidian-rose",
        name="Obsidian rose",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#09090b"),
                GradientStop(0.5, "#701a75"),
                GradientStop(1.0, "#be185d"),
            ),
        ),
    ),
    "builtin.northern-fire": PresetDefinition(
        id="builtin.northern-fire",
        name="Northern fire",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_up",
            stops=(
                GradientStop(0.0, "#0f172a"),
                GradientStop(0.55, "#0e7490"),
                GradientStop(1.0, "#b45309"),
            ),
        ),
    ),
    "builtin.emerald-dusk": PresetDefinition(
        id="builtin.emerald-dusk",
        name="Emerald dusk",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="vertical",
            stops=(
                GradientStop(0.0, "#052e16"),
                GradientStop(0.5, "#14532d"),
                GradientStop(1.0, "#1e1b4b"),
            ),
        ),
    ),
    "builtin.velvet-wine": PresetDefinition(
        id="builtin.velvet-wine",
        name="Velvet wine",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="horizontal",
            stops=(
                GradientStop(0.0, "#1c1917"),
                GradientStop(0.5, "#581c87"),
                GradientStop(1.0, "#be123c"),
            ),
        ),
    ),
    "builtin.solar-flare": PresetDefinition(
        id="builtin.solar-flare",
        name="Solar flare",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#450a0a"),
                GradientStop(0.5, "#a16207"),
                GradientStop(1.0, "#ea580c"),
            ),
        ),
    ),
    "builtin.peach-sorbet": PresetDefinition(
        id="builtin.peach-sorbet",
        name="Peach sorbet",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_down",
            stops=(
                GradientStop(0.0, "#fff7ed"),
                GradientStop(0.5, "#fdba74"),
                GradientStop(1.0, "#fbcfe8"),
            ),
        ),
    ),
    "builtin.sea-glass": PresetDefinition(
        id="builtin.sea-glass",
        name="Sea glass",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="diagonal_up",
            stops=(
                GradientStop(0.0, "#ecfdf5"),
                GradientStop(0.5, "#99f6e4"),
                GradientStop(1.0, "#7dd3fc"),
            ),
        ),
    ),
    "builtin.lilac-haze": PresetDefinition(
        id="builtin.lilac-haze",
        name="Lilac haze",
        kind="gradient",
        wallpaper=WallpaperGradient(
            direction="horizontal",
            stops=(
                GradientStop(0.0, "#faf5ff"),
                GradientStop(0.5, "#c4b5fd"),
                GradientStop(1.0, "#f5d0fe"),
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
