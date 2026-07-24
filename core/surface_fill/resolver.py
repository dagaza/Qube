"""Resolve surface profiles and wallpaper references."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from core.surface_fill.constants import V2_SURFACES
from core.surface_fill.models import (
    SurfaceProfile,
    SurfaceProfileSet,
    Wallpaper,
    WallpaperNone,
    WallpaperPreset,
    WallpaperThemeDefault,
    default_surface_profile,
)
from core.surface_fill.presets import preset_wallpaper, theme_default_preset_id
from core.theme.definition import ColorSchemeDefinition
from core.theme.tokens import ThemeMode


class SurfaceFillResolver:
    """Merge stored profiles and resolve theme_default / preset references."""

    def resolve_profile_set(
        self,
        profile_set: SurfaceProfileSet,
        *,
        scheme: ColorSchemeDefinition,
        family: str,
        mode: ThemeMode,
    ) -> SurfaceProfileSet:
        resolved: dict[str, SurfaceProfile] = {}
        for surface_id in V2_SURFACES:
            profile = profile_set.for_surface(surface_id)
            resolved[surface_id] = self.resolve_profile(
                profile,
                surface_id=surface_id,
                scheme=scheme,
                family=family,
                mode=mode,
            )
        return SurfaceProfileSet(profiles=resolved)

    def resolve_profile(
        self,
        profile: SurfaceProfile,
        *,
        surface_id: str,
        scheme: ColorSchemeDefinition,
        family: str,
        mode: ThemeMode,
    ) -> SurfaceProfile:
        _ = surface_id, scheme
        wallpaper = self.resolve_wallpaper(
            profile.wallpaper,
            family=family,
            mode=mode,
        )
        if wallpaper is profile.wallpaper:
            return profile
        return replace(profile, wallpaper=wallpaper)

    def resolve_wallpaper(
        self,
        wallpaper: Wallpaper,
        *,
        family: str,
        mode: ThemeMode,
    ) -> Wallpaper:
        if isinstance(wallpaper, WallpaperNone):
            return wallpaper
        if isinstance(wallpaper, WallpaperThemeDefault):
            preset_id = theme_default_preset_id(
                family=family,
                base_mode=mode.value,
            )
            return preset_wallpaper(preset_id)
        if isinstance(wallpaper, WallpaperPreset):
            return preset_wallpaper(wallpaper.preset_id)
        return wallpaper

    def effective_profile(
        self,
        profile_set: SurfaceProfileSet,
        surface_id: str,
        *,
        schemes: Mapping[str, ColorSchemeDefinition],
        scheme_id: str,
        mode: ThemeMode,
    ) -> SurfaceProfile:
        from core.theme.catalog import resolve_scheme_family

        scheme = schemes.get(scheme_id)
        if scheme is None:
            return profile_set.for_surface(surface_id)
        family = resolve_scheme_family(scheme, schemes)
        return self.resolve_profile(
            profile_set.for_surface(surface_id),
            surface_id=surface_id,
            scheme=scheme,
            family=family,
            mode=mode,
        )


def merge_surface_profile_sets(
    base: SurfaceProfileSet,
    overrides: SurfaceProfileSet | None,
) -> SurfaceProfileSet:
    if overrides is None or not overrides.profiles:
        return base
    merged = dict(base.profiles)
    merged.update(overrides.profiles)
    return SurfaceProfileSet(profiles=merged)
