"""Surface fill data models — wallpapers, overlays, and surface profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from core.surface_fill.constants import (
    GradientDirection,
    OverlayStrength,
    SURFACE_CHAT_TRANSCRIPT,
    SURFACE_LIBRARY_PREVIEW,
    V2_SURFACES,
)

__all__ = [
    "GradientStop",
    "OverlaySpec",
    "SurfaceProfile",
    "SurfaceProfileSet",
    "ValidatedSurfaceProfile",
    "Wallpaper",
    "WallpaperGradient",
    "WallpaperImage",
    "WallpaperNone",
    "WallpaperPreset",
    "WallpaperSolid",
    "WallpaperThemeDefault",
    "default_surface_profile",
    "default_surface_profile_set",
]


@dataclass(frozen=True)
class WallpaperNone:
    kind: str = "none"


@dataclass(frozen=True)
class WallpaperThemeDefault:
    kind: str = "theme_default"


@dataclass(frozen=True)
class WallpaperPreset:
    preset_id: str
    kind: str = "preset"


@dataclass(frozen=True)
class WallpaperSolid:
    color: str
    kind: str = "solid"


@dataclass(frozen=True)
class GradientStop:
    position: float
    color: str


@dataclass(frozen=True)
class WallpaperGradient:
    direction: GradientDirection
    stops: tuple[GradientStop, ...]
    kind: str = "gradient"


@dataclass(frozen=True)
class WallpaperImage:
    source: str
    fit: str = "cover"
    kind: str = "image"


Wallpaper = (
    WallpaperNone
    | WallpaperThemeDefault
    | WallpaperPreset
    | WallpaperSolid
    | WallpaperGradient
    | WallpaperImage
)


@dataclass(frozen=True)
class OverlaySpec:
    strength: OverlayStrength = "balanced"


@dataclass(frozen=True)
class SurfaceProfile:
    wallpaper: Wallpaper
    overlay: OverlaySpec = OverlaySpec()


@dataclass(frozen=True)
class SurfaceProfileSet:
    profiles: Mapping[str, SurfaceProfile]

    def for_surface(self, surface_id: str) -> SurfaceProfile:
        profile = self.profiles.get(surface_id)
        if profile is not None:
            return profile
        return default_surface_profile()

    def with_surface(self, surface_id: str, profile: SurfaceProfile) -> SurfaceProfileSet:
        merged = dict(self.profiles)
        merged[surface_id] = profile
        return SurfaceProfileSet(profiles=merged)

    def without_surface(self, surface_id: str) -> SurfaceProfileSet:
        merged = {key: value for key, value in self.profiles.items() if key != surface_id}
        return SurfaceProfileSet(profiles=merged)


@dataclass(frozen=True)
class ValidatedSurfaceProfile:
    surface_id: str
    profile: SurfaceProfile
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def default_surface_profile() -> SurfaceProfile:
    return SurfaceProfile(wallpaper=WallpaperThemeDefault())


def default_surface_profile_set() -> SurfaceProfileSet:
    return SurfaceProfileSet(
        profiles={
            SURFACE_CHAT_TRANSCRIPT: default_surface_profile(),
            SURFACE_LIBRARY_PREVIEW: default_surface_profile(),
        }
    )
