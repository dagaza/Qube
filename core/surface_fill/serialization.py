"""JSON serialization for surface profiles."""

from __future__ import annotations

from typing import Any, Mapping

from core.surface_fill.constants import GRADIENT_DIRECTIONS, OVERLAY_STRENGTHS, V2_SURFACES
from core.surface_fill.models import (
    GradientStop,
    OverlaySpec,
    SurfaceProfile,
    SurfaceProfileSet,
    Wallpaper,
    WallpaperGradient,
    WallpaperImage,
    WallpaperNone,
    WallpaperPreset,
    WallpaperSolid,
    WallpaperThemeDefault,
    default_surface_profile,
)


def wallpaper_to_dict(wallpaper: Wallpaper) -> dict[str, Any]:
    if isinstance(wallpaper, WallpaperNone):
        return {"kind": "none"}
    if isinstance(wallpaper, WallpaperThemeDefault):
        return {"kind": "theme_default"}
    if isinstance(wallpaper, WallpaperPreset):
        return {"kind": "preset", "preset_id": wallpaper.preset_id}
    if isinstance(wallpaper, WallpaperSolid):
        return {"kind": "solid", "color": wallpaper.color}
    if isinstance(wallpaper, WallpaperGradient):
        return {
            "kind": "gradient",
            "direction": wallpaper.direction,
            "stops": [
                {"position": stop.position, "color": stop.color}
                for stop in wallpaper.stops
            ],
        }
    if isinstance(wallpaper, WallpaperImage):
        payload: dict[str, Any] = {"kind": "image", "source": wallpaper.source}
        if wallpaper.fit != "cover":
            payload["fit"] = wallpaper.fit
        return payload
    raise TypeError(f"Unsupported wallpaper type: {type(wallpaper)!r}")


def wallpaper_from_dict(raw: Mapping[str, Any]) -> Wallpaper:
    kind = str(raw.get("kind") or "").strip().lower()
    if kind == "none":
        return WallpaperNone()
    if kind == "theme_default":
        return WallpaperThemeDefault()
    if kind == "preset":
        preset_id = str(raw.get("preset_id") or "").strip()
        if not preset_id:
            raise ValueError("preset wallpaper requires preset_id")
        return WallpaperPreset(preset_id=preset_id)
    if kind == "solid":
        color = str(raw.get("color") or "").strip()
        if not color:
            raise ValueError("solid wallpaper requires color")
        return WallpaperSolid(color=color)
    if kind == "gradient":
        direction = str(raw.get("direction") or "").strip().lower()
        if direction not in GRADIENT_DIRECTIONS:
            raise ValueError(f"Invalid gradient direction: {direction!r}")
        stops_raw = raw.get("stops")
        if not isinstance(stops_raw, list) or len(stops_raw) != 2:
            raise ValueError("gradient wallpaper requires exactly 2 stops")
        stops: list[GradientStop] = []
        for item in stops_raw:
            if not isinstance(item, Mapping):
                raise ValueError("gradient stop must be an object")
            position = float(item.get("position", 0))
            color = str(item.get("color") or "").strip()
            if not color:
                raise ValueError("gradient stop requires color")
            stops.append(GradientStop(position=position, color=color))
        return WallpaperGradient(direction=direction, stops=(stops[0], stops[1]))  # type: ignore[arg-type]
    if kind == "image":
        source = str(raw.get("source") or "").strip()
        if not source:
            raise ValueError("image wallpaper requires source")
        fit = str(raw.get("fit") or "cover").strip().lower()
        if fit != "cover":
            raise ValueError(f"Unsupported image fit in v2: {fit!r}")
        return WallpaperImage(source=source, fit=fit)
    raise ValueError(f"Unknown wallpaper kind: {kind!r}")


def overlay_to_dict(overlay: OverlaySpec) -> dict[str, Any]:
    return {"strength": overlay.strength}


def overlay_from_dict(raw: Mapping[str, Any] | None) -> OverlaySpec:
    if not raw:
        return OverlaySpec()
    strength = str(raw.get("strength") or "balanced").strip().lower()
    if strength not in OVERLAY_STRENGTHS:
        raise ValueError(f"Invalid overlay strength: {strength!r}")
    return OverlaySpec(strength=strength)  # type: ignore[arg-type]


def surface_profile_to_dict(profile: SurfaceProfile) -> dict[str, Any]:
    return {
        "wallpaper": wallpaper_to_dict(profile.wallpaper),
        "overlay": overlay_to_dict(profile.overlay),
    }


def surface_profile_from_dict(raw: Mapping[str, Any]) -> SurfaceProfile:
    wallpaper_raw = raw.get("wallpaper")
    if not isinstance(wallpaper_raw, Mapping):
        raise ValueError("surface profile requires wallpaper object")
    overlay_raw = raw.get("overlay")
    overlay = (
        overlay_from_dict(overlay_raw)
        if isinstance(overlay_raw, Mapping)
        else OverlaySpec()
    )
    return SurfaceProfile(
        wallpaper=wallpaper_from_dict(wallpaper_raw),
        overlay=overlay,
    )


def surface_profile_set_to_dict(profile_set: SurfaceProfileSet) -> dict[str, Any]:
    return {
        surface_id: surface_profile_to_dict(profile)
        for surface_id, profile in profile_set.profiles.items()
        if surface_id in V2_SURFACES
    }


def surface_profile_set_from_dict(raw: Mapping[str, Any] | None) -> SurfaceProfileSet:
    if not raw:
        return SurfaceProfileSet(profiles={})
    profiles: dict[str, SurfaceProfile] = {}
    for surface_id, profile_raw in raw.items():
        if surface_id not in V2_SURFACES:
            continue
        if not isinstance(profile_raw, Mapping):
            raise ValueError(f"Invalid profile for surface {surface_id!r}")
        profiles[surface_id] = surface_profile_from_dict(profile_raw)
    return SurfaceProfileSet(profiles=profiles)


def surface_profile_set_to_json(profile_set: SurfaceProfileSet) -> str:
    import json

    return json.dumps(surface_profile_set_to_dict(profile_set), sort_keys=True)


def surface_profile_set_from_json(raw: str) -> SurfaceProfileSet:
    import json

    if not raw or not str(raw).strip():
        return SurfaceProfileSet(profiles={})
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("surface profiles JSON must be an object")
    return surface_profile_set_from_dict(payload)
