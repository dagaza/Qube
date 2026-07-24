"""Surface profile validation."""

from __future__ import annotations

from core.paths import resource_path
from core.surface_fill.constants import V2_SURFACES
from core.surface_fill.image_paths import resolve_wallpaper_image_path
from core.surface_fill.models import (
    SurfaceProfile,
    ValidatedSurfaceProfile,
    Wallpaper,
    WallpaperGradient,
    WallpaperImage,
    WallpaperNone,
    WallpaperPreset,
    WallpaperSolid,
    WallpaperThemeDefault,
)
from core.surface_fill.presets import preset_exists
from core.theme.color_utils import parse_color


class SurfaceFillValidator:
    MAX_IMAGE_BYTES = 15 * 1024 * 1024
    MAX_IMAGE_DIMENSION = 8192

    def validate_profile(
        self,
        surface_id: str,
        profile: SurfaceProfile,
        *,
        resolved_wallpaper: Wallpaper | None = None,
    ) -> ValidatedSurfaceProfile:
        if surface_id not in V2_SURFACES:
            return ValidatedSurfaceProfile(
                surface_id=surface_id,
                profile=profile,
                errors=(f"Unknown surface: {surface_id!r}",),
            )
        wallpaper = resolved_wallpaper or profile.wallpaper
        errors: list[str] = []
        warnings: list[str] = []
        self._validate_wallpaper(wallpaper, errors=errors, warnings=warnings)
        if (
            profile.overlay.strength == "vivid"
            and isinstance(wallpaper, WallpaperImage)
        ):
            warnings.append(
                "Vivid overlay with an image wallpaper may reduce readability"
            )
        return ValidatedSurfaceProfile(
            surface_id=surface_id,
            profile=profile,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    def _validate_wallpaper(
        self,
        wallpaper: Wallpaper,
        *,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if isinstance(wallpaper, (WallpaperNone, WallpaperThemeDefault)):
            return
        if isinstance(wallpaper, WallpaperPreset):
            if not preset_exists(wallpaper.preset_id):
                errors.append(f"Unknown wallpaper preset: {wallpaper.preset_id!r}")
            return
        if isinstance(wallpaper, WallpaperSolid):
            try:
                parse_color(wallpaper.color)
            except ValueError as exc:
                errors.append(str(exc))
            return
        if isinstance(wallpaper, WallpaperGradient):
            if len(wallpaper.stops) != 2:
                errors.append("Gradient wallpaper requires exactly 2 stops")
            for stop in wallpaper.stops:
                try:
                    parse_color(stop.color)
                except ValueError as exc:
                    errors.append(str(exc))
                if not 0.0 <= stop.position <= 1.0:
                    errors.append(f"Gradient stop position out of range: {stop.position}")
            return
        if isinstance(wallpaper, WallpaperImage):
            self._validate_image_path(wallpaper.source, errors=errors, warnings=warnings)

    def _validate_image_path(
        self,
        source: str,
        *,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        path = resolve_wallpaper_image_path(source)
        if path is None:
            errors.append(f"Image path is not allowed: {source!r}")
            return
        if not path.is_file():
            if source.startswith("assets/wallpapers/") or str(path).startswith(
                str(resource_path("assets", "wallpapers"))
            ):
                warnings.append(f"Bundled wallpaper asset not yet installed: {source!r}")
                return
            errors.append(f"Image file not found: {source!r}")
            return
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"Cannot read image file: {exc}")
            return
        if size > self.MAX_IMAGE_BYTES:
            warnings.append(
                f"Image is larger than {self.MAX_IMAGE_BYTES // (1024 * 1024)} MB; "
                "consider using a smaller file"
            )
