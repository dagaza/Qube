"""Surface fills — wallpapers and overlays for named UI surfaces."""

from core.surface_fill.constants import (
    GRADIENT_DIRECTIONS,
    OVERLAY_STRENGTHS,
    SURFACE_CHAT_TRANSCRIPT,
    SURFACE_LIBRARY_PREVIEW,
    V2_SURFACES,
)
from core.surface_fill.models import (
    GradientStop,
    OverlaySpec,
    SurfaceProfile,
    SurfaceProfileSet,
    ValidatedSurfaceProfile,
    Wallpaper,
    WallpaperGradient,
    WallpaperImage,
    WallpaperNone,
    WallpaperPreset,
    WallpaperSolid,
    WallpaperThemeDefault,
    default_surface_profile,
    default_surface_profile_set,
)
from core.surface_fill.compositor import ComposedWallpaper, SurfaceFillCompositor
from core.surface_fill.image_paths import resolve_wallpaper_image_path
from core.surface_fill.overlay import (
    overlay_render_params,
    overlay_scrim_rgba,
    overlay_strength_with_boost,
)
from core.surface_fill.renderer import SurfaceFillRenderer
from core.surface_fill.presets import (
    get_preset,
    list_preset_ids,
    preset_exists,
    preset_wallpaper,
    theme_default_preset_id,
)
from core.surface_fill.resolver import SurfaceFillResolver, merge_surface_profile_sets
from core.surface_fill.serialization import (
    surface_profile_from_dict,
    surface_profile_set_from_dict,
    surface_profile_set_from_json,
    surface_profile_set_to_dict,
    surface_profile_set_to_json,
    surface_profile_to_dict,
    wallpaper_from_dict,
    wallpaper_to_dict,
)
from core.surface_fill.storage import (
    KEY_SURFACE_PROFILES_ACTIVE,
    KEY_SURFACE_PROFILES_DRAFT,
    SurfaceFillStorage,
    surface_fill_storage_from_app_settings,
    wallpapers_directory,
)
from core.surface_fill.validation import SurfaceFillValidator

__all__ = [
    "GRADIENT_DIRECTIONS",
    "KEY_SURFACE_PROFILES_ACTIVE",
    "KEY_SURFACE_PROFILES_DRAFT",
    "OVERLAY_STRENGTHS",
    "SURFACE_CHAT_TRANSCRIPT",
    "SURFACE_LIBRARY_PREVIEW",
    "SurfaceFillResolver",
    "SurfaceFillStorage",
    "SurfaceFillValidator",
    "SurfaceProfile",
    "SurfaceProfileSet",
    "ValidatedSurfaceProfile",
    "V2_SURFACES",
    "Wallpaper",
    "WallpaperGradient",
    "WallpaperImage",
    "WallpaperNone",
    "WallpaperPreset",
    "WallpaperSolid",
    "WallpaperThemeDefault",
    "default_surface_profile",
    "default_surface_profile_set",
    "get_preset",
    "list_preset_ids",
    "merge_surface_profile_sets",
    "overlay_render_params",
    "overlay_scrim_rgba",
    "overlay_strength_with_boost",
    "preset_exists",
    "preset_wallpaper",
    "surface_fill_storage_from_app_settings",
    "surface_profile_from_dict",
    "surface_profile_set_from_dict",
    "surface_profile_set_from_json",
    "surface_profile_set_to_dict",
    "surface_profile_set_to_json",
    "surface_profile_to_dict",
    "theme_default_preset_id",
    "wallpaper_from_dict",
    "wallpaper_to_dict",
    "wallpapers_directory",
]
