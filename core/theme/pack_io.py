"""Theme pack import/export — color scheme, surface profiles, and wallpaper assets."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.paths import user_data_root
from core.surface_fill.image_paths import resolve_wallpaper_image_path
from core.surface_fill.import_wallpaper import USER_WALLPAPER_EXTENSIONS, import_wallpaper_image
from core.surface_fill.models import (
    SurfaceProfile,
    SurfaceProfileSet,
    Wallpaper,
    WallpaperImage,
    default_surface_profile_set,
)
from core.surface_fill.serialization import (
    surface_profile_set_from_dict,
    surface_profile_set_to_dict,
)
from core.theme.io import import_color_scheme

PACK_SCHEMA_VERSION = 1
PACK_MANIFEST_NAME = "pack.json"
PACK_ASSET_PREFIX = "assets/wallpapers/"
MAX_PACK_BYTES = 50 * 1024 * 1024
MAX_PACK_FILES = 64


@dataclass(frozen=True)
class ThemePackImportResult:
    """Parsed theme pack before scheme registration."""

    scheme: dict[str, Any]
    surface_profiles: SurfaceProfileSet
    assets_imported: tuple[str, ...]


@dataclass(frozen=True)
class ThemePackAppliedResult:
    """Theme pack after scheme registration."""

    scheme_id: str
    surface_profiles: SurfaceProfileSet
    assets_imported: tuple[str, ...]


def _normalize_zip_name(name: str) -> str:
    return name.replace("\\", "/")


def _is_safe_zip_member(name: str) -> bool:
    normalized = _normalize_zip_name(name)
    if not normalized or normalized.startswith("/"):
        return False
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


def _user_wallpapers_root() -> Path:
    return (user_data_root() / "wallpapers").resolve()


def _is_user_wallpaper_file(path: Path) -> bool:
    try:
        path.resolve().relative_to(_user_wallpapers_root())
    except ValueError:
        return False
    return path.is_file()


def _pack_asset_name(filename: str) -> str:
    cleaned = Path(str(filename or "").strip()).name
    if not cleaned:
        raise ValueError("Wallpaper asset filename is required")
    suffix = Path(cleaned).suffix.lower()
    if suffix not in USER_WALLPAPER_EXTENSIONS:
        raise ValueError(f"Unsupported wallpaper asset type: {suffix!r}")
    return f"{PACK_ASSET_PREFIX}{cleaned}"


def _collect_user_wallpaper_assets(
    profile_set: SurfaceProfileSet,
) -> dict[str, Path]:
    """Map pack asset path -> source file on disk (deduped by source path)."""
    assets: dict[str, Path] = {}
    seen_sources: dict[Path, str] = {}
    for profile in profile_set.profiles.values():
        wallpaper = profile.wallpaper
        if not isinstance(wallpaper, WallpaperImage):
            continue
        source_path = resolve_wallpaper_image_path(wallpaper.source)
        if source_path is None or not _is_user_wallpaper_file(source_path):
            continue
        resolved = source_path.resolve()
        if resolved in seen_sources:
            continue
        pack_path = _pack_asset_name(resolved.name)
        if pack_path in assets and assets[pack_path] != resolved:
            stem = resolved.stem
            suffix = resolved.suffix
            counter = 2
            while pack_path in assets:
                pack_path = _pack_asset_name(f"{stem}-{counter}{suffix}")
                counter += 1
        seen_sources[resolved] = pack_path
        assets[pack_path] = resolved
    return assets


def _rewrite_image_source_for_export(
    wallpaper: Wallpaper,
    *,
    source_to_pack_path: dict[str, str],
) -> Wallpaper:
    if not isinstance(wallpaper, WallpaperImage):
        return wallpaper
    source_path = resolve_wallpaper_image_path(wallpaper.source)
    if source_path is None or not _is_user_wallpaper_file(source_path):
        return wallpaper
    pack_path = source_to_pack_path.get(str(source_path.resolve()))
    if not pack_path:
        return wallpaper
    pack_relative = pack_path.removeprefix(PACK_ASSET_PREFIX)
    return WallpaperImage(source=pack_relative, fit=wallpaper.fit)


def rewrite_surface_profiles_for_export(
    profile_set: SurfaceProfileSet,
    *,
    assets: Mapping[str, Path],
) -> SurfaceProfileSet:
    source_to_pack_path = {str(path.resolve()): pack_path for pack_path, path in assets.items()}
    profiles: dict[str, SurfaceProfile] = {}
    for surface_id, profile in profile_set.profiles.items():
        wallpaper = _rewrite_image_source_for_export(
            profile.wallpaper,
            source_to_pack_path=source_to_pack_path,
        )
        profiles[surface_id] = SurfaceProfile(
            wallpaper=wallpaper,
            overlay=profile.overlay,
        )
    return SurfaceProfileSet(profiles=profiles)


def build_theme_pack_manifest(
    *,
    scheme: Mapping[str, Any],
    surface_profiles: SurfaceProfileSet,
    assets: Mapping[str, Path],
) -> dict[str, Any]:
    asset_entries = sorted(
        path.removeprefix(PACK_ASSET_PREFIX)
        for path in assets.keys()
    )
    return {
        "pack_schema": PACK_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "scheme": dict(scheme),
        "surface_profiles": surface_profile_set_to_dict(surface_profiles),
        "assets": asset_entries,
    }


def export_theme_pack_to_path(
    path: Path,
    *,
    scheme: Mapping[str, Any],
    surface_profiles: SurfaceProfileSet,
) -> None:
    """Write a zip theme pack with manifest and bundled user wallpaper assets."""
    assets = _collect_user_wallpaper_assets(surface_profiles)
    export_profiles = rewrite_surface_profiles_for_export(surface_profiles, assets=assets)
    manifest = build_theme_pack_manifest(
        scheme=scheme,
        surface_profiles=export_profiles,
        assets=assets,
    )
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            PACK_MANIFEST_NAME,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        for pack_path, source_path in sorted(assets.items()):
            archive.write(source_path, arcname=pack_path)


def _validate_pack_manifest(payload: Mapping[str, Any]) -> None:
    schema = payload.get("pack_schema")
    if schema != PACK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported theme pack schema: {schema!r} (expected {PACK_SCHEMA_VERSION})"
        )
    scheme = payload.get("scheme")
    if not isinstance(scheme, dict):
        raise ValueError("Theme pack requires a scheme object")
    if scheme.get("schema") is None:
        raise ValueError("Theme pack scheme is missing schema version")
    profiles = payload.get("surface_profiles")
    if profiles is not None and not isinstance(profiles, dict):
        raise ValueError("Theme pack surface_profiles must be an object")
    assets = payload.get("assets")
    if assets is not None and not isinstance(assets, list):
        raise ValueError("Theme pack assets must be an array")


def _import_pack_assets(
    archive: zipfile.ZipFile,
    *,
    asset_names: list[str],
) -> dict[str, str]:
    """Copy pack assets into the user wallpapers dir; return pack-relative -> filename."""
    remapped: dict[str, str] = {}
    for raw_name in asset_names:
        relative = str(raw_name or "").strip().replace("\\", "/")
        if not relative:
            continue
        if ".." in relative.split("/"):
            raise ValueError(f"Invalid asset path in theme pack: {relative!r}")
        pack_member = _pack_asset_name(Path(relative).name)
        if pack_member not in archive.namelist():
            pack_member = f"{PACK_ASSET_PREFIX}{Path(relative).name}"
        if pack_member not in archive.namelist():
            raise ValueError(f"Theme pack is missing asset: {relative!r}")
        with archive.open(pack_member) as src:
            temp_dir = user_data_root() / "tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / Path(pack_member).name
            temp_path.write_bytes(src.read())
        try:
            result = import_wallpaper_image(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        remapped[relative] = result.filename
        remapped[Path(relative).name] = result.filename
    return remapped


def _rewrite_image_source_for_import(
    wallpaper: Wallpaper,
    *,
    asset_remap: Mapping[str, str],
) -> Wallpaper:
    if not isinstance(wallpaper, WallpaperImage):
        return wallpaper
    source = str(wallpaper.source or "").strip()
    if not source:
        return wallpaper
    filename = asset_remap.get(source) or asset_remap.get(Path(source).name)
    if filename:
        return WallpaperImage(source=filename, fit=wallpaper.fit)
    return wallpaper


def rewrite_surface_profiles_for_import(
    profile_set: SurfaceProfileSet,
    *,
    asset_remap: Mapping[str, str],
) -> SurfaceProfileSet:
    profiles: dict[str, SurfaceProfile] = {}
    for surface_id, profile in profile_set.profiles.items():
        wallpaper = _rewrite_image_source_for_import(
            profile.wallpaper,
            asset_remap=asset_remap,
        )
        profiles[surface_id] = SurfaceProfile(
            wallpaper=wallpaper,
            overlay=profile.overlay,
        )
    return SurfaceProfileSet(profiles=profiles)


def read_theme_pack_from_path(path: Path) -> ThemePackImportResult:
    """Parse a theme pack zip without registering the color scheme."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Theme pack not found: {source}")
    size = source.stat().st_size
    if size > MAX_PACK_BYTES:
        raise ValueError(
            f"Theme pack is too large ({size} bytes; max {MAX_PACK_BYTES})"
        )

    with zipfile.ZipFile(source, mode="r") as archive:
        if len(archive.infolist()) > MAX_PACK_FILES:
            raise ValueError(
                f"Theme pack contains too many files (max {MAX_PACK_FILES})"
            )
        for info in archive.infolist():
            if not _is_safe_zip_member(info.filename):
                raise ValueError(f"Unsafe path in theme pack: {info.filename!r}")
            if info.file_size > MAX_PACK_BYTES:
                raise ValueError(f"Theme pack file is too large: {info.filename!r}")

        if PACK_MANIFEST_NAME not in archive.namelist():
            raise ValueError(f"Theme pack is missing {PACK_MANIFEST_NAME}")

        manifest_raw = json.loads(archive.read(PACK_MANIFEST_NAME).decode("utf-8"))
        if not isinstance(manifest_raw, dict):
            raise ValueError("Theme pack manifest must be a JSON object")
        _validate_pack_manifest(manifest_raw)

        scheme = dict(manifest_raw["scheme"])
        profiles = surface_profile_set_from_dict(manifest_raw.get("surface_profiles"))
        asset_names = [
            str(item).strip()
            for item in (manifest_raw.get("assets") or [])
            if str(item).strip()
        ]
        asset_remap = _import_pack_assets(archive, asset_names=asset_names)
        profiles = rewrite_surface_profiles_for_import(profiles, asset_remap=asset_remap)

        return ThemePackImportResult(
            scheme=scheme,
            surface_profiles=profiles,
            assets_imported=tuple(sorted(set(asset_remap.values()))),
        )


def validate_theme_pack_scheme(
    scheme: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> None:
    """Validate scheme payload without persisting."""
    import_color_scheme(dict(scheme), registry=registry)


def default_pack_surface_profiles() -> SurfaceProfileSet:
    return default_surface_profile_set()
