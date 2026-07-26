"""Phase 0 tests for core.surface_fill."""

from __future__ import annotations

import json

import pytest

from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
from core.surface_fill.models import (
    GradientStop,
    OverlaySpec,
    SurfaceProfile,
    SurfaceProfileSet,
    WallpaperGradient,
    WallpaperImage,
    WallpaperNone,
    WallpaperPreset,
    WallpaperSolid,
    WallpaperThemeDefault,
)
from core.surface_fill.overlay import overlay_scrim_rgba, overlay_strength_with_boost
from core.surface_fill.presets import (
    preset_exists,
    preset_wallpaper,
    theme_default_preset_id,
)
from core.surface_fill.resolver import SurfaceFillResolver
from core.surface_fill.serialization import (
    surface_profile_set_from_json,
    surface_profile_set_to_dict,
    wallpaper_from_dict,
    wallpaper_to_dict,
)
from core.surface_fill.storage import (
    KEY_SURFACE_PROFILES_ACTIVE,
    KEY_SURFACE_PROFILES_DRAFT,
    SurfaceFillStorage,
)
from core.surface_fill.validation import SurfaceFillValidator
from core.theme.manager import ThemeManager
from core.theme.schemes import BUILTIN_SCHEMES, DEFAULT_SCHEME_ID_DARK
from core.theme.storage import ThemeStorage
from core.theme.tokens import ThemeMode


class _MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def get(self, key: str, default: object) -> object:
        return self._data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._data[key] = value


class _NoopApplicator:
    def apply(self, resolved, profiler=None) -> None:
        return None


def test_wallpaper_json_round_trip():
    gradient = WallpaperGradient(
        direction="diagonal_down",
        stops=(
            GradientStop(0.0, "#1e1e2e"),
            GradientStop(1.0, "#313244"),
        ),
    )
    payload = wallpaper_to_dict(gradient)
    restored = wallpaper_from_dict(payload)
    assert restored == gradient


def test_surface_profile_set_json_round_trip():
    profile_set = SurfaceProfileSet(
        profiles={
            SURFACE_CHAT_TRANSCRIPT: SurfaceProfile(
                wallpaper=WallpaperPreset(preset_id="builtin.mist"),
                overlay=OverlaySpec(strength="subtle"),
            ),
        }
    )
    raw = json.dumps(surface_profile_set_to_dict(profile_set))
    restored = surface_profile_set_from_json(raw)
    assert restored.profiles[SURFACE_CHAT_TRANSCRIPT].overlay.strength == "subtle"
    assert isinstance(
        restored.profiles[SURFACE_CHAT_TRANSCRIPT].wallpaper,
        WallpaperPreset,
    )


def test_theme_default_resolves_to_family_preset():
    resolver = SurfaceFillResolver()
    scheme = BUILTIN_SCHEMES[DEFAULT_SCHEME_ID_DARK]
    profile = SurfaceProfile(wallpaper=WallpaperThemeDefault())
    resolved = resolver.resolve_profile(
        profile,
        surface_id=SURFACE_CHAT_TRANSCRIPT,
        scheme=scheme,
        family="catppuccin",
        mode=ThemeMode.DARK,
    )
    assert isinstance(resolved.wallpaper, WallpaperGradient)
    assert resolved.wallpaper.stops[0].color == "#1e1e2e"


def test_theme_default_preset_id_for_nord():
    assert theme_default_preset_id(family="nord", base_mode="dark") == "builtin.mist"


def test_preset_wallpaper_expands():
    wallpaper = preset_wallpaper("builtin.aurora")
    assert isinstance(wallpaper, WallpaperGradient)
    assert wallpaper.direction == "diagonal_down"


def test_validator_rejects_unknown_preset():
    validator = SurfaceFillValidator()
    result = validator.validate_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperPreset(preset_id="builtin.unknown")),
    )
    assert not result.ok
    assert any("Unknown wallpaper preset" in err for err in result.errors)


def test_validator_accepts_gradient():
    validator = SurfaceFillValidator()
    profile = SurfaceProfile(
        wallpaper=WallpaperGradient(
            direction="vertical",
            stops=(
                GradientStop(0.0, "#111111"),
                GradientStop(1.0, "#222222"),
            ),
        )
    )
    result = validator.validate_profile(SURFACE_CHAT_TRANSCRIPT, profile)
    assert result.ok


def test_validator_blocks_path_traversal():
    validator = SurfaceFillValidator()
    profile = SurfaceProfile(
        wallpaper=WallpaperImage(source="../../../etc/passwd"),
    )
    result = validator.validate_profile(SURFACE_CHAT_TRANSCRIPT, profile)
    assert not result.ok


def test_validator_accepts_bundled_asset_when_present():
    validator = SurfaceFillValidator()
    profile = SurfaceProfile(
        wallpaper=WallpaperImage(source="assets/wallpapers/nebula.jpg"),
    )
    result = validator.validate_profile(SURFACE_CHAT_TRANSCRIPT, profile)
    assert result.ok
    assert not any("not yet installed" in warn for warn in result.warnings)


def test_validator_warns_on_missing_bundled_asset(monkeypatch, tmp_path):
    from pathlib import Path

    missing = tmp_path / "bundled-missing.jpg"
    validator = SurfaceFillValidator()
    profile = SurfaceProfile(
        wallpaper=WallpaperImage(source="assets/wallpapers/missing.jpg"),
    )
    monkeypatch.setattr(
        "core.surface_fill.validation.resolve_wallpaper_image_path",
        lambda _source: Path(missing),
    )
    result = validator.validate_profile(SURFACE_CHAT_TRANSCRIPT, profile)
    assert result.ok
    assert any("not yet installed" in warn for warn in result.warnings)


def test_overlay_scrim_uses_theme_not_stored_rgba():
    from core.theme.accessors import theme_for

    theme_dark = theme_for(is_dark=True)
    theme_light = theme_for(is_dark=False)
    overlay = OverlaySpec(strength="balanced")
    dark_scrim = overlay_scrim_rgba(overlay, theme_dark)
    light_scrim = overlay_scrim_rgba(overlay, theme_light)
    assert dark_scrim != light_scrim


def test_overlay_strength_boost_for_reader_focus():
    assert overlay_strength_with_boost("vivid", boost=1) == "balanced"
    assert overlay_strength_with_boost("balanced", boost=1) == "subtle"
    assert overlay_strength_with_boost("subtle", boost=1) == "subtle"


def test_overlay_strength_changes_rendered_output(_qube_app):
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QPainter, QPixmap

    from core.surface_fill.renderer import SurfaceFillRenderer
    from core.theme.accessors import theme_for

    renderer = SurfaceFillRenderer()
    theme = theme_for(is_dark=True)
    rect = QRect(0, 0, 48, 48)
    wallpaper = WallpaperGradient(
        direction="horizontal",
        stops=(
            GradientStop(0.0, "#ff0000"),
            GradientStop(1.0, "#0000ff"),
        ),
    )

    def _render(strength: str) -> int:
        profile = SurfaceProfile(
            wallpaper=wallpaper,
            overlay=OverlaySpec(strength=strength),  # type: ignore[arg-type]
        )
        pixmap = QPixmap(48, 48)
        pixmap.fill()
        painter = QPainter(pixmap)
        renderer.paint(painter, rect, profile, theme=theme)
        painter.end()
        return pixmap.toImage().pixel(24, 24)

    assert _render("subtle") != _render("balanced") != _render("vivid")


def test_overlay_skipped_for_wallpaper_none(_qube_app):
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QPainter, QPixmap

    from core.surface_fill.renderer import SurfaceFillRenderer
    from core.theme.accessors import theme_for

    renderer = SurfaceFillRenderer()
    theme = theme_for(is_dark=True)
    rect = QRect(0, 0, 32, 32)

    none_profile = SurfaceProfile(
        wallpaper=WallpaperNone(),
        overlay=OverlaySpec(strength="vivid"),
    )
    solid_profile = SurfaceProfile(
        wallpaper=WallpaperSolid(color="#224466"),
        overlay=OverlaySpec(strength="vivid"),
    )

    def _center_pixel(profile: SurfaceProfile) -> int:
        pixmap = QPixmap(32, 32)
        pixmap.fill()
        painter = QPainter(pixmap)
        renderer.paint(painter, rect, profile, theme=theme)
        painter.end()
        return pixmap.toImage().pixel(16, 16)

    assert _center_pixel(none_profile) != _center_pixel(solid_profile)


def test_surface_fill_storage_persists_active_and_draft():
    store = _MemoryStore()
    storage = SurfaceFillStorage(settings_get=store.get, settings_set=store.set)
    active = SurfaceProfileSet(
        profiles={
            SURFACE_CHAT_TRANSCRIPT: SurfaceProfile(wallpaper=WallpaperNone()),
        }
    )
    draft = active.with_surface(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperSolid(color="#112233")),
    )
    storage.save_active(active, persist=True)
    storage.save_draft(draft, persist=True)
    assert store.get(KEY_SURFACE_PROFILES_ACTIVE, "")
    assert store.get(KEY_SURFACE_PROFILES_DRAFT, "")
    reloaded_active, reloaded_draft = storage.load()
    assert isinstance(
        reloaded_active.for_surface(SURFACE_CHAT_TRANSCRIPT).wallpaper,
        WallpaperNone,
    )
    assert isinstance(
        reloaded_draft.for_surface(SURFACE_CHAT_TRANSCRIPT).wallpaper,
        WallpaperSolid,
    )


def test_theme_manager_surface_profile_draft_and_apply():
    theme_store = _MemoryStore()
    surface_store = _MemoryStore()
    manager = ThemeManager(
        storage=ThemeStorage(settings_get=theme_store.get, settings_set=theme_store.set),
        surface_storage=SurfaceFillStorage(
            settings_get=surface_store.get,
            settings_set=surface_store.set,
        ),
        applicator=_NoopApplicator(),  # type: ignore[arg-type]
    )
    profile = SurfaceProfile(
        wallpaper=WallpaperSolid(color="#abcdef"),
        overlay=OverlaySpec(strength="vivid"),
    )
    manager.set_surface_profile_draft(SURFACE_CHAT_TRANSCRIPT, profile, persist=True)
    assert isinstance(
        manager.surface_profile(SURFACE_CHAT_TRANSCRIPT).wallpaper,
        WallpaperSolid,
    )
    manager.apply_surface_profiles(persist=True)
    assert manager.surface_profiles_draft is None
    assert isinstance(
        manager.surface_profiles_active.for_surface(SURFACE_CHAT_TRANSCRIPT).wallpaper,
        WallpaperSolid,
    )


def test_theme_manager_resolved_effective_surface_profile():
    theme_store = _MemoryStore()
    surface_store = _MemoryStore()
    manager = ThemeManager(
        storage=ThemeStorage(settings_get=theme_store.get, settings_set=theme_store.set),
        surface_storage=SurfaceFillStorage(
            settings_get=surface_store.get,
            settings_set=surface_store.set,
        ),
        applicator=_NoopApplicator(),  # type: ignore[arg-type]
    )
    manager.apply(scheme_id="builtin.nord", persist=False)
    manager.set_surface_profile_draft(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperThemeDefault()),
        persist=False,
    )
    effective = manager.resolved_effective_surface_profile(SURFACE_CHAT_TRANSCRIPT)
    assert isinstance(effective.wallpaper, WallpaperGradient)


def test_theme_manager_surface_refresh_callback_on_apply():
    theme_store = _MemoryStore()
    surface_store = _MemoryStore()
    manager = ThemeManager(
        storage=ThemeStorage(settings_get=theme_store.get, settings_set=theme_store.set),
        surface_storage=SurfaceFillStorage(
            settings_get=surface_store.get,
            settings_set=surface_store.set,
        ),
        applicator=_NoopApplicator(),  # type: ignore[arg-type]
    )
    calls: list[str] = []

    def _refresh() -> None:
        calls.append("surface")

    manager.register_surface_refresh(_refresh)
    manager.apply(persist=False)
    assert calls == ["surface"]


def test_preset_catalog_includes_documented_ids():
    for preset_id in (
        "builtin.paper",
        "builtin.mist",
        "builtin.aurora",
        "builtin.nebula",
        "builtin.forest",
        "builtin.ocean",
        "builtin.slate-gradient",
        "builtin.catppuccin-gradient",
    ):
        assert preset_exists(preset_id)


def test_gradient_rejects_invalid_stop_counts():
    with pytest.raises(ValueError, match="2–5 stops"):
        wallpaper_from_dict(
            {
                "kind": "gradient",
                "direction": "vertical",
                "stops": [{"position": 0, "color": "#111111"}],
            }
        )
    with pytest.raises(ValueError, match="2–5 stops"):
        wallpaper_from_dict(
            {
                "kind": "gradient",
                "direction": "vertical",
                "stops": [
                    {"position": 0.0, "color": "#111111"},
                    {"position": 0.2, "color": "#222222"},
                    {"position": 0.4, "color": "#333333"},
                    {"position": 0.6, "color": "#444444"},
                    {"position": 0.8, "color": "#555555"},
                    {"position": 1.0, "color": "#666666"},
                ],
            }
        )


def test_gradient_three_stop_roundtrip():
    gradient = WallpaperGradient(
        direction="vertical",
        stops=(
            GradientStop(0.0, "#111111"),
            GradientStop(0.5, "#abcdef"),
            GradientStop(1.0, "#222222"),
        ),
    )
    payload = wallpaper_to_dict(gradient)
    restored = wallpaper_from_dict(payload)
    assert restored == gradient
