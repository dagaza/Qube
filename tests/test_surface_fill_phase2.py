"""Phase 2 tests — Settings wallpaper UI and bundled assets."""

from __future__ import annotations

from pathlib import Path

from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
from core.surface_fill.import_wallpaper import import_wallpaper_image
from core.surface_fill.models import (
    SurfaceProfile,
    WallpaperImage,
    WallpaperNone,
    WallpaperPreset,
    WallpaperThemeDefault,
)
from core.surface_fill.presets import preset_asset_path
from core.surface_fill.thumbnails import list_picker_preset_ids, preset_thumbnail_pixmap


def test_bundled_wallpaper_assets_exist():
    for preset_id in ("builtin.nebula", "builtin.forest", "builtin.ocean"):
        path = preset_asset_path(preset_id)
        assert path is not None
        assert path.is_file()


def test_preset_thumbnail_pixmap_non_empty(_qube_app):
    pixmap = preset_thumbnail_pixmap("builtin.mist", size=48, is_dark=True)
    assert not pixmap.isNull()
    assert pixmap.width() == 48


def test_picker_lists_all_builtin_presets():
    ids = list_picker_preset_ids()
    assert "builtin.nebula" in ids
    assert "builtin.catppuccin-gradient" in ids


def test_import_wallpaper_copies_to_user_dir(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: tmp_path,
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(source, format="PNG")

    result = import_wallpaper_image(source)
    assert (tmp_path / result.filename).is_file()


def test_list_user_wallpaper_filenames(tmp_path, monkeypatch):
    import os
    from PIL import Image

    from core.surface_fill.import_wallpaper import list_user_wallpaper_filenames

    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: tmp_path,
    )
    first = tmp_path / "alpha.jpg"
    second = tmp_path / "beta.png"
    Image.new("RGB", (8, 8), color=(10, 10, 10)).save(first, format="JPEG")
    Image.new("RGB", (8, 8), color=(20, 20, 20)).save(second, format="PNG")
    os.utime(first, (1_000, 1_000))
    os.utime(second, (2_000, 2_000))

    assert list_user_wallpaper_filenames() == ["beta.png", "alpha.jpg"]


def test_wallpaper_editor_lists_imported_images(_qube_app, tmp_path, monkeypatch):
    from PIL import Image

    from core.surface_fill.import_wallpaper import import_wallpaper_image
    from ui.components.wallpaper_picker import WallpaperEditorWidget

    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "core.surface_fill.thumbnails.wallpapers_directory",
        lambda: tmp_path,
    )
    source = tmp_path / "mountain.png"
    Image.new("RGB", (48, 48), color=(30, 90, 150)).save(source, format="PNG")
    result = import_wallpaper_image(source)

    editor = WallpaperEditorWidget("Chat wallpaper")
    editor.show()
    editor._sync_mode_stack("image")

    assert result.filename in editor._image_tiles
    editor._on_image_tile_activated(result.filename)
    wallpaper = editor.profile().wallpaper
    assert isinstance(wallpaper, WallpaperImage)
    assert wallpaper.source == result.filename


def test_wallpaper_options_container_hides_for_none_mode(_qube_app):
    from ui.components.wallpaper_picker import WallpaperEditorWidget

    editor = WallpaperEditorWidget("Chat wallpaper")
    editor.show()
    editor.set_profile(
        SurfaceProfile(wallpaper=WallpaperNone()),
        block_signals=True,
    )
    assert editor._options_container.isHidden()

    editor.set_profile(
        SurfaceProfile(wallpaper=WallpaperPreset(preset_id="builtin.mist")),
        block_signals=True,
    )
    assert not editor._options_container.isHidden()
    assert not editor._mode_panels["preset"].isHidden()
    assert editor._mode_panels["solid"].isHidden()


def test_wallpaper_preset_tile_selects_profile(_qube_app):
    from core.surface_fill.models import WallpaperPreset
    from ui.components.wallpaper_picker import WallpaperEditorWidget

    editor = WallpaperEditorWidget("Chat wallpaper")
    editor._on_preset_tile_activated("builtin.aurora")

    wallpaper = editor.profile().wallpaper
    assert isinstance(wallpaper, WallpaperPreset)
    assert wallpaper.preset_id == "builtin.aurora"
    assert editor._mode_cbs["preset"].isChecked()


def test_settings_themes_wallpapers_section_builds(fresh_main_window):
    from core.surface_fill.models import default_surface_profile_set

    manager = fresh_main_window.theme_manager
    manager._surface_profiles_active = default_surface_profile_set()
    manager._surface_profiles_draft = None

    settings = fresh_main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    settings._sync_themes_draft_from_applied()

    assert hasattr(settings, "themes_chat_wallpaper")
    assert hasattr(settings, "themes_library_wallpaper")
    chat_wallpaper = (
        settings._draft_surface_profiles()
        .for_surface(SURFACE_CHAT_TRANSCRIPT)
        .wallpaper
    )
    assert chat_wallpaper.kind == "theme_default"
    assert isinstance(settings.themes_chat_wallpaper.profile().wallpaper, WallpaperThemeDefault)


def test_settings_wallpaper_draft_dirty_until_apply(fresh_main_window):
    settings = fresh_main_window.ensure_settings_view()
    settings.select_settings_section("appearance.themes")
    manager = fresh_main_window.theme_manager

    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperNone()),
    )

    assert settings._themes_surface_profiles_dirty()
    assert settings._themes_draft_is_dirty()

    manager._surface_profiles_draft = settings._draft_surface_profiles()
    manager.apply_surface_profiles(persist=False)
    settings._themes_draft_surface_profiles = settings._applied_surface_profiles()

    assert not settings._themes_surface_profiles_dirty()
    assert (
        manager.surface_profiles_active.for_surface(SURFACE_CHAT_TRANSCRIPT).wallpaper.kind
        == "none"
    )


def test_settings_wallpaper_revert_discards_draft(fresh_main_window):
    from core.surface_fill.models import default_surface_profile_set

    settings = fresh_main_window.ensure_settings_view()
    manager = fresh_main_window.theme_manager
    manager._surface_profiles_active = default_surface_profile_set()
    settings.select_settings_section("appearance.themes")
    settings._ensure_themes_preview_initialized()

    settings._set_draft_surface_profile(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperPreset(preset_id="builtin.aurora")),
    )
    assert settings._themes_surface_profiles_dirty()

    settings._on_themes_revert_clicked()
    assert not settings._themes_surface_profiles_dirty()
    assert isinstance(
        settings._draft_surface_profiles().for_surface(SURFACE_CHAT_TRANSCRIPT).wallpaper,
        WallpaperThemeDefault,
    )
