"""Tests for theme pack import/export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
from core.surface_fill.models import (
    SurfaceProfile,
    SurfaceProfileSet,
    WallpaperImage,
    WallpaperPreset,
    default_surface_profile_set,
)
from core.theme.io import SCHEMA_VERSION
from core.theme.manager import ThemeManager
from core.theme.pack_io import (
    PACK_MANIFEST_NAME,
    PACK_SCHEMA_VERSION,
    export_theme_pack_to_path,
    read_theme_pack_from_path,
    rewrite_surface_profiles_for_export,
)
from core.theme.schemes import DEFAULT_SCHEME_ID_DARK
from core.theme.storage import ThemeStorage


def _make_test_png(path: Path, *, color: tuple[int, int, int] = (30, 60, 90)) -> None:
    Image.new("RGB", (24, 24), color=color).save(path, format="PNG")


def _scheme_payload() -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "id": "user.exported-pack",
        "name": "Exported Pack",
        "base_mode": "dark",
        "algorithm": "default",
        "extends": DEFAULT_SCHEME_ID_DARK,
        "overrides": {"accent": "#89b4fa"},
    }


@pytest.fixture
def wallpapers_dir(tmp_path, monkeypatch):
    directory = tmp_path / "wallpapers"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "core.surface_fill.storage.wallpapers_directory",
        lambda: directory,
    )
    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: directory,
    )
    monkeypatch.setattr(
        "core.theme.pack_io.user_data_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "core.surface_fill.image_paths.user_data_root",
        lambda: tmp_path,
    )
    return directory


def test_export_collects_user_wallpaper_assets(wallpapers_dir):
    image_path = wallpapers_dir / "chat-bg.png"
    _make_test_png(image_path)

    profiles = default_surface_profile_set().with_surface(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperImage(source="chat-bg.png")),
    )
    pack_path = wallpapers_dir / "out.qube-theme.zip"
    export_theme_pack_to_path(
        pack_path,
        scheme=_scheme_payload(),
        surface_profiles=profiles,
    )

    with zipfile.ZipFile(pack_path) as archive:
        names = archive.namelist()
        assert PACK_MANIFEST_NAME in names
        assert "assets/wallpapers/chat-bg.png" in names
        manifest = json.loads(archive.read(PACK_MANIFEST_NAME))
        assert manifest["pack_schema"] == PACK_SCHEMA_VERSION
        assert manifest["assets"] == ["chat-bg.png"]
        chat = manifest["surface_profiles"][SURFACE_CHAT_TRANSCRIPT]["wallpaper"]
        assert chat["kind"] == "image"
        assert chat["source"] == "chat-bg.png"


def test_export_skips_preset_wallpapers(wallpapers_dir):
    profiles = default_surface_profile_set().with_surface(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperPreset(preset_id="builtin.nebula")),
    )
    pack_path = wallpapers_dir / "preset-only.qube-theme.zip"
    export_theme_pack_to_path(
        pack_path,
        scheme=_scheme_payload(),
        surface_profiles=profiles,
    )
    with zipfile.ZipFile(pack_path) as archive:
        assert all(not name.startswith("assets/") for name in archive.namelist())


def test_import_theme_pack_registers_scheme_and_copies_assets(
    wallpapers_dir, tmp_path, monkeypatch
):
    source_image = wallpapers_dir / "library.png"
    _make_test_png(source_image, color=(120, 40, 40))

    profiles = default_surface_profile_set().with_surface(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperImage(source="library.png")),
    )
    pack_path = tmp_path / "roundtrip.qube-theme.zip"
    export_theme_pack_to_path(
        pack_path,
        scheme=_scheme_payload(),
        surface_profiles=profiles,
    )

    imported_dir = tmp_path / "imported-wallpapers"
    imported_dir.mkdir()
    user_root = tmp_path / "import-user"
    user_root.mkdir()

    from core.theme import pack_io as pack_io_module

    monkeypatch.setattr(pack_io_module, "user_data_root", lambda: user_root)
    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: imported_dir,
    )
    monkeypatch.setattr(
        "core.surface_fill.image_paths.user_data_root",
        lambda: user_root,
    )

    parsed = read_theme_pack_from_path(pack_path)
    assert parsed.assets_imported
    imported_name = parsed.assets_imported[0]
    assert (imported_dir / imported_name).is_file()
    chat = parsed.surface_profiles.for_surface(SURFACE_CHAT_TRANSCRIPT)
    assert isinstance(chat.wallpaper, WallpaperImage)
    assert chat.wallpaper.source == imported_name


def test_read_theme_pack_rejects_unsafe_paths(tmp_path):
    pack_path = tmp_path / "unsafe.qube-theme.zip"
    manifest = {
        "pack_schema": PACK_SCHEMA_VERSION,
        "scheme": _scheme_payload(),
        "surface_profiles": {},
        "assets": [],
    }
    with zipfile.ZipFile(pack_path, "w") as archive:
        archive.writestr(PACK_MANIFEST_NAME, json.dumps(manifest))
        archive.writestr("../escape.png", b"not-an-image")

    with pytest.raises(ValueError, match="Unsafe path"):
        read_theme_pack_from_path(pack_path)


def test_theme_manager_export_import_roundtrip(
    wallpapers_dir, tmp_path, monkeypatch, grant_pro_share_themes
):
    image_path = wallpapers_dir / "draft-bg.png"
    _make_test_png(image_path)

    themes_dir = tmp_path / "themes"
    themes_dir.mkdir()
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: themes_dir)

    storage = ThemeStorage(settings_get=lambda _k, d=None: d, settings_set=lambda _k, _v: None)
    manager = ThemeManager(storage=storage)
    profiles = default_surface_profile_set().with_surface(
        SURFACE_CHAT_TRANSCRIPT,
        SurfaceProfile(wallpaper=WallpaperImage(source="draft-bg.png")),
    )
    pack_path = tmp_path / "manager-pack.qube-theme.zip"
    manager.export_theme_pack_to_path(
        pack_path,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        surface_profiles=profiles,
    )

    import_dir = tmp_path / "manager-import-wallpapers"
    import_dir.mkdir()
    user_root = tmp_path / "manager-user"
    user_root.mkdir()

    from core.theme import pack_io as pack_io_module

    monkeypatch.setattr(pack_io_module, "user_data_root", lambda: user_root)
    monkeypatch.setattr(
        "core.surface_fill.import_wallpaper.wallpapers_directory",
        lambda: import_dir,
    )
    monkeypatch.setattr(
        "core.surface_fill.image_paths.user_data_root",
        lambda: user_root,
    )

    result = manager.import_theme_pack_from_path(pack_path)
    assert result.scheme_id.startswith("user.")
    assert result.assets_imported
    assert manager.get_scheme_definition(result.scheme_id).name


def test_rewrite_surface_profiles_for_export_maps_sources(wallpapers_dir):
    image_path = wallpapers_dir / "mapped.png"
    _make_test_png(image_path)
    assets = {"assets/wallpapers/mapped.png": image_path.resolve()}
    profiles = SurfaceProfileSet(
        profiles={
            SURFACE_CHAT_TRANSCRIPT: SurfaceProfile(
                wallpaper=WallpaperImage(source=str(image_path))
            )
        }
    )
    rewritten = rewrite_surface_profiles_for_export(profiles, assets=assets)
    wallpaper = rewritten.for_surface(SURFACE_CHAT_TRANSCRIPT).wallpaper
    assert isinstance(wallpaper, WallpaperImage)
    assert wallpaper.source == "mapped.png"
