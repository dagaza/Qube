"""Model Manager publisher logo path resolution (dev vs PyInstaller frozen)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from ui.views.model_manager_view import (
    _default_hub_fallback_logo,
    _resolve_bundled_asset_url,
    _resolve_hub_brand_logo,
)


def test_resolve_hub_brand_logo_dev_assets_path():
    resolved = _resolve_hub_brand_logo("/assets/logos/mistral.svg")
    assert resolved is not None
    assert resolved.is_file()
    assert resolved.name == "mistral.svg"


def test_resolve_hub_brand_logo_absolute_path(tmp_path: Path):
    logo = tmp_path / "cached-avatar.png"
    logo.write_bytes(b"x")
    assert _resolve_hub_brand_logo(str(logo)) == logo


def test_resolve_hub_brand_logo_empty_returns_none():
    assert _resolve_hub_brand_logo("") is None
    assert _resolve_bundled_asset_url("   ") is None


def test_resolve_hub_brand_logo_frozen_uses_meipass(tmp_path: Path):
    internal = tmp_path / "_internal" / "assets" / "logos"
    internal.mkdir(parents=True)
    mistral = internal / "mistral.svg"
    mistral.write_text("<svg/>", encoding="utf-8")
    fake_exe = tmp_path / "Qube.exe"
    fake_exe.touch()
    with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
        sys, "executable", str(fake_exe)
    ), mock.patch.object(sys, "_MEIPASS", str(tmp_path / "_internal"), create=True):
        wrong = tmp_path / "assets" / "logos" / "mistral.svg"
        assert not wrong.is_file()
        resolved = _resolve_hub_brand_logo("/assets/logos/mistral.svg")
        assert resolved == mistral
        assert resolved.is_file()


def test_default_hub_fallback_logo_resolves_in_dev():
    assert _default_hub_fallback_logo() is not None
