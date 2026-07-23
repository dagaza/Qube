"""Tests for scripts/render_linux_packages.py."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_linux_packages.py"
    spec = importlib.util.spec_from_file_location("render_linux_packages", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_deb_tree_writes_expected_layout(tmp_path, monkeypatch):
    mod = _load_module()
    dist = tmp_path / "dist" / "Qube"
    dist.mkdir(parents=True)
    (dist / "Qube").write_text("#!/bin/sh\n", encoding="utf-8")
    mode = (dist / "Qube").stat().st_mode | stat.S_IXUSR
    (dist / "Qube").chmod(mode)

    packaging = tmp_path / "packaging" / "linux"
    packaging.mkdir(parents=True)
    (packaging / "qube.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (packaging / "qube-uninstall.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (packaging / "qube.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    (tmp_path / "assets" / "logos").mkdir(parents=True)
    (tmp_path / "assets" / "logos" / "qube_logo_256.png").write_bytes(b"png")

    monkeypatch.setattr(mod, "_REPO", tmp_path)
    monkeypatch.setattr(mod, "repo_root", lambda: tmp_path)

    staging = tmp_path / "staging"
    mod.stage_deb_tree(staging)

    assert (staging / "opt" / "qube" / "Qube").is_file()
    assert (staging / "opt" / "qube" / "uninstall" / "uninstall.sh").is_file()
    assert (staging / "usr" / "bin" / "qube").is_file()
    assert (staging / "usr" / "bin" / "qube-uninstall").is_file()
    assert (staging / "usr" / "share" / "applications" / "qube.desktop").is_file()
    assert (staging / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / "qube.png").is_file()


def test_stage_appdir_writes_apprun_and_bundle(tmp_path, monkeypatch):
    mod = _load_module()
    dist = tmp_path / "dist" / "Qube"
    dist.mkdir(parents=True)
    (dist / "Qube").write_text("#!/bin/sh\n", encoding="utf-8")

    packaging = tmp_path / "packaging" / "linux"
    packaging.mkdir(parents=True)
    (packaging / "AppRun").write_text("#!/bin/sh\n", encoding="utf-8")
    (packaging / "qube.appimage.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    (tmp_path / "assets" / "logos").mkdir(parents=True)
    (tmp_path / "assets" / "logos" / "qube_logo_256.png").write_bytes(b"png")

    monkeypatch.setattr(mod, "_REPO", tmp_path)
    monkeypatch.setattr(mod, "repo_root", lambda: tmp_path)

    appdir = tmp_path / "Qube.AppDir"
    mod.stage_appdir(appdir)

    assert (appdir / "AppRun").exists()
    assert (appdir / "qube.desktop").exists()
    assert (appdir / "usr" / "bin" / "Qube" / "Qube").is_file()
