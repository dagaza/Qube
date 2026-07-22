"""Tests for scripts/render_macos_uninstaller.py."""

from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path


def _load_render():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_macos_uninstaller.py"
    spec = importlib.util.spec_from_file_location("render_macos_uninstaller", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_executable(path: Path) -> None:
    assert path.is_file()
    if os.name != "nt":
        assert path.stat().st_mode & stat.S_IXUSR


def test_render_uninstall_script_includes_manifest_paths(monkeypatch, tmp_path):
    mod = _load_render()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    script = mod.render_uninstall_script(version="9.9.9")
    assert 'remove_path "/Applications/Qube.app"' in script
    assert f'remove_path "{(tmp_path / ".qube").as_posix()}"' in script
    assert "com.dagaza.Qube.plist" in script


def test_build_uninstaller_app_writes_bundle(tmp_path, monkeypatch):
    mod = _load_render()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()

    packaging = tmp_path / "packaging" / "macos" / "uninstaller"
    packaging.mkdir(parents=True)
    (packaging / "Info.plist.tmpl").write_text(
        "<plist><key>CFBundleShortVersionString</key><string>{{VERSION}}</string></plist>",
        encoding="utf-8",
    )
    (packaging / "uninstall-launcher.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (packaging / "uninstall.sh.tmpl").write_text(
        "APP\n{{APP_REMOVE_LINES}}\nDATA\n{{DATA_REMOVE_LINES}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    app_path = mod.build_uninstaller_app(version="1.2.3", output_dir=tmp_path / "dist")
    assert app_path.name == "Uninstall Qube.app"
    assert (app_path / "Contents" / "Info.plist").read_text(encoding="utf-8").count("1.2.3") == 1
    script = (app_path / "Contents" / "Resources" / "uninstall.sh").read_text(encoding="utf-8")
    assert "APP" in script
    assert "DATA" in script
    launcher = app_path / "Contents" / "MacOS" / "uninstall"
    assert launcher.exists()
    _assert_executable(launcher)


def test_embed_uninstall_script_in_app(tmp_path, monkeypatch):
    mod = _load_render()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()

    app_path = tmp_path / "Qube.app" / "Contents" / "MacOS"
    app_path.mkdir(parents=True)
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    packaging = tmp_path / "packaging" / "macos" / "uninstaller"
    packaging.mkdir(parents=True)
    (packaging / "uninstall.sh.tmpl").write_text(
        "APP\n{{APP_REMOVE_LINES}}\nDATA\n{{DATA_REMOVE_LINES}}\n",
        encoding="utf-8",
    )

    target = mod.embed_uninstall_script_in_app(tmp_path / "Qube.app", version="3.0.0")
    assert target.is_file()
    assert "APP" in target.read_text(encoding="utf-8")
    _assert_executable(target)
