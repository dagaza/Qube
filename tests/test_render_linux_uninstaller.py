"""Tests for scripts/render_linux_uninstaller.py."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path


def _load_render():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_linux_uninstaller.py"
    spec = importlib.util.spec_from_file_location("render_linux_uninstaller", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _patch_user_data_root(monkeypatch, path: Path) -> None:
    monkeypatch.setattr("core.uninstall_paths.user_data_root", lambda: path)


def test_render_uninstall_script_includes_manifest_paths(monkeypatch, tmp_path):
    mod = _load_render()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    data_root = tmp_path / ".qube"
    _patch_user_data_root(monkeypatch, data_root)

    script = mod.render_uninstall_script(version="9.9.9")
    assert 'remove_path "/opt/qube"' in script
    assert 'remove_path "$HOME/.qube"' in script
    assert 'remove_path "/usr/share/applications/qube.desktop"' in script
    assert "detect_deb_package" in script
    assert "remove_deb_package" in script


def test_write_uninstall_script_is_executable(tmp_path, monkeypatch):
    mod = _load_render()
    monkeypatch.setattr(mod, "_repo_root", lambda: Path(__file__).resolve().parents[1])

    target = tmp_path / "uninstall.sh"
    mod.write_uninstall_script(target, version="1.0.0")
    assert target.is_file()
    if os.name != "nt":
        assert target.stat().st_mode & stat.S_IXUSR
