"""Tests for scripts/render_winget_manifests.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_render():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_winget_manifests.py"
    spec = importlib.util.spec_from_file_location("render_winget_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_writes_split_manifests(tmp_path, monkeypatch):
    mod = _load_render()
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    out = mod.render("1.2.3", "AB" * 32)
    assert out.is_dir()
    assert (out / "dagaza.Qube.yaml").is_file()
    assert (out / "dagaza.Qube.installer.yaml").is_file()
    assert (out / "dagaza.Qube.locale.en-US.yaml").is_file()
    installer = (out / "dagaza.Qube.installer.yaml").read_text(encoding="utf-8")
    assert "InstallerSha256: AB" in installer
    assert "Qube-1.2.3-Setup.exe" in installer
