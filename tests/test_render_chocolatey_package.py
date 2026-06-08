"""Tests for scripts/render_chocolatey_package.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_render():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_chocolatey_package.py"
    spec = importlib.util.spec_from_file_location("render_chocolatey_package", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_writes_package_files(tmp_path, monkeypatch):
    mod = _load_render()
    templates = tmp_path / "chocolatey" / "templates"
    tools = templates / "tools"
    tools.mkdir(parents=True)
    (templates / "qube.nuspec").write_text(
        "<version>{{VERSION}}</version>",
        encoding="utf-8",
    )
    (tools / "chocolateyinstall.ps1").write_text(
        "$checksum = '{{SHA256}}'\n$version = '{{VERSION}}'",
        encoding="utf-8",
    )
    (tools / "chocolateyuninstall.ps1").write_text(
        "# uninstall {{VERSION}}",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    out = mod.render("1.2.3", "AB" * 32)
    assert out.is_dir()
    assert (out / "qube.nuspec").is_file()
    assert (out / "tools" / "chocolateyinstall.ps1").is_file()
    assert (out / "tools" / "chocolateyuninstall.ps1").is_file()

    nuspec = (out / "qube.nuspec").read_text(encoding="utf-8")
    assert "<version>1.2.3</version>" in nuspec

    install = (out / "tools" / "chocolateyinstall.ps1").read_text(encoding="utf-8")
    assert "AB" * 32 in install
    assert "1.2.3" in install
    assert "Qube-1.2.3-Setup.exe" not in install  # URL built in template, not render
