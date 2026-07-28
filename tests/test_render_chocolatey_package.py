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


def test_render_writes_all_variant_packages(tmp_path, monkeypatch):
    mod = _load_render()
    templates = tmp_path / "chocolatey" / "templates"
    tools = templates / "tools"
    tools.mkdir(parents=True)
    (templates / "package.nuspec").write_text(
        "<id>{{PACKAGE_ID}}</id><version>{{VERSION}}</version>",
        encoding="utf-8",
    )
    (tools / "chocolateyinstall.ps1").write_text(
        "$url = '{{INSTALLER_URL}}'\n$checksum = '{{SHA256}}'\n",
        encoding="utf-8",
    )
    (tools / "chocolateyuninstall.ps1").write_text(
        "# uninstall {{VERSION}}",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    hash_value = "AB" * 32
    hashes = {"cpu": hash_value, "vulkan": hash_value, "cuda": hash_value}
    out = mod.render("1.2.3", hashes)

    for pkg in ("qube", "qube-vulkan", "qube-cuda"):
        package_dir = out / pkg
        assert (package_dir / f"{pkg}.nuspec").is_file()
        assert (package_dir / "tools" / "chocolateyinstall.ps1").is_file()

    install = (out / "qube-cuda" / "tools" / "chocolateyinstall.ps1").read_text(encoding="utf-8")
    assert "Qube-1.2.3-cuda-Setup.exe" in install
    assert hash_value in install

    nuspec = (out / "qube-vulkan" / "qube-vulkan.nuspec").read_text(encoding="utf-8")
    assert "<id>qube-vulkan</id>" in nuspec
