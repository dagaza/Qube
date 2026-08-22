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


def test_render_writes_split_manifests_for_all_variants(tmp_path, monkeypatch):
    mod = _load_render()
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    out = mod.render(
        "1.2.5",
        {
            "cpu": "AA" * 32,
            "vulkan": "BB" * 32,
            "cuda": "CC" * 32,
        },
    )
    assert out.is_dir()
    for package_id, exe_fragment, hash_prefix in (
        ("dagaza.Qube", "Qube-1.2.5-Setup.exe", "AA"),
        ("dagaza.Qube.Vulkan", "Qube-1.2.5-vulkan-Setup.exe", "BB"),
        ("dagaza.Qube.CUDA", "Qube-1.2.5-cuda-Setup.exe", "CC"),
    ):
        package_dir = out / package_id
        assert package_dir.is_dir()
        assert (package_dir / f"{package_id}.yaml").is_file()
        assert (package_dir / f"{package_id}.installer.yaml").is_file()
        assert (package_dir / f"{package_id}.locale.en-US.yaml").is_file()
        installer = (package_dir / f"{package_id}.installer.yaml").read_text(encoding="utf-8")
        assert f"InstallerSha256: {hash_prefix}" in installer
        assert exe_fragment in installer
    vulkan_installer = (out / "dagaza.Qube.Vulkan" / "dagaza.Qube.Vulkan.installer.yaml").read_text(
        encoding="utf-8"
    )
    assert "KhronosGroup.VulkanRT" in vulkan_installer
