"""Tests for core/linux_vulkan_bundle_fixup.py."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

from core import linux_vulkan_bundle_fixup as mod


def test_disable_conflicting_bundled_libs_renames_vulkan_and_drm(
    tmp_path: Path,
) -> None:
    internal = tmp_path / "_internal"
    av = internal / "av.libs"
    av.mkdir(parents=True)
    loader = internal / "libvulkan.so.1"
    loader.write_bytes(b"vulkan")
    drm = av / "libdrm.so.2.4.0"
    drm.write_bytes(b"drm")

    moved = mod.disable_conflicting_bundled_libs(tmp_path)

    assert not loader.is_file()
    assert (internal / "libvulkan.so.1.bundled.bak").is_file()
    assert not drm.is_file()
    assert (av / "libdrm.so.2.4.0.bundled.bak").is_file()
    assert len(moved) == 2


def test_apply_fixup_skips_non_vulkan_variant(
    tmp_path: Path, monkeypatch
) -> None:
    exe = tmp_path / "Qube"
    exe.write_text("", encoding="utf-8")
    (tmp_path / ".qube_linux_variant").write_text("cpu\n", encoding="utf-8")
    internal = tmp_path / "_internal"
    internal.mkdir()
    (internal / "libvulkan.so.1").write_bytes(b"x")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    monkeypatch.setattr(mod.sys, "platform", "linux")

    assert mod.apply_vulkan_bundle_fixup_if_needed() == []
    assert (internal / "libvulkan.so.1").is_file()


def test_apply_fixup_runs_for_vulkan_when_writable(
    tmp_path: Path, monkeypatch
) -> None:
    exe = tmp_path / "Qube"
    exe.write_text("", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IWUSR)
    (tmp_path / ".qube_linux_variant").write_text("vulkan\n", encoding="utf-8")
    internal = tmp_path / "_internal"
    internal.mkdir()
    (internal / "libvulkan.so.1").write_bytes(b"x")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    monkeypatch.setattr(mod.sys, "platform", "linux")

    disabled = mod.apply_vulkan_bundle_fixup_if_needed()
    assert disabled
    assert not (internal / "libvulkan.so.1").is_file()