"""Tests for core/gpu_driver_preload.py."""

from __future__ import annotations

import builtins
import io

from core import gpu_driver_preload as mod


def _fake_maps(monkeypatch, paths: list[str]) -> None:
    lines = "".join(
        f"7f00-7f01 r-xp 00000000 08:01 1 {p}\n" for p in paths
    )
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if file == "/proc/self/maps":
            return io.StringIO(lines)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(mod.sys, "platform", "linux")


def test_shadowing_detected_when_only_bundled_copy_is_mapped(monkeypatch) -> None:
    torch_lib = "/venv/site-packages/torch/lib/libdrm_amdgpu.so"
    _fake_maps(monkeypatch, [torch_lib])
    assert mod.shadowing_libdrm_amdgpu_path() == torch_lib


def test_no_shadowing_when_system_copy_is_mapped(monkeypatch) -> None:
    _fake_maps(
        monkeypatch,
        [
            "/venv/site-packages/torch/lib/libdrm_amdgpu.so",
            "/usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1.134.0",
        ],
    )
    assert mod.shadowing_libdrm_amdgpu_path() is None


def test_shadowing_libvulkan_detects_bundled_loader(monkeypatch) -> None:
    _fake_maps(
        monkeypatch,
        [
            "/opt/qube/_internal/libvulkan.so.1",
            "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so",
        ],
    )

    def fake_system_path(soname: str):
        if soname == "libvulkan.so.1":
            from pathlib import Path

            return Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1.3.275.0")
        return None

    monkeypatch.setattr(mod, "_system_lib_path", fake_system_path)
    assert (
        mod.shadowing_libvulkan_loader_path()
        == "/opt/qube/_internal/libvulkan.so.1"
    )


def test_radv_icd_mapped_when_radeon_driver_present(monkeypatch) -> None:
    _fake_maps(
        monkeypatch,
        ["/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so"],
    )
    assert mod.radv_icd_mapped() is True


def test_preload_is_noop_off_linux(monkeypatch) -> None:
    mod.reset_preload_state_for_tests()
    monkeypatch.setattr(mod.sys, "platform", "win32")
    assert mod.preload_system_gpu_driver_libs() == []
    mod.reset_preload_state_for_tests()
