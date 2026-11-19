"""PyInstaller runtime hook: pin distro libdrm/Vulkan before bundled copies load."""

from __future__ import annotations

import ctypes
import os
import sys


def _install_root() -> str | None:
    if not getattr(sys, "frozen", False):
        return None
    return os.path.dirname(os.path.abspath(sys.executable))


def _linux_variant() -> str:
    root = _install_root()
    if root:
        for name in (".qube_linux_variant", ".qube-linux-variant"):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    return f.read().strip().lower()
    return os.environ.get("QUBE_LINUX_VARIANT", "").strip().lower()


def _preload_system_sonames(sonames: tuple[str, ...]) -> None:
    if not sys.platform.startswith("linux"):
        return
    dirs = (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib64",
    )
    for soname in sonames:
        for directory in dirs:
            path = os.path.join(directory, soname)
            if not os.path.isfile(path):
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            break


if _linux_variant() == "vulkan":
    _preload_system_sonames(
        (
            "libstdc++.so.6",
            "libdrm.so.2",
            "libdrm_amdgpu.so.1",
            "libvulkan.so.1",
        ),
    )