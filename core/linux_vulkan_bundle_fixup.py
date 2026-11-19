"""Disable bundled Linux libs that shadow distro Vulkan/RADV (Vulkan package variant).

PyInstaller one-dir builds can ship a bundled ``libvulkan.so.1`` under ``_internal/``. With
``LD_LIBRARY_PATH`` aimed at that tree, Mesa's ``libvulkan_radeon`` may fail to load (older
bundled ``libstdc++``). We only disable that loader copy; PyAV keeps hashed ``libdrm`` under
``av.libs`` (required for ``import av`` / STT) while ``gpu_driver_preload`` pins distro
``libdrm`` for Vulkan. Package postinst runs the same loader rename on ``.deb`` / ``.rpm``
installs; this module repeats it at startup when the install dir is writable.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("Qube.LinuxVulkanFixup")

_VARIANT_MARKERS = (".qube_linux_variant", ".qube-linux-variant")
_BUNDLED_SUFFIX = ".bundled.bak"


def frozen_install_root() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent


def read_linux_gpu_variant(install_root: Path | None = None) -> str:
    root = install_root or frozen_install_root()
    if root is not None:
        for name in _VARIANT_MARKERS:
            path = root / name
            if path.is_file():
                return path.read_text(encoding="utf-8").strip().lower()
    return os.environ.get("QUBE_LINUX_VARIANT", "").strip().lower()


def _rename_if_present(source: Path) -> str | None:
    if not source.is_file():
        return None
    backup = Path(f"{source}{_BUNDLED_SUFFIX}")
    if backup.exists():
        return None
    try:
        source.rename(backup)
    except OSError as exc:
        logger.debug("could not rename %s: %s", source, exc)
        return None
    return str(backup)


def disable_conflicting_bundled_libs(install_root: Path) -> list[str]:
    """Rename bundled ``libvulkan.so.1`` when possible. Returns new backup paths."""
    disabled: list[str] = []
    internal = install_root / "_internal"
    if not internal.is_dir():
        return disabled

    loader = internal / "libvulkan.so.1"
    moved = _rename_if_present(loader)
    if moved:
        disabled.append(moved)

    return disabled


def apply_vulkan_bundle_fixup_if_needed() -> list[str]:
    """No-op unless this is a frozen Linux Vulkan build with a writable install tree."""
    if not sys.platform.startswith("linux"):
        return []
    root = frozen_install_root()
    if root is None:
        return []
    if read_linux_gpu_variant(root) != "vulkan":
        return []
    if not os.access(root, os.W_OK):
        return []
    disabled = disable_conflicting_bundled_libs(root)
    if disabled:
        logger.info(
            "[VulkanBundleFixup] disabled bundled libs shadowing distro Vulkan: %s",
            disabled,
        )
    return disabled