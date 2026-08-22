"""Locate the Windows Vulkan loader for bundling in release builds."""

from __future__ import annotations

import os
from pathlib import Path


def iter_vulkan_loader_candidates() -> list[Path]:
    """Return existing vulkan-1.dll paths, most preferred first."""
    seen: set[Path] = set()
    candidates: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    sdk = os.environ.get("VULKAN_SDK", "").strip()
    if sdk:
        root = Path(sdk)
        for rel in (
            "Bin/vulkan-1.dll",
            "Runtime/x64/vulkan-1.dll",
            "Runtime/vulkan-1.dll",
        ):
            add(root / rel)

    sdk_root = Path(r"C:\VulkanSDK")
    if sdk_root.is_dir():
        for sdk_dir in sorted(sdk_root.iterdir(), reverse=True):
            if not sdk_dir.is_dir():
                continue
            for rel in (
                "Bin/vulkan-1.dll",
                "Runtime/x64/vulkan-1.dll",
                "Runtime/vulkan-1.dll",
            ):
                add(sdk_dir / rel)

    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll"
    add(system32)

    return [path for path in candidates if path.is_file()]


def find_vulkan_loader() -> Path:
    matches = iter_vulkan_loader_candidates()
    if not matches:
        raise FileNotFoundError(
            "vulkan-1.dll not found. Install the Vulkan SDK or Vulkan Runtime "
            "(set VULKAN_SDK or install KhronosGroup.VulkanRT)."
        )
    return matches[0]
