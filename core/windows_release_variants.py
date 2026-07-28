"""Windows release artifact naming and metadata."""

from __future__ import annotations

WINDOWS_RELEASE_VARIANTS: tuple[str, ...] = ("cpu", "vulkan", "cuda")


def normalize_windows_variant(variant: str) -> str:
    key = str(variant or "cpu").strip().lower()
    if key not in WINDOWS_RELEASE_VARIANTS:
        raise ValueError(
            f"Unsupported Windows variant {variant!r}; expected one of {WINDOWS_RELEASE_VARIANTS}"
        )
    return key


def installer_filename(version: str, variant: str) -> str:
    variant = normalize_windows_variant(variant)
    if variant == "cpu":
        return f"Qube-{version}-Setup.exe"
    return f"Qube-{version}-{variant}-Setup.exe"


def installer_description(variant: str) -> str:
    variant = normalize_windows_variant(variant)
    base = "Local hardware-accelerated AI desktop assistant"
    if variant == "cpu":
        return f"{base} (CPU inference build)"
    if variant == "vulkan":
        return f"{base} (Vulkan GPU offload build for AMD/Intel GPUs)"
    return f"{base} (CUDA GPU offload build for NVIDIA GPUs)"
