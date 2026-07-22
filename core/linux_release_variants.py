"""Linux release artifact naming and Debian package metadata."""

from __future__ import annotations

LINUX_RELEASE_VARIANTS: tuple[str, ...] = ("cpu", "vulkan", "cuda")


def normalize_linux_variant(variant: str) -> str:
    key = str(variant or "cpu").strip().lower()
    if key not in LINUX_RELEASE_VARIANTS:
        raise ValueError(f"Unsupported Linux variant {variant!r}; expected one of {LINUX_RELEASE_VARIANTS}")
    return key


def appimage_filename(version: str, variant: str) -> str:
    variant = normalize_linux_variant(variant)
    return f"Qube-{version}-x86_64-{variant}.AppImage"


def deb_filename(version: str, variant: str) -> str:
    variant = normalize_linux_variant(variant)
    if variant == "cpu":
        return f"qube_{version}_amd64.deb"
    return f"qube-{variant}_{version}_amd64.deb"


def deb_package_name(variant: str) -> str:
    variant = normalize_linux_variant(variant)
    if variant == "cpu":
        return "qube"
    return f"qube-{variant}"


def deb_conflicts(variant: str) -> str:
    variant = normalize_linux_variant(variant)
    names = [deb_package_name(v) for v in LINUX_RELEASE_VARIANTS if v != variant]
    return ", ".join(names)


def deb_description(variant: str) -> str:
    variant = normalize_linux_variant(variant)
    base = "Local hardware-accelerated AI desktop assistant"
    if variant == "cpu":
        return f"{base} (CPU inference build)"
    if variant == "vulkan":
        return f"{base} (Vulkan GPU offload build for AMD/Intel GPUs)"
    return f"{base} (CUDA GPU offload build for NVIDIA GPUs)"
