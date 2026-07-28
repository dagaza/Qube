"""Chocolatey package identifiers and metadata for Windows release variants."""

from __future__ import annotations

from core.windows_release_variants import (
    WINDOWS_RELEASE_VARIANTS,
    installer_description,
    installer_filename,
    normalize_windows_variant,
)

CHOCOLATEY_VARIANTS: tuple[str, ...] = WINDOWS_RELEASE_VARIANTS

_PACKAGE_IDS: dict[str, str] = {
    "cpu": "qube",
    "vulkan": "qube-vulkan",
    "cuda": "qube-cuda",
}

_PACKAGE_TITLES: dict[str, str] = {
    "cpu": "Qube",
    "vulkan": "Qube (Vulkan)",
    "cuda": "Qube (CUDA)",
}

_BASE_TAGS: tuple[str, ...] = (
    "ai",
    "assistant",
    "local",
    "privacy",
    "voice",
    "llm",
    "desktop",
)

_VARIANT_TAGS: dict[str, tuple[str, ...]] = {
    "cpu": (),
    "vulkan": ("vulkan", "gpu"),
    "cuda": ("cuda", "nvidia", "gpu"),
}


def package_id(variant: str) -> str:
    key = normalize_windows_variant(variant)
    return _PACKAGE_IDS[key]


def package_title(variant: str) -> str:
    key = normalize_windows_variant(variant)
    return _PACKAGE_TITLES[key]


def package_tags(variant: str) -> str:
    key = normalize_windows_variant(variant)
    return " ".join(_BASE_TAGS + _VARIANT_TAGS[key])


def package_summary(variant: str) -> str:
    return installer_description(variant)


def package_description(variant: str, repo: str = "dagaza/Qube") -> str:
    key = normalize_windows_variant(variant)
    base = (
        "Qube is a fully local, privacy-first, voice-to-voice AI desktop assistant. "
        "It integrates speech-to-text, text-to-speech, retrieval-augmented generation, "
        "and local LLM inference into a native PyQt6 desktop shell."
    )
    if key == "cpu":
        suffix = (
            " This package is the CPU inference build (works on any PC). "
            "For GPU acceleration, also see qube-vulkan (AMD/Intel) or qube-cuda (NVIDIA) "
            "on Chocolatey, WinGet, or GitHub Releases."
        )
    elif key == "vulkan":
        suffix = (
            " This package is the Vulkan GPU offload build for AMD and Intel GPUs. "
            "Install only one Qube variant; user data is shared across variants."
        )
    else:
        suffix = (
            " This package is the CUDA GPU offload build for NVIDIA GPUs "
            "(recent proprietary driver required). Install only one Qube variant; "
            "user data is shared across variants."
        )
    return base + suffix


def installer_url(version: str, variant: str, repo: str = "dagaza/Qube") -> str:
    key = normalize_windows_variant(variant)
    filename = installer_filename(version, key)
    return f"https://github.com/{repo}/releases/download/v{version}/{filename}"


def nuspec_filename(variant: str) -> str:
    return f"{package_id(variant)}.nuspec"
