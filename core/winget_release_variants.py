"""WinGet package identifiers and metadata for Windows release variants."""

from __future__ import annotations

from core.windows_release_variants import (
    WINDOWS_RELEASE_VARIANTS,
    installer_description,
    installer_filename,
    normalize_windows_variant,
)

WINGET_VARIANTS: tuple[str, ...] = WINDOWS_RELEASE_VARIANTS

_PACKAGE_IDS: dict[str, str] = {
    "cpu": "dagaza.Qube",
    "vulkan": "dagaza.Qube.Vulkan",
    "cuda": "dagaza.Qube.CUDA",
}

_PACKAGE_NAMES: dict[str, str] = {
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

_MONIKERS: dict[str, str] = {
    "cpu": "qube",
    "vulkan": "qube-vulkan",
    "cuda": "qube-cuda",
}


def package_identifier(variant: str) -> str:
    key = normalize_windows_variant(variant)
    return _PACKAGE_IDS[key]


def package_name(variant: str) -> str:
    key = normalize_windows_variant(variant)
    return _PACKAGE_NAMES[key]


def package_moniker(variant: str) -> str:
    key = normalize_windows_variant(variant)
    return _MONIKERS[key]


def package_tags(variant: str) -> tuple[str, ...]:
    key = normalize_windows_variant(variant)
    return _BASE_TAGS + _VARIANT_TAGS[key]


def installer_url(version: str, variant: str, repo: str = "dagaza/Qube") -> str:
    key = normalize_windows_variant(variant)
    filename = installer_filename(version, key)
    return f"https://github.com/{repo}/releases/download/v{version}/{filename}"


def short_description(variant: str) -> str:
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
            "For GPU acceleration, also see dagaza.Qube.Vulkan (AMD/Intel) or "
            "dagaza.Qube.CUDA (NVIDIA) on WinGet, or download from GitHub Releases."
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
