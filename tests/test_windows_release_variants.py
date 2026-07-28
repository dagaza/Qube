"""Tests for core/windows_release_variants.py."""

from __future__ import annotations

import pytest

from core.windows_release_variants import (
    installer_description,
    installer_filename,
    normalize_windows_variant,
)


def test_normalize_windows_variant_accepts_known_values():
    assert normalize_windows_variant("cpu") == "cpu"
    assert normalize_windows_variant("CUDA") == "cuda"


def test_normalize_windows_variant_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_windows_variant("metal")


def test_installer_filename_cpu_keeps_legacy_name():
    assert installer_filename("1.2.4", "cpu") == "Qube-1.2.4-Setup.exe"
    assert installer_filename("1.2.4", "vulkan") == "Qube-1.2.4-vulkan-Setup.exe"
    assert installer_filename("1.2.4", "cuda") == "Qube-1.2.4-cuda-Setup.exe"


def test_installer_description_mentions_backend():
    assert "Vulkan" in installer_description("vulkan")
    assert "CUDA" in installer_description("cuda")
