"""Tests for core/linux_release_variants.py."""

from __future__ import annotations

import pytest

from core.linux_release_variants import (
    appimage_filename,
    deb_conflicts,
    deb_description,
    deb_filename,
    deb_package_name,
    normalize_linux_variant,
    rpm_filename,
    tarball_filename,
)


def test_normalize_linux_variant_accepts_known_values():
    assert normalize_linux_variant("cpu") == "cpu"
    assert normalize_linux_variant("VULKAN") == "vulkan"


def test_normalize_linux_variant_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_linux_variant("metal")


def test_appimage_filename_includes_variant():
    assert appimage_filename("1.2.3", "cuda") == "Qube-1.2.3-x86_64-cuda.AppImage"


def test_deb_filename_cpu_keeps_legacy_name():
    assert deb_filename("1.2.3", "cpu") == "qube_1.2.3_amd64.deb"
    assert deb_filename("1.2.3", "vulkan") == "qube-vulkan_1.2.3_amd64.deb"


def test_deb_package_conflicts_are_mutually_exclusive():
    assert "qube-vulkan" in deb_conflicts("cpu")
    assert "qube" in deb_conflicts("cuda")
    assert "qube-cuda" in deb_conflicts("vulkan")


def test_deb_description_mentions_backend():
    assert "Vulkan" in deb_description("vulkan")
    assert "CUDA" in deb_description("cuda")


def test_rpm_filename_matches_fpm_convention():
    assert rpm_filename("1.2.3", "cpu") == "qube-1.2.3-1.x86_64.rpm"
    assert rpm_filename("1.2.3", "cuda") == "qube-cuda-1.2.3-1.x86_64.rpm"


def test_tarball_filename_matches_appimage_variant_suffix():
    assert tarball_filename("1.2.3", "vulkan") == "Qube-1.2.3-x86_64-vulkan.tar.gz"
