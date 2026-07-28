"""Tests for core/chocolatey_release_variants.py."""

from __future__ import annotations

from core.chocolatey_release_variants import (
    CHOCOLATEY_VARIANTS,
    installer_url,
    package_description,
    package_id,
    package_summary,
    package_tags,
    package_title,
)


def test_chocolatey_variants_match_windows_release_variants():
    assert CHOCOLATEY_VARIANTS == ("cpu", "vulkan", "cuda")


def test_package_ids():
    assert package_id("cpu") == "qube"
    assert package_id("vulkan") == "qube-vulkan"
    assert package_id("cuda") == "qube-cuda"


def test_installer_urls():
    assert (
        installer_url("1.2.5", "cpu")
        == "https://github.com/dagaza/Qube/releases/download/v1.2.5/Qube-1.2.5-Setup.exe"
    )
    assert (
        installer_url("1.2.5", "vulkan")
        == "https://github.com/dagaza/Qube/releases/download/v1.2.5/Qube-1.2.5-vulkan-Setup.exe"
    )
    assert (
        installer_url("1.2.5", "cuda")
        == "https://github.com/dagaza/Qube/releases/download/v1.2.5/Qube-1.2.5-cuda-Setup.exe"
    )


def test_titles_and_tags():
    assert package_title("cuda") == "Qube (CUDA)"
    assert "gpu" not in package_tags("cpu")
    assert "vulkan" in package_tags("vulkan")
    assert "nvidia" in package_tags("cuda")


def test_descriptions_mention_variant_context():
    assert "qube-vulkan" in package_description("cpu")
    assert "AMD and Intel" in package_description("vulkan")
    assert "NVIDIA" in package_summary("cuda")
