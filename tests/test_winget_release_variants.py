"""Tests for core/winget_release_variants.py."""

from __future__ import annotations

from core.winget_release_variants import (
    WINGET_VARIANTS,
    installer_url,
    package_description,
    package_identifier,
    package_moniker,
    package_name,
    package_tags,
)


def test_winget_variants_match_windows_release_variants():
    assert WINGET_VARIANTS == ("cpu", "vulkan", "cuda")


def test_package_identifiers():
    assert package_identifier("cpu") == "dagaza.Qube"
    assert package_identifier("vulkan") == "dagaza.Qube.Vulkan"
    assert package_identifier("cuda") == "dagaza.Qube.CUDA"


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


def test_package_names_and_monikers():
    assert package_name("cpu") == "Qube"
    assert package_name("vulkan") == "Qube (Vulkan)"
    assert package_name("cuda") == "Qube (CUDA)"
    assert package_moniker("cpu") == "qube"
    assert package_moniker("vulkan") == "qube-vulkan"
    assert package_moniker("cuda") == "qube-cuda"


def test_package_tags_include_gpu_labels():
    assert "gpu" not in package_tags("cpu")
    assert "vulkan" in package_tags("vulkan")
    assert "cuda" in package_tags("cuda")
    assert "nvidia" in package_tags("cuda")


def test_descriptions_mention_variant_context():
    assert "dagaza.Qube.Vulkan" in package_description("cpu")
    assert "AMD and Intel" in package_description("vulkan")
    assert "NVIDIA" in package_description("cuda")
