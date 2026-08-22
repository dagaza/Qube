"""Tests for core/windows_gpu_bundle.py."""

from __future__ import annotations

from pathlib import Path

from core.windows_gpu_bundle import (
    missing_cuda_ggml_lib,
    missing_cuda_wheel_libs,
    missing_llama_lib,
    missing_vulkan_ggml_lib,
    missing_vulkan_loader_libs,
)


def test_missing_vulkan_loader_reports_absent_dll(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    lib_dir = dist / "_internal" / "llama_cpp" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "llama.dll").write_text("", encoding="utf-8")
    (lib_dir / "ggml-vulkan.dll").write_text("", encoding="utf-8")

    assert missing_llama_lib(dist) is None
    assert missing_vulkan_ggml_lib(dist) is None
    assert missing_vulkan_loader_libs(dist) == ["vulkan-1.dll"]


def test_missing_cuda_bundle_reports_absent_runtime(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    lib_dir = dist / "_internal" / "llama_cpp" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "llama.dll").write_text("", encoding="utf-8")
    (lib_dir / "ggml-cuda.dll").write_text("", encoding="utf-8")

    assert missing_llama_lib(dist) is None
    assert missing_cuda_ggml_lib(dist) is None
    assert set(missing_cuda_wheel_libs(dist)) == {
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
    }
