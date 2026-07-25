"""Tests for core/linux_cuda_bundle.py."""

from __future__ import annotations

import stat
from pathlib import Path

from core.linux_cuda_bundle import (
    GITHUB_RELEASE_ASSET_LIMIT_BYTES,
    REQUIRED_CUDA_WHEEL_LIBS,
    missing_cuda_wheel_libs,
    missing_llama_lib,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_required_cuda_libs_constant_matches_bundle_verifier() -> None:
    assert "libcudart.so.12" in REQUIRED_CUDA_WHEEL_LIBS
    assert "libcublasLt.so.12" in REQUIRED_CUDA_WHEEL_LIBS


def test_missing_cuda_wheel_libs_reports_only_absent_files(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    lib_dir = dist / "_internal" / "llama_cpp" / "lib"
    _touch(lib_dir / "libcudart.so.12")

    missing = missing_cuda_wheel_libs(dist)
    assert "libcudart.so.12" not in missing
    assert "libcublas.so.12" in missing


def test_missing_llama_lib_none_when_present(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    _touch(dist / "_internal" / "llama_cpp" / "lib" / "libllama.so")
    assert missing_llama_lib(dist) is None


def test_github_release_limit_is_two_gib() -> None:
    assert GITHUB_RELEASE_ASSET_LIMIT_BYTES == 2 * 1024 * 1024 * 1024
