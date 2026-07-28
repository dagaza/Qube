"""Tests for NVIDIA wheel library discovery."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from core import nvidia_wheel_lib_dirs as nwld


def _install_fake_nvidia_package(monkeypatch: pytest.MonkeyPatch, name: str, root: Path) -> None:
    mod = ModuleType(name)
    mod.__path__ = [str(root)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, mod)


def test_nvidia_package_lib_dirs_prefers_bin_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package_root = tmp_path / "cuda_runtime"
    (package_root / "lib" / "x64").mkdir(parents=True)
    (package_root / "bin").mkdir(parents=True)
    (package_root / "bin" / "cudart64_12.dll").write_bytes(b"fake")
    _install_fake_nvidia_package(monkeypatch, "nvidia.cuda_runtime", package_root)

    monkeypatch.setattr(nwld.sys, "platform", "win32")
    dirs = nwld.nvidia_package_lib_dirs("nvidia.cuda_runtime")

    assert dirs[0] == package_root / "bin"
    assert (package_root / "lib") in dirs


def test_iter_nvidia_wheel_libs_finds_dlls_in_bin_when_lib_has_no_dlls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path / "cuda_runtime"
    cublas_root = tmp_path / "cublas"
    (runtime_root / "lib" / "x64").mkdir(parents=True)
    (runtime_root / "bin").mkdir(parents=True)
    (runtime_root / "bin" / "cudart64_12.dll").write_bytes(b"fake")
    (cublas_root / "bin").mkdir(parents=True)
    (cublas_root / "bin" / "cublas64_12.dll").write_bytes(b"fake")
    (cublas_root / "bin" / "cublasLt64_12.dll").write_bytes(b"fake")
    _install_fake_nvidia_package(monkeypatch, "nvidia.cuda_runtime", runtime_root)
    _install_fake_nvidia_package(monkeypatch, "nvidia.cublas", cublas_root)

    monkeypatch.setattr(nwld.sys, "platform", "win32")
    libs = nwld.iter_nvidia_wheel_libs("nvidia.cuda_runtime", "nvidia.cublas")

    assert {path.name for path in libs} == {
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
    }
