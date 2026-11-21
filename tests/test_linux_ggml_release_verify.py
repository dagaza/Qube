"""libggml-cpu release verification (AVX-VNNI regression)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.linux_ggml_release_verify import (
    apply_bundled_llama_runtime_env,
    verify_avx512_vnni_kernels_present,
    verify_no_avx_vnni_true_stub,
    verify_q8_0_dot_avoids_vex_avx_vnni,
    verify_release_libggml_cpu,
)
from core.llama_cpp_build import linux_llama_cpp_cmake_args

BROKEN_DEB = Path("/opt/qube/_internal/libggml-cpu.so.0")


def _venv_lib() -> Path | None:
    try:
        import llama_cpp
    except ImportError:
        return None
    return Path(llama_cpp.__file__).resolve().parent / "lib" / "libggml-cpu.so.0"


@pytest.mark.skipif(shutil.which("objdump") is None, reason="objdump not installed")
def test_cmake_policy_disables_avx_vnni_not_avx512() -> None:
    for variant in ("cpu", "vulkan", "cuda"):
        args = linux_llama_cpp_cmake_args(variant)
        assert "-DGGML_NATIVE=OFF" in args
        assert "-DGGML_AVX_VNNI=OFF" in args
        assert "GGML_AVX512_VNNI=ON" in args


@pytest.mark.skipif(
    _venv_lib() is None or not _venv_lib().is_file(),
    reason="llama_cpp not installed",
)
def test_venv_lib_passes_release_verify() -> None:
    lib = _venv_lib()
    assert lib is not None
    verify_release_libggml_cpu(lib)


@pytest.mark.skipif(not BROKEN_DEB.is_file(), reason="no 1.3.55 deb installed")
def test_broken_deb_lib_fails_avx_vnni_stub() -> None:
    with pytest.raises(ValueError, match="TRUE stub"):
        verify_no_avx_vnni_true_stub(BROKEN_DEB)


@pytest.mark.skipif(not BROKEN_DEB.is_file(), reason="no 1.3.55 deb installed")
def test_broken_deb_q8_uses_vex_vnni() -> None:
    with pytest.raises(ValueError, match="VEX vpdpbusd"):
        verify_q8_0_dot_avoids_vex_avx_vnni(BROKEN_DEB)


@pytest.mark.skipif(
    _venv_lib() is None or not _venv_lib().is_file(),
    reason="llama_cpp not installed",
)
def test_venv_lib_still_has_evex_vnni_kernels() -> None:
    lib = _venv_lib()
    assert lib is not None
    verify_avx512_vnni_kernels_present(lib)


def test_apply_bundled_llama_runtime_env_sets_lib_path(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    lib_dir = dist / "_internal" / "llama_cpp" / "lib"
    lib_dir.mkdir(parents=True)
    internal = apply_bundled_llama_runtime_env(dist)
    assert internal == dist / "_internal"
    import os

    assert os.environ["LLAMA_CPP_LIB_PATH"] == str(lib_dir)
    assert str(lib_dir) in os.environ["LD_LIBRARY_PATH"]