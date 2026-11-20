"""Linux llama-cpp-python source-build CMake flags for Qube release variants."""

from __future__ import annotations


def linux_llama_cpp_cmake_args(variant: str) -> str:
    """
    Extra ``CMAKE_ARGS`` for ``pip install --no-binary=llama-cpp-python``.

    All Linux release variants (cpu, vulkan, cuda source builds) compile ggml from
    source in CI, except CUDA when a vetted prebuilt wheel passes post-install lib
    checks. GitHub-hosted
    runners are often Intel parts with AVX-VNNI, so a default ``GGML_NATIVE`` build
    can bake in VEX-encoded ``vpdpbusd`` paths and a compile-time
    ``ggml_cpu_has_avx_vnni()`` stub that always returns true.

    AMD Zen 4 / 4c (e.g. Ryzen 7840HS) expose ``avx512_vnni`` but not Linux's
    ``avx_vnni`` flag; VEX AVX-VNNI opcodes can SIGILL while EVEX AVX512-VNNI
    kernels still work. Disabling ``GGML_AVX_VNNI`` keeps AVX512-VNNI and other
    CPU optimizations via ``GGML_NATIVE``.
    """
    key = (variant or "cpu").strip().lower()
    parts: list[str] = []
    if key == "vulkan":
        parts.append("-DGGML_VULKAN=on")
    elif key == "cuda":
        parts.append("-DGGML_CUDA=on")
    parts.append("-DGGML_AVX_VNNI=OFF")
    return " ".join(parts)