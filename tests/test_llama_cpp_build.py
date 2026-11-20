"""Linux llama-cpp CMAKE policy (Zen4 AVX-VNNI vs AVX512-VNNI)."""

from core.llama_cpp_build import linux_llama_cpp_cmake_args


def test_vulkan_cmake_disables_avx_vnni() -> None:
    args = linux_llama_cpp_cmake_args("vulkan")
    assert "-DGGML_VULKAN=on" in args
    assert "-DGGML_AVX_VNNI=OFF" in args


def test_cuda_source_cmake_disables_avx_vnni() -> None:
    args = linux_llama_cpp_cmake_args("cuda")
    assert "-DGGML_CUDA=on" in args
    assert "-DGGML_AVX_VNNI=OFF" in args


def test_cpu_variant_returns_only_avx_vnni_off() -> None:
    assert linux_llama_cpp_cmake_args("cpu") == "-DGGML_AVX_VNNI=OFF"