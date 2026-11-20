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


def test_cpu_variant_enables_avx512_vnni_only() -> None:
    assert linux_llama_cpp_cmake_args("cpu") == (
        "-DGGML_NATIVE=OFF -DGGML_AVX_VNNI=OFF "
        "-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON"
    )