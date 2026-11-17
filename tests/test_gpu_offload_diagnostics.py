"""Tests for GPU offload inactive detection."""

from __future__ import annotations

from core.inference_transparency import gpu_offload_inactive_reason
from core.notification_types import native_gpu_offload_inactive_event


def test_gpu_offload_inactive_reason_when_build_has_no_offload() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=54,
            supports_gpu_offload=False,
            load_fallback_label="requested",
            dri_render_accessible=True,
            package_has_vulkan_libs=False,
        )
        == "no_gpu_offload_build"
    )


def test_gpu_offload_inactive_reason_when_vulkan_libs_present() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=54,
            supports_gpu_offload=False,
            load_fallback_label="requested",
            dri_render_accessible=True,
            package_has_vulkan_libs=True,
            shadowing_libdrm_path=None,
        )
        == "vulkan_runtime_unavailable"
    )


def test_gpu_offload_inactive_reason_when_libdrm_shadowed() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=54,
            supports_gpu_offload=False,
            dri_render_accessible=True,
            package_has_vulkan_libs=True,
            shadowing_libdrm_path="/venv/site-packages/torch/lib/libdrm_amdgpu.so",
        )
        == "libdrm_shadowed"
    )


def test_gpu_offload_inactive_reason_checks_gpu_ctx_fallback() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=54,
            supports_gpu_offload=False,
            load_fallback_label="gpu_ctx4096",
            dri_render_accessible=True,
            package_has_vulkan_libs=False,
        )
        == "no_gpu_offload_build"
    )


def test_gpu_offload_inactive_reason_when_dri_missing() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=33,
            supports_gpu_offload=False,
            load_fallback_label="requested",
            dri_render_accessible=False,
        )
        == "no_dri_access"
    )


def test_gpu_offload_inactive_reason_skips_zero_layers() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=0,
            supports_gpu_offload=False,
        )
        is None
    )


def test_gpu_offload_inactive_reason_skips_vram_fallback() -> None:
    assert (
        gpu_offload_inactive_reason(
            requested_n_gpu_layers=54,
            supports_gpu_offload=False,
            load_fallback_label="cpu_ctx2048",
        )
        is None
    )


def test_native_gpu_offload_inactive_event_body() -> None:
    event = native_gpu_offload_inactive_event(
        model_name="Qwen3.5-9B-Q6_K.gguf",
        reason="no_gpu_offload_build",
    )
    assert event.title == "Model on CPU (GPU offload inactive)"
    assert "loaded successfully" in event.body
    assert "install_llama_cpp_gpu.sh" in event.body


def test_native_gpu_offload_inactive_event_vulkan_runtime() -> None:
    event = native_gpu_offload_inactive_event(
        model_name="model.gguf",
        reason="vulkan_runtime_unavailable",
    )
    assert "Chat still works on CPU" in event.body
    assert "vulkaninfo" in event.body
