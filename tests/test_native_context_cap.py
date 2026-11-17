"""Tests for unified-memory GPU context capping."""

from __future__ import annotations

from core.gpu_layers_cap import reset_gpu_vram_cache_for_tests
from core.native_context_cap import cap_native_load_n_ctx
from core.native_llama_load import native_load_attempts


def test_cap_skips_cpu_only_load() -> None:
    assert cap_native_load_n_ctx(requested=32000, n_gpu_layers=0) == (32000, None)


def test_cap_reduces_large_ctx_on_unified_gpu(monkeypatch, tmp_path) -> None:
    reset_gpu_vram_cache_for_tests()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * (5 * 1024 * 1024))

    monkeypatch.setattr(
        "core.native_context_cap.is_unified_gpu_memory",
        lambda: True,
    )
    monkeypatch.setattr(
        "core.native_context_cap.detect_gpu_vram_bytes",
        lambda: int(15 * 1024**3),
    )
    monkeypatch.setattr(
        "core.native_context_cap._system_ram_bytes",
        lambda: int(16 * 1024**3),
    )

    n_ctx, reason = cap_native_load_n_ctx(
        requested=32000,
        n_gpu_layers=32,
        model_path=str(gguf),
    )
    assert reason == "unified_memory_gpu_ctx_cap"
    assert n_ctx <= 4096


def test_native_load_attempts_includes_gpu_context_steps() -> None:
    plans = native_load_attempts(32, 32000)
    labels = [p[2] for p in plans]
    assert "requested" in labels
    assert "gpu_ctx8192" in labels
    assert "gpu_ctx4096" in labels
