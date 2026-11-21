"""Linux release smoke CI policy."""

from __future__ import annotations

from core.linux_release_smoke import (
    bundled_decode_skip_reason,
    vulkan_gpu_layers_skip_reason,
)


def test_cuda_skipped_on_github_actions(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("QUBE_SMOKE_FORCE_DECODE", raising=False)
    reason = bundled_decode_skip_reason("cuda")
    assert reason is not None
    assert "libcuda" in reason


def test_cpu_skipped_without_avx512_vnni_on_ci(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("QUBE_SMOKE_FORCE_DECODE", raising=False)
    from core import linux_release_smoke

    monkeypatch.setattr(linux_release_smoke, "linux_host_has_avx512_vnni", lambda: False)
    reason = bundled_decode_skip_reason("cpu")
    assert reason is not None
    assert "avx512" in reason.lower()


def test_force_decode_overrides_skip(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("QUBE_SMOKE_FORCE_DECODE", "1")
    assert bundled_decode_skip_reason("cuda") is None


def test_vulkan_gpu_layers_skipped_on_ci(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    reason = vulkan_gpu_layers_skip_reason()
    assert reason is not None