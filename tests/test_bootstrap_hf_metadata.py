"""Tests for Hugging Face bootstrap size resolution."""

from __future__ import annotations

from unittest.mock import patch

from core.bootstrap_hf_metadata import (
    BootstrapSizeSource,
    ResolvedBootstrapSize,
    format_bootstrap_size_tag_tooltip,
    resolve_all_bootstrap_sizes,
    resolve_bootstrap_size,
)
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId


def test_resolve_gguf_size_from_huggingface():
    model_id = BootstrapModelId.LLM_QWEN35_9B
    spec = BOOTSTRAP_MODELS[model_id]
    live_size = spec.size_bytes + 42

    with patch(
        "core.bootstrap_hf_metadata._fetch_hf_file_size_bytes",
        return_value=live_size,
    ):
        resolved = resolve_bootstrap_size(model_id)

    assert resolved.size_bytes == live_size
    assert resolved.source is BootstrapSizeSource.HUGGINGFACE
    assert spec.hf_filename in resolved.detail


def test_resolve_gguf_falls_back_to_estimate_when_hf_unavailable():
    model_id = BootstrapModelId.SIDECAR_QWEN17
    fallback = BOOTSTRAP_MODELS[model_id].size_bytes

    with patch("core.bootstrap_hf_metadata._fetch_hf_file_size_bytes", return_value=None):
        resolved = resolve_bootstrap_size(model_id)

    assert resolved.size_bytes == fallback
    assert resolved.source is BootstrapSizeSource.ESTIMATE


def test_resolve_kokoro_sums_release_assets():
    with patch(
        "core.bootstrap_hf_metadata._fetch_url_content_length",
        side_effect=[10_000_000, 5_000_000],
    ):
        resolved = resolve_bootstrap_size(BootstrapModelId.KOKORO_TTS)

    assert resolved.size_bytes == 15_000_000
    assert resolved.source is BootstrapSizeSource.ESTIMATE
    assert "kokoro-onnx" in resolved.detail


def test_resolve_all_bootstrap_sizes_returns_every_model():
    with patch(
        "core.bootstrap_hf_metadata._fetch_hf_file_size_bytes",
        return_value=None,
    ):
        resolved = resolve_all_bootstrap_sizes()

    assert set(resolved) == set(BootstrapModelId)
    for entry in resolved.values():
        assert entry.size_bytes > 0


def test_size_tag_tooltip_verified():
    entry = ResolvedBootstrapSize(
        model_id=BootstrapModelId.SIDECAR_QWEN17,
        size_bytes=1_400_000_000,
        source=BootstrapSizeSource.HUGGINGFACE,
        detail="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf",
    )
    text = format_bootstrap_size_tag_tooltip(entry)
    assert "Verified" in text
    assert "Hugging Face" in text
    assert "Technical:" in text
    assert "unsloth" in text


def test_size_tag_tooltip_estimated():
    entry = ResolvedBootstrapSize(
        model_id=BootstrapModelId.WHISPER_SMALL,
        size_bytes=500_000_000,
        source=BootstrapSizeSource.ESTIMATE,
        detail="Systran/faster-whisper-small cache footprint (approximate)",
    )
    text = format_bootstrap_size_tag_tooltip(entry)
    assert "Estimated" in text
    assert "approximate" in text.lower()
    assert "Technical:" in text
    assert "Whisper" in text


def test_size_tag_tooltip_loading():
    entry = ResolvedBootstrapSize(
        model_id=BootstrapModelId.SIDECAR_QWEN17,
        size_bytes=1_400_000_000,
        source=BootstrapSizeSource.ESTIMATE,
        detail="Loading…",
    )
    text = format_bootstrap_size_tag_tooltip(entry)
    assert "checking online" in text.lower()
    assert "Technical:" in text
