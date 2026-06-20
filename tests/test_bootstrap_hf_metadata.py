"""Tests for Hugging Face bootstrap size resolution."""

from __future__ import annotations

from unittest.mock import patch

from core.bootstrap_hf_metadata import (
    BootstrapSizeSource,
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
    model_id = BootstrapModelId.NOMIC_EMBED
    fallback = BOOTSTRAP_MODELS[model_id].size_bytes

    with patch("core.bootstrap_hf_metadata._fetch_hf_file_size_bytes", return_value=None):
        resolved = resolve_bootstrap_size(model_id)

    assert resolved.size_bytes == fallback
    assert resolved.source is BootstrapSizeSource.ESTIMATE


def test_resolve_kokoro_sums_hf_files():
    with patch(
        "core.bootstrap_hf_metadata._fetch_hf_file_size_bytes",
        side_effect=[10_000_000, 5_000_000],
    ):
        resolved = resolve_bootstrap_size(BootstrapModelId.KOKORO_TTS)

    assert resolved.size_bytes == 15_000_000
    assert resolved.source is BootstrapSizeSource.HUGGINGFACE


def test_resolve_all_bootstrap_sizes_returns_every_model():
    with patch(
        "core.bootstrap_hf_metadata._fetch_hf_file_size_bytes",
        return_value=None,
    ):
        resolved = resolve_all_bootstrap_sizes()

    assert set(resolved) == set(BootstrapModelId)
    for entry in resolved.values():
        assert entry.size_bytes > 0
