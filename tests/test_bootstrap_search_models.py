"""Tests for bootstrap search preset helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.bootstrap_manifest import BootstrapModelId
from core.bootstrap_search_models import (
    balanced_search_preset_present,
    embedding_preset_cached_on_disk,
    fastembed_model_cache_markers,
    format_search_preset_download_failure,
)
from core.embedding_modes import DEFAULT_MODE


def test_balanced_search_preset_present_uses_disk_cache():
    from core.embedding_models import clear_embedding_availability_cache

    clear_embedding_availability_cache()
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=False,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert balanced_search_preset_present() is False
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=True,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert balanced_search_preset_present() is True


def test_format_search_preset_download_failure_mode_switch():
    body = format_search_preset_download_failure("fast", during_mode_switch=True)
    assert "try switching again" in body
    assert "Prepare search models on this page" not in body


def test_format_search_preset_download_failure_prepare_hint():
    body = format_search_preset_download_failure("fast", during_mode_switch=False)
    assert "Prepare search models on this page" in body


def test_model_is_present_for_balanced_search_preset():
    from core.bootstrap_download import model_is_present

    with patch(
        "core.bootstrap_search_models.balanced_search_preset_present",
        return_value=True,
    ):
        assert model_is_present(BootstrapModelId.SEARCH_PRESET_BALANCED) is True


def test_embedding_preset_cached_on_disk_matches_qdrant_fastembed_layout():
    with patch(
        "core.bootstrap_search_models.search_models_cache_dir",
        return_value=Path("/tmp/qube-search-cache"),
    ), patch("pathlib.Path.is_dir", return_value=True), patch(
        "pathlib.Path.iterdir",
        return_value=[
            Path("/tmp/qube-search-cache/models--qdrant--bge-small-en-v1.5-onnx-q"),
        ],
    ), patch(
        "pathlib.Path.rglob",
        return_value=[
            Path(
                "/tmp/qube-search-cache/models--qdrant--bge-small-en-v1.5-onnx-q/"
                "snapshots/abc/model_optimized.onnx"
            )
        ],
    ):
        assert embedding_preset_cached_on_disk("fast") is True


def test_embedding_preset_cached_on_disk_checks_hf_hub_layout():
    with patch("pathlib.Path.is_dir", return_value=True), patch(
        "pathlib.Path.rglob",
        return_value=[],
    ):
        assert embedding_preset_cached_on_disk("balanced") is False


def test_fastembed_model_cache_markers_include_model_basename():
    markers = fastembed_model_cache_markers("BAAI/bge-small-en-v1.5")
    assert "bge-small-en-v1.5" in markers
    assert "BAAI--bge-small-en-v1.5" in markers
