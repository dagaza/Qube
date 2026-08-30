"""Tests for bootstrap search preset helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.bootstrap_manifest import BootstrapModelId
from core.bootstrap_search_models import (
    active_search_preset_satisfied,
    all_search_presets_satisfied,
    balanced_search_preset_present,
    clear_search_preset_incomplete_cache,
    embedding_preset_cached_on_disk,
    fastembed_model_cache_markers,
    format_search_preset_download_failure,
    search_preset_has_incomplete_artifacts,
)
from core.embedding_modes import DEFAULT_MODE


def test_balanced_search_preset_present_requires_qube_preset_dir():
    from core.embedding_models import clear_embedding_availability_cache

    clear_embedding_availability_cache()
    with patch(
        "core.bootstrap_search_download.qube_preset_complete",
        return_value=False,
    ), patch(
        "core.bootstrap_search_models.search_preset_has_incomplete_artifacts",
        return_value=False,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert balanced_search_preset_present() is False
    with patch(
        "core.bootstrap_search_download.qube_preset_complete",
        return_value=True,
    ), patch(
        "core.bootstrap_search_models.search_preset_has_incomplete_artifacts",
        return_value=False,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert balanced_search_preset_present() is True
    with patch(
        "core.bootstrap_search_download.qube_preset_complete",
        return_value=False,
    ), patch(
        "core.bootstrap_search_models.search_preset_has_incomplete_artifacts",
        return_value=True,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert balanced_search_preset_present() is False


def test_format_search_preset_download_failure_mode_switch():
    body = format_search_preset_download_failure("fast", during_mode_switch=True)
    assert "Try switching again" in body
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


def test_embedding_preset_cached_on_disk_checks_qube_preset_dir():
    with patch("core.bootstrap_search_download.qube_preset_complete", return_value=True):
        assert embedding_preset_cached_on_disk("balanced") is True


def test_search_preset_has_incomplete_artifacts_detects_hf_blob(tmp_path: Path):
    cache = tmp_path / "search"
    snapshot = cache / "models--xenova--jina-embeddings-v2-small-en" / "blobs"
    snapshot.mkdir(parents=True)
    (snapshot / "abc.incomplete").write_text("partial", encoding="utf-8")
    with patch(
        "core.bootstrap_search_models.search_models_cache_dir",
        return_value=cache,
    ), patch("core.bootstrap_search_download.qube_preset_complete", return_value=False), patch(
        "core.bootstrap_search_download.qube_preset_dir",
        return_value=cache / "presets" / "balanced",
    ):
        assert search_preset_has_incomplete_artifacts("balanced") is True


def test_clear_search_preset_incomplete_cache_removes_snapshot(tmp_path: Path):
    cache = tmp_path / "search"
    snapshot = cache / "models--xenova--jina-embeddings-v2-small-en"
    blob_dir = snapshot / "blobs"
    blob_dir.mkdir(parents=True)
    (blob_dir / "abc.incomplete").write_text("partial", encoding="utf-8")
    with patch(
        "core.bootstrap_search_models.search_models_cache_dir",
        return_value=cache,
    ), patch("core.bootstrap_search_download.qube_preset_complete", return_value=False), patch(
        "core.bootstrap_search_download.qube_preset_dir",
        return_value=cache / "presets" / "balanced",
    ):
        assert clear_search_preset_incomplete_cache("balanced") is True
        assert not snapshot.exists()


def test_embedding_preset_cached_on_disk_matches_qdrant_fastembed_layout():
    with patch(
        "core.bootstrap_search_models.search_preset_has_incomplete_artifacts",
        return_value=False,
    ), patch(
        "core.bootstrap_search_download.qube_preset_complete",
        return_value=False,
    ), patch(
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
        "pathlib.Path.iterdir",
        return_value=[],
    ), patch(
        "pathlib.Path.rglob",
        return_value=[],
    ):
        assert embedding_preset_cached_on_disk("balanced") is False


def test_fastembed_model_cache_markers_include_model_basename():
    markers = fastembed_model_cache_markers("BAAI/bge-small-en-v1.5")
    assert "bge-small-en-v1.5" in markers
    assert "BAAI--bge-small-en-v1.5" in markers


def test_active_search_preset_satisfied_when_active_mode_cached():
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=True,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert active_search_preset_satisfied() is True


def test_active_search_preset_satisfied_false_when_missing():
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=False,
    ), patch(
        "core.embedding_models.preset_embedder_ready",
        return_value=False,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert active_search_preset_satisfied(probe=True) is False


def test_all_search_presets_satisfied_when_one_missing_on_disk():
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        side_effect=lambda mode: mode != "balanced",
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert all_search_presets_satisfied() is False


def test_all_search_presets_satisfied_when_all_cached():
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=True,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert all_search_presets_satisfied() is True
