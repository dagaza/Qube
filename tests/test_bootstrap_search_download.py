"""Tests for streamed fastembed preset downloads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.bootstrap_download import (
    _download_balanced_search_preset,
    _sanitize_hub_repo_file_path,
)
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId
from core.bootstrap_search_download import (
    download_embedding_preset,
    qube_preset_complete,
    qube_preset_dir,
    resolve_preset_download_spec,
)


def test_resolve_preset_download_spec_balanced_jina_repo():
    spec = resolve_preset_download_spec("balanced")
    assert spec.hf_repo == "Xenova/jina-embeddings-v2-small-en"
    assert spec.model_marker == "onnx/model.onnx"
    assert "onnx/model.onnx" in spec.files
    assert "preprocessor_config.json" not in spec.files


def test_canonical_hf_repo_keeps_known_casing_and_passes_unknown_through():
    from core.bootstrap_search_download import _canonical_hf_repo

    assert _canonical_hf_repo("balanced", "xenova/jina-embeddings-v2-small-en") == (
        "Xenova/jina-embeddings-v2-small-en"
    )
    assert _canonical_hf_repo("balanced", "someone/other-model") == "someone/other-model"


def test_sanitize_hub_repo_file_path_allows_onnx_and_tokenizer_assets():
    assert _sanitize_hub_repo_file_path("onnx/model.onnx") == "onnx/model.onnx"
    assert _sanitize_hub_repo_file_path("config.json") == "config.json"
    assert _sanitize_hub_repo_file_path("tokenizer.json") == "tokenizer.json"


def test_download_embedding_preset_streams_each_file(tmp_path: Path) -> None:
    events: list[tuple[str, int]] = []

    def on_progress(_step: str, filename: str, percent: int, _source: str) -> None:
        events.append((filename, percent))

    preset_dir = tmp_path / "presets" / "balanced"
    dl_spec = resolve_preset_download_spec("balanced")

    def _complete(_mode: str | None = None) -> bool:
        return all((preset_dir / rel).is_file() for rel in dl_spec.files)

    with patch(
        "core.bootstrap_search_download.qube_preset_dir",
        return_value=preset_dir,
    ), patch(
        "core.bootstrap_search_download.qube_preset_complete",
        side_effect=_complete,
    ), patch("core.bootstrap_download._download_hf_hub_file") as stream:

        def _touch(**kwargs: object) -> None:
            dest = kwargs["dest_path"]
            assert isinstance(dest, Path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")

        stream.side_effect = _touch
        download_embedding_preset("balanced", on_progress, source_display="Hugging Face")

    assert stream.call_count == len(dl_spec.files)
    repos = {call.kwargs["repo_id"] for call in stream.call_args_list}
    assert repos == {"Xenova/jina-embeddings-v2-small-en"}
    assert events
    assert events[-1][1] == 100


def test_download_embedding_preset_skips_when_complete() -> None:
    events: list[int] = []

    with patch("core.bootstrap_search_download.qube_preset_complete", return_value=True), patch(
        "core.bootstrap_download._download_hf_hub_file"
    ) as stream:
        download_embedding_preset(
            "balanced",
            lambda _step, _name, pct, _source: events.append(pct),
        )

    stream.assert_not_called()
    assert events == [100]


def test_download_balanced_search_preset_uses_streaming_helper() -> None:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.SEARCH_PRESET_BALANCED]

    with patch(
        "core.bootstrap_search_models.balanced_search_preset_present",
        return_value=False,
    ), patch("core.bootstrap_search_download.download_embedding_preset") as download, patch(
        "core.bootstrap_search_download.reranker_complete", return_value=True
    ), patch("core.bootstrap_search_download.download_reranker") as rr:
        _download_balanced_search_preset(lambda *_: None, spec)

    download.assert_called_once()
    assert download.call_args.args[0] == "balanced"
    rr.assert_not_called()


def test_balanced_search_also_downloads_missing_reranker() -> None:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.SEARCH_PRESET_BALANCED]
    with patch(
        "core.bootstrap_search_models.balanced_search_preset_present", return_value=True
    ), patch("core.bootstrap_search_download.download_embedding_preset") as emb, patch(
        "core.bootstrap_search_download.reranker_complete", return_value=False
    ), patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True), patch(
        "core.bootstrap_search_download.download_reranker"
    ) as rr:
        _download_balanced_search_preset(lambda *_: None, spec)
    emb.assert_not_called()
    rr.assert_called_once()


def test_reranker_not_downloaded_when_disabled() -> None:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.SEARCH_PRESET_BALANCED]
    with patch(
        "core.bootstrap_search_models.balanced_search_preset_present", return_value=True
    ), patch("core.bootstrap_search_download.reranker_complete", return_value=False), patch(
        "core.app_settings.get_knowledge_reranker_enabled", return_value=False
    ), patch("core.bootstrap_search_download.download_reranker") as rr:
        _download_balanced_search_preset(lambda *_: None, spec)
    rr.assert_not_called()


def test_reranker_failure_is_reported_not_fatal() -> None:
    """A reranker download failure becomes a bootstrap error string; startup continues."""
    from core.bootstrap_download import download_bootstrap_models

    with patch(
        "core.bootstrap_search_models.balanced_search_preset_present", return_value=True
    ), patch("core.bootstrap_search_download.reranker_complete", return_value=False), patch(
        "core.app_settings.get_knowledge_reranker_enabled", return_value=True
    ), patch(
        "core.bootstrap_search_download.download_reranker",
        side_effect=RuntimeError("network down"),
    ):
        errors = download_bootstrap_models(
            {BootstrapModelId.SEARCH_PRESET_BALANCED}, lambda *_: None
        )
    assert len(errors) == 1 and "network down" in errors[0]


def test_reranker_failure_marker_backs_off_then_clears(tmp_path: Path) -> None:
    import core.bootstrap_download as bd

    spec = BOOTSTRAP_MODELS[BootstrapModelId.SEARCH_PRESET_BALANCED]
    with patch(
        "core.bootstrap_search_download.reranker_dir", return_value=tmp_path / "reranker"
    ), patch("core.bootstrap_search_download.reranker_complete", return_value=False), patch(
        "core.app_settings.get_knowledge_reranker_enabled", return_value=True
    ):
        assert bd._reranker_download_backing_off() is False
        with patch(
            "core.bootstrap_search_download.download_reranker",
            side_effect=RuntimeError("offline"),
        ):
            try:
                bd._download_search_reranker(lambda *_: None, spec)
            except RuntimeError:
                pass
        assert bd._reranker_download_backing_off() is True
        with patch("core.bootstrap_search_download.download_reranker"):
            bd._download_search_reranker(lambda *_: None, spec)
        assert bd._reranker_download_backing_off() is False


def test_needing_download_includes_balanced_when_only_reranker_missing() -> None:
    from core.bootstrap_download import selected_models_needing_download

    sel = {BootstrapModelId.SEARCH_PRESET_BALANCED}
    with patch("core.bootstrap_download.model_is_present", return_value=True), patch(
        "core.bootstrap_download.search_reranker_missing", return_value=True
    ), patch("core.bootstrap_download._reranker_download_backing_off", return_value=False):
        assert selected_models_needing_download(sel) == sel
    with patch("core.bootstrap_download.model_is_present", return_value=True), patch(
        "core.bootstrap_download.search_reranker_missing", return_value=True
    ), patch("core.bootstrap_download._reranker_download_backing_off", return_value=True):
        assert selected_models_needing_download(sel) == set()
    with patch("core.bootstrap_download.model_is_present", return_value=True), patch(
        "core.bootstrap_download.search_reranker_missing", return_value=False
    ):
        assert selected_models_needing_download(sel) == set()


def test_download_embedding_preset_raises_when_files_missing(tmp_path: Path) -> None:
    with patch(
        "core.bootstrap_search_download.qube_preset_dir",
        return_value=tmp_path / "presets" / "fast",
    ), patch("core.bootstrap_search_download.qube_preset_complete", return_value=False), patch(
        "core.bootstrap_download._download_hf_hub_file"
    ):
        with pytest.raises(RuntimeError, match="Could not download"):
            download_embedding_preset("fast", lambda *_: None)
