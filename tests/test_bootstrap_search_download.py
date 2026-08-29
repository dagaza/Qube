"""Tests for streamed fastembed preset downloads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.bootstrap_download import _download_balanced_search_preset
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId
from core.bootstrap_search_download import (
    download_embedding_preset,
    qube_preset_complete,
    qube_preset_dir,
    resolve_preset_download_spec,
)


def test_resolve_preset_download_spec_balanced_jina_repo():
    spec = resolve_preset_download_spec("balanced")
    assert spec.hf_repo == "xenova/jina-embeddings-v2-small-en"
    assert spec.model_marker == "onnx/model.onnx"
    assert "onnx/model.onnx" in spec.files


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
    ), patch("core.bootstrap_download._download_gguf") as stream:

        def _touch(**kwargs: object) -> None:
            dest = kwargs["dest_path"]
            assert isinstance(dest, Path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")

        stream.side_effect = _touch
        download_embedding_preset("balanced", on_progress, source_display="Hugging Face")

    assert stream.call_count == len(dl_spec.files)
    repos = {call.kwargs["repo_id"] for call in stream.call_args_list}
    assert repos == {"xenova/jina-embeddings-v2-small-en"}
    assert events
    assert events[-1][1] == 100


def test_download_embedding_preset_skips_when_complete() -> None:
    events: list[int] = []

    with patch("core.bootstrap_search_download.qube_preset_complete", return_value=True), patch(
        "core.bootstrap_download._download_gguf"
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
    ), patch("core.bootstrap_search_download.download_embedding_preset") as download:
        _download_balanced_search_preset(lambda *_: None, spec)

    download.assert_called_once()
    assert download.call_args.args[0] == "balanced"


def test_download_embedding_preset_raises_when_files_missing(tmp_path: Path) -> None:
    with patch(
        "core.bootstrap_search_download.qube_preset_dir",
        return_value=tmp_path / "presets" / "fast",
    ), patch("core.bootstrap_search_download.qube_preset_complete", return_value=False), patch(
        "core.bootstrap_download._download_gguf"
    ):
        with pytest.raises(RuntimeError, match="Could not download"):
            download_embedding_preset("fast", lambda *_: None)
