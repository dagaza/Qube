"""Tests for real bootstrap asset downloads (Whisper, Kokoro)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.bootstrap_download import _download_kokoro, _download_whisper
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId
from core.stt_models import BUNDLED_WHISPER_WEIGHT_FILES


def test_download_whisper_streams_each_weight_with_progress(tmp_path: Path) -> None:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.WHISPER_SMALL]
    events: list[tuple[str, int]] = []

    def on_progress(_step: str, filename: str, percent: int, _source: str) -> None:
        events.append((filename, percent))

    whisper_dir = tmp_path / "stt" / "small"

    with patch("core.bootstrap_download.bundled_whisper_dir", return_value=whisper_dir), patch(
        "core.bootstrap_download.model_is_present",
        return_value=False,
    ), patch(
        "core.stt_models.bundled_whisper_present",
        return_value=True,
    ), patch("core.bootstrap_download._download_hf_hub_file") as stream:
        _download_whisper(on_progress, spec)

    assert stream.call_count == len(BUNDLED_WHISPER_WEIGHT_FILES)
    filenames = [call.kwargs["filename"] for call in stream.call_args_list]
    assert filenames == list(BUNDLED_WHISPER_WEIGHT_FILES)
    assert all(dest.parent == whisper_dir for dest in (call.kwargs["dest_path"] for call in stream.call_args_list))
    assert events
    assert events[-1] == ("Whisper Small", 100)


def test_download_whisper_skips_when_already_present() -> None:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.WHISPER_SMALL]
    events: list[int] = []

    with patch("core.bootstrap_download.model_is_present", return_value=True), patch(
        "core.bootstrap_download._download_hf_hub_file"
    ) as stream:
        _download_whisper(
            lambda _step, _name, pct, _source: events.append(pct),
            spec,
        )

    stream.assert_not_called()
    assert events == [100]


def test_download_kokoro_streams_each_asset_with_progress(tmp_path: Path) -> None:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.KOKORO_TTS]
    events: list[tuple[str, int]] = []

    def on_progress(_step: str, filename: str, percent: int, _source: str) -> None:
        events.append((filename, percent))

    tts_dir = tmp_path / "tts"
    tts_dir.mkdir()
    default_path = tts_dir / "kokoro-v1.0.onnx"

    with patch(
        "core.bootstrap_download.tts_default_path",
        return_value=str(default_path),
    ), patch("core.bootstrap_download._download_url_streaming") as stream:
        _download_kokoro(on_progress, spec)

    assert stream.call_count == 2
    filenames = [call.kwargs["filename"] for call in stream.call_args_list]
    assert filenames == ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
