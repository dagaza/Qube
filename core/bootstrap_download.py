"""Download bootstrap model assets before splash embedder load."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url

from core.app_settings import get_llm_models_dir
from core.auxiliary_cognition import get_cognition_models_dir
from core.bootstrap_manifest import (
    BOOTSTRAP_MODELS,
    BootstrapModelId,
    format_byte_size,
    total_selected_bytes,
)
from core.stt_models import (
    BUNDLED_STT_HF_REPO,
    BUNDLED_WHISPER_WEIGHT_FILES,
    bundled_whisper_dir,
    get_stt_models_dir,
)
from core.tts_models import bundled_default_path as tts_default_path
from workers.model_download_worker import SAFETY_BUFFER_BYTES, _sanitize_repo_file_path, _sanitize_repo_id

logger = logging.getLogger("Qube.Bootstrap.Download")

DownloadProgressCallback = Callable[[str, str, int, str], None]
# step_label, filename, percent, source_display

# Simulated throughput for first-run / mock splash downloads (~640 Mbps, 80 MiB/s).
# Keeps progress visible while avoiding multi-minute waits for multi-GB selections.
_MOCK_GOOD_BANDWIDTH_BYTES_PER_SEC = 80 * 1024 * 1024
_MOCK_TICK_SEC = 0.05
_MOCK_MIN_MODEL_SEC = 0.35


def bootstrap_download_mock_enabled() -> bool:
    """True when bootstrap downloads should be simulated instead of fetched."""
    return os.environ.get("QUBE_BOOTSTRAP_MOCK_DOWNLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def bootstrap_real_download_forced() -> bool:
    """Force real downloads even when mock flags are set (``QUBE_BOOTSTRAP_REAL_DOWNLOAD=1``)."""
    return os.environ.get("QUBE_BOOTSTRAP_REAL_DOWNLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def bootstrap_download_should_mock(
    *,
    explicit_mock: bool = False,
) -> bool:
    """Whether splash should simulate downloads instead of fetching files.

    Mock only when ``--mock-bootstrap-download`` (or ``QUBE_BOOTSTRAP_MOCK_DOWNLOAD=1``).
    First-run consent alone does not enable mock downloads.
    """
    if bootstrap_real_download_forced():
        return False
    return explicit_mock or bootstrap_download_mock_enabled()


def settings_bootstrap_download_should_mock() -> bool:
    """Whether Settings missing-model downloads should simulate (no files written).

    Uses the same explicit mock flags as splash (``QUBE_BOOTSTRAP_MOCK_DOWNLOAD`` /
    ``--mock-bootstrap-download``), but never auto-mocks from first-run consent alone.
    """
    if bootstrap_real_download_forced():
        return False
    return bootstrap_download_mock_enabled()


def run_bootstrap_model_download(
    selected: set[BootstrapModelId],
    on_progress: DownloadProgressCallback,
) -> tuple[list[str], bool]:
    """Run real or mock bootstrap download. Returns (errors, used_mock)."""
    if settings_bootstrap_download_should_mock():
        return simulate_bootstrap_downloads(selected, on_progress=on_progress), True
    return download_bootstrap_models(selected, on_progress=on_progress), False


def mock_download_speed_multiplier() -> float:
    """Scale mock download duration (>1 finishes sooner). Defaults to 1."""
    raw = os.environ.get("QUBE_BOOTSTRAP_MOCK_DOWNLOAD_SPEED", "1").strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 1.0


def estimate_mock_download_seconds(
    selected: set[BootstrapModelId],
    *,
    bandwidth_bytes_per_sec: float = _MOCK_GOOD_BANDWIDTH_BYTES_PER_SEC,
    speed_multiplier: float | None = None,
) -> float:
    """Estimated wall time to simulate downloads for ``selected`` models."""
    total = total_selected_bytes(selected)
    if total <= 0:
        return _MOCK_MIN_MODEL_SEC
    effective_bps = bandwidth_bytes_per_sec * (
        mock_download_speed_multiplier() if speed_multiplier is None else max(0.1, speed_multiplier)
    )
    return max(_MOCK_MIN_MODEL_SEC, total / effective_bps)


def _mock_progress_filename(model_id: BootstrapModelId) -> str:
    spec = BOOTSTRAP_MODELS[model_id]
    if model_id == BootstrapModelId.WHISPER_SMALL:
        return "Whisper Small"
    if model_id == BootstrapModelId.SEARCH_PRESET_BALANCED:
        from core.embedding_modes import DEFAULT_MODE, get_mode_spec

        return get_mode_spec(DEFAULT_MODE).fastembed_model
    return spec.hf_filename or spec.label


def simulate_bootstrap_downloads(
    selected: set[BootstrapModelId],
    on_progress: DownloadProgressCallback,
    *,
    bandwidth_bytes_per_sec: float = _MOCK_GOOD_BANDWIDTH_BYTES_PER_SEC,
    speed_multiplier: float | None = None,
) -> list[str]:
    """Simulate sequential download progress without writing files."""
    multiplier = (
        mock_download_speed_multiplier() if speed_multiplier is None else max(0.1, speed_multiplier)
    )
    effective_bps = bandwidth_bytes_per_sec * multiplier
    ordered = [mid for mid in BootstrapModelId if mid in selected]
    logger.info(
        "Mock bootstrap download: %d models, ~%.1fs at %.1f MiB/s (x%.1f speed).",
        len(ordered),
        estimate_mock_download_seconds(
            selected,
            bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
            speed_multiplier=multiplier,
        ),
        effective_bps / (1024 * 1024),
        multiplier,
    )

    for model_id in ordered:
        spec = BOOTSTRAP_MODELS[model_id]
        step_label = f"Downloading {spec.label}"
        filename = _mock_progress_filename(model_id)
        size = spec.size_bytes
        duration = max(_MOCK_MIN_MODEL_SEC, size / effective_bps) if size > 0 else _MOCK_MIN_MODEL_SEC
        elapsed = 0.0
        on_progress(step_label, filename, 0, spec.source_display)
        while elapsed < duration:
            pct = min(99, int(elapsed / duration * 100))
            on_progress(step_label, filename, pct, spec.source_display)
            sleep_time = min(_MOCK_TICK_SEC, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
        on_progress(step_label, filename, 100, spec.source_display)

    return []


def resolve_model_destination(model_id: BootstrapModelId) -> Path | None:
    spec = BOOTSTRAP_MODELS.get(model_id)
    if spec is None:
        return None
    if model_id in {BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.SIDECAR_QWEN05}:
        return Path(get_cognition_models_dir()) / spec.hf_filename
    if model_id in {
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    }:
        return Path(get_llm_models_dir()) / spec.hf_filename
    if model_id == BootstrapModelId.WHISPER_SMALL:
        return Path(get_stt_models_dir())
    if model_id == BootstrapModelId.KOKORO_TTS:
        return Path(tts_default_path())
    return None


def selected_models_needing_download(
    selected: set[BootstrapModelId],
) -> set[BootstrapModelId]:
    """Return bootstrap catalogue entries that are selected but absent on disk."""
    return {mid for mid in selected if not model_is_present(mid)}


def model_is_present(model_id: BootstrapModelId) -> bool:
    if model_id == BootstrapModelId.WHISPER_SMALL:
        from core.stt_models import (
            bundled_whisper_present,
            is_protected_stt_model,
            resolve_active_stt_model_spec,
        )

        spec = resolve_active_stt_model_spec()
        if not is_protected_stt_model(spec):
            custom = Path(spec)
            return custom.is_dir() and (custom / "model.bin").is_file()
        return bundled_whisper_present()

    if model_id == BootstrapModelId.KOKORO_TTS:
        from core.tts_models import BUNDLED_DEFAULT_FILENAME, BUNDLED_VOICES_FILENAME

        base = Path(tts_default_path()).parent
        return (
            (base / BUNDLED_DEFAULT_FILENAME).is_file()
            and (base / BUNDLED_VOICES_FILENAME).is_file()
        )

    if model_id == BootstrapModelId.SEARCH_PRESET_BALANCED:
        from core.bootstrap_search_models import balanced_search_preset_present

        return balanced_search_preset_present()

    dest = resolve_model_destination(model_id)
    return dest is not None and dest.is_file()


def infer_installed_selection() -> set[BootstrapModelId] | None:
    """Infer bootstrap selection from on-disk assets for upgrade migration."""
    found = {mid for mid in BootstrapModelId if model_is_present(mid)}
    if BootstrapModelId.SIDECAR_QWEN17 in found or BootstrapModelId.SIDECAR_QWEN05 in found:
        return found
    return None


def _download_url_streaming(
    *,
    url: str,
    dest_path: Path,
    on_progress: DownloadProgressCallback,
    step_label: str,
    filename: str,
    source_display: str,
    connect_timeout_sec: float = 30.0,
    read_timeout_sec: float = 300.0,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        on_progress(step_label, filename, 100, source_display)
        return

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    on_progress(step_label, filename, 0, source_display)
    with requests.get(
        url,
        stream=True,
        timeout=(connect_timeout_sec, read_timeout_sec),
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with open(tmp_path, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(done * 100 / total)
                else:
                    pct = min(99, done // (1024 * 1024))
                on_progress(step_label, filename, pct, source_display)

    os.replace(tmp_path, dest_path)
    on_progress(step_label, filename, 100, source_display)


def _sanitize_hub_repo_file_path(name: str) -> str:
    """Repo-relative Hub path for GGUF, ONNX, tokenizer, and other preset assets."""
    n = (name or "").strip().strip("/")
    if not n or ".." in n or n.startswith("/"):
        raise ValueError("Invalid file path.")
    return n


def _download_hf_hub_file(
    *,
    repo_id: str,
    filename: str,
    dest_path: Path,
    on_progress: DownloadProgressCallback,
    step_label: str,
    source_display: str,
    progress_label: str | None = None,
) -> None:
    """Stream one file from a Hugging Face model repo into ``dest_path``."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    display_name = progress_label or dest_path.name
    if dest_path.is_file():
        on_progress(step_label, display_name, 100, source_display)
        return

    repo = _sanitize_repo_id(repo_id)
    fname = _sanitize_hub_repo_file_path(filename)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    url = hf_hub_url(repo_id=repo, filename=fname, repo_type="model")
    on_progress(step_label, display_name, 0, source_display)

    with requests.get(url, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with open(tmp_path, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024 * 4):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = int(done * 100 / total)
                else:
                    pct = min(99, done // (1024 * 1024))
                on_progress(step_label, display_name, pct, source_display)

    os.replace(tmp_path, dest_path)
    on_progress(step_label, display_name, 100, source_display)


def _download_gguf(
    *,
    repo_id: str,
    filename: str,
    dest_path: Path,
    on_progress: DownloadProgressCallback,
    step_label: str,
    source_display: str,
) -> None:
    _sanitize_repo_file_path(filename)
    _download_hf_hub_file(
        repo_id=repo_id,
        filename=filename,
        dest_path=dest_path,
        on_progress=on_progress,
        step_label=step_label,
        source_display=source_display,
    )


def _download_whisper(on_progress: DownloadProgressCallback, spec) -> None:
    from core.stt_models import bundled_whisper_present

    step_label = f"Downloading {spec.label}"
    source = spec.source_display
    filename = "Whisper Small"
    if model_is_present(BootstrapModelId.WHISPER_SMALL):
        on_progress(step_label, filename, 100, source)
        return

    dest_dir = bundled_whisper_dir()
    logger.info(
        "Downloading Whisper weights from %s into %s",
        BUNDLED_STT_HF_REPO,
        dest_dir,
    )
    on_progress(step_label, filename, 0, source)

    def _report_progress(_step: str, _file: str, percent: int, src: str) -> None:
        on_progress(step_label, filename, percent, src)

    try:
        for weight_file in BUNDLED_WHISPER_WEIGHT_FILES:
            _download_hf_hub_file(
                repo_id=BUNDLED_STT_HF_REPO,
                filename=weight_file,
                dest_path=dest_dir / weight_file,
                on_progress=_report_progress,
                step_label=step_label,
                source_display=source,
                progress_label=weight_file,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download Whisper Small from Hugging Face ({BUNDLED_STT_HF_REPO}). "
            f"Check your internet connection and try again."
        ) from exc

    if not bundled_whisper_present():
        raise RuntimeError(
            "Whisper download reported success but model files were not found under "
            f"{dest_dir}."
        )
    on_progress(step_label, filename, 100, source)


def _download_kokoro(on_progress: DownloadProgressCallback, spec) -> None:
    from core.tts_models import BUNDLED_DEFAULT_FILENAME, BUNDLED_VOICES_FILENAME

    base = Path(tts_default_path()).parent
    step_label = f"Downloading {spec.label}"
    files = (
        (
            base / BUNDLED_DEFAULT_FILENAME,
            BUNDLED_DEFAULT_FILENAME,
            "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx",
        ),
        (
            base / BUNDLED_VOICES_FILENAME,
            BUNDLED_VOICES_FILENAME,
            "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin",
        ),
    )
    for path, name, url in files:
        _download_url_streaming(
            url=url,
            dest_path=path,
            on_progress=on_progress,
            step_label=step_label,
            filename=name,
            source_display=spec.source_display,
        )


def _download_balanced_search_preset(on_progress: DownloadProgressCallback, spec) -> None:
    from core.bootstrap_search_download import download_embedding_preset
    from core.bootstrap_search_models import balanced_search_preset_present
    from core.embedding_modes import DEFAULT_MODE

    step_label = f"Downloading {spec.label}"
    if balanced_search_preset_present():
        from core.embedding_modes import get_mode_spec

        on_progress(step_label, get_mode_spec(DEFAULT_MODE).fastembed_model, 100, spec.source_display)
        return

    download_embedding_preset(
        DEFAULT_MODE,
        on_progress,
        step_label=step_label,
        source_display=spec.source_display,
    )


def download_bootstrap_models(
    selected: set[BootstrapModelId],
    on_progress: DownloadProgressCallback,
) -> list[str]:
    """Download missing assets for ``selected``. Returns human-readable errors."""
    errors: list[str] = []
    ordered = [mid for mid in BootstrapModelId if mid in selected]

    for model_id in ordered:
        spec = BOOTSTRAP_MODELS[model_id]
        step_label = f"Downloading {spec.label}"
        from core.bootstrap_trace import record_bootstrap_trace

        record_bootstrap_trace("download_model_start", model_id=model_id.value, label=spec.label)
        try:
            if model_id == BootstrapModelId.WHISPER_SMALL:
                _download_whisper(on_progress, spec)
                record_bootstrap_trace("download_model_done", model_id=model_id.value)
                continue
            if model_id == BootstrapModelId.KOKORO_TTS:
                _download_kokoro(on_progress, spec)
                record_bootstrap_trace("download_model_done", model_id=model_id.value)
                continue
            if model_id == BootstrapModelId.SEARCH_PRESET_BALANCED:
                _download_balanced_search_preset(on_progress, spec)
                record_bootstrap_trace("download_model_done", model_id=model_id.value)
                continue

            dest = resolve_model_destination(model_id)
            if dest is None:
                if not spec.hf_repo or not spec.hf_filename:
                    errors.append(f"No download source for {spec.label}.")
                    record_bootstrap_trace(
                        "download_model_skipped",
                        model_id=model_id.value,
                        reason="no_download_source",
                    )
                continue

            _download_gguf(
                repo_id=spec.hf_repo,
                filename=spec.hf_filename,
                dest_path=dest,
                on_progress=on_progress,
                step_label=step_label,
                source_display=spec.source_display,
            )
            record_bootstrap_trace("download_model_done", model_id=model_id.value)
        except Exception as exc:
            logger.exception("Bootstrap download failed for %s", spec.label)
            record_bootstrap_trace(
                "download_model_failed",
                model_id=model_id.value,
                error=str(exc),
            )
            errors.append(f"{spec.label}: {exc}")

    return errors


def format_download_detail(filename: str, percent: int, source_display: str, size_label: str = "") -> str:
    size_part = f" ({size_label})" if size_label else ""
    line = f"Downloading {filename}{size_part} — {percent}%"
    if source_display:
        line += f"\nSource: {source_display}"
    return line
