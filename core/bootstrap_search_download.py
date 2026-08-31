"""Streamed downloads for Fast/Balanced/Power fastembed ONNX presets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pathlib import Path

from core.embedding_modes import ModeId, get_mode_spec, normalize_mode_id
from core.paths import configure_user_model_paths, search_models_cache_dir

logger = logging.getLogger("Qube.Bootstrap.SearchDownload")

DownloadProgressCallback = Callable[[str, str, int, str], None]

_STANDARD_TOKENIZER_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


@dataclass(frozen=True)
class FastembedPresetDownloadSpec:
    hf_repo: str
    files: tuple[str, ...]
    model_marker: str


_PRESET_DOWNLOAD_SPECS: dict[ModeId, FastembedPresetDownloadSpec] = {
    "fast": FastembedPresetDownloadSpec(
        hf_repo="qdrant/bge-small-en-v1.5-onnx-q",
        files=_STANDARD_TOKENIZER_FILES + ("model_optimized.onnx",),
        model_marker="model_optimized.onnx",
    ),
    "balanced": FastembedPresetDownloadSpec(
        hf_repo="xenova/jina-embeddings-v2-small-en",
        files=_STANDARD_TOKENIZER_FILES + ("onnx/model.onnx",),
        model_marker="onnx/model.onnx",
    ),
    "power": FastembedPresetDownloadSpec(
        hf_repo="qdrant/bge-large-en-v1.5-onnx",
        files=_STANDARD_TOKENIZER_FILES + ("model.onnx",),
        model_marker="model.onnx",
    ),
}


def _dedupe_files(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for group in groups for path in group))


def _try_fastembed_download_spec(mode_id: ModeId) -> FastembedPresetDownloadSpec | None:
    """Resolve preset files from fastembed when installed (keeps us aligned with library)."""
    try:
        from fastembed import TextEmbedding

        model_name = get_mode_spec(mode_id).fastembed_model
        for entry in TextEmbedding.list_supported_models():
            if entry.get("model") != model_name:
                continue
            sources = entry.get("sources") or {}
            hf_repo = sources.get("hf")
            if not hf_repo:
                return None
            model_file = str(entry.get("model_file") or "onnx/model.onnx")
            additional = tuple(str(path) for path in (entry.get("additional_files") or ()))
            files = _dedupe_files(_STANDARD_TOKENIZER_FILES, (model_file,), additional)
            return FastembedPresetDownloadSpec(
                hf_repo=str(hf_repo),
                files=files,
                model_marker=model_file,
            )
    except Exception as exc:
        logger.debug("Could not resolve fastembed download spec for %s: %s", mode_id, exc)
    return None


def resolve_preset_download_spec(mode_id: str | None = None) -> FastembedPresetDownloadSpec:
    mode = normalize_mode_id(mode_id)
    dynamic = _try_fastembed_download_spec(mode)
    if dynamic is not None:
        return dynamic
    return _PRESET_DOWNLOAD_SPECS[mode]


def qube_preset_dir(mode_id: str | None = None) -> Path:
    configure_user_model_paths()
    return search_models_cache_dir() / "presets" / normalize_mode_id(mode_id)


def qube_preset_complete(mode_id: str | None = None) -> bool:
    spec = resolve_preset_download_spec(mode_id)
    base = qube_preset_dir(mode_id)
    marker = base / spec.model_marker
    if not marker.is_file():
        return False
    return all((base / rel_path).is_file() for rel_path in spec.files)


def resolve_qube_preset_path(mode_id: str | None = None) -> str | None:
    """Return the on-disk preset directory when a streamed download is complete."""
    if qube_preset_complete(mode_id):
        return str(qube_preset_dir(mode_id))
    return None


def download_embedding_preset(
    mode_id: str | None,
    on_progress: DownloadProgressCallback,
    *,
    step_label: str | None = None,
    source_display: str = "Hugging Face",
) -> None:
    """Download ONNX + tokenizer assets with streamed HTTP progress."""
    from core.bootstrap_download import _download_hf_hub_file

    mode = normalize_mode_id(mode_id)
    mode_spec = get_mode_spec(mode)
    dl_spec = resolve_preset_download_spec(mode)
    filename = mode_spec.fastembed_model
    label = step_label or f"Downloading {mode_spec.label} search preset"

    if qube_preset_complete(mode):
        on_progress(label, filename, 100, source_display)
        return

    from core.bootstrap_search_models import clear_search_preset_incomplete_cache

    clear_search_preset_incomplete_cache(mode)

    base = qube_preset_dir(mode)
    base.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading search preset mode=%s model=%s repo=%s into %s",
        mode,
        filename,
        dl_spec.hf_repo,
        base,
    )
    on_progress(label, filename, 0, source_display)

    def _report(_step: str, _file: str, percent: int, src: str) -> None:
        on_progress(label, filename, percent, src)

    try:
        for repo_file in dl_spec.files:
            dest = base / repo_file
            from core.bootstrap_trace import record_bootstrap_trace

            record_bootstrap_trace(
                "preset_file_start",
                mode=mode,
                repo=dl_spec.hf_repo,
                repo_file=repo_file,
                dest=str(dest),
                already_present=dest.is_file(),
            )
            if dest.is_file():
                continue
            _download_hf_hub_file(
                repo_id=dl_spec.hf_repo,
                filename=repo_file,
                dest_path=dest,
                on_progress=_report,
                step_label=label,
                source_display=source_display,
                progress_label=Path(repo_file).name,
            )
            record_bootstrap_trace(
                "preset_file_done",
                mode=mode,
                repo_file=repo_file,
                bytes=dest.stat().st_size if dest.is_file() else 0,
            )
    except Exception as exc:
        from core.bootstrap_trace import record_bootstrap_trace

        record_bootstrap_trace(
            "preset_download_failed",
            mode=mode,
            repo=dl_spec.hf_repo,
            error=str(exc),
        )
        from core.bootstrap_search_models import format_search_preset_download_failure

        raise RuntimeError(format_search_preset_download_failure(mode)) from exc

    if not qube_preset_complete(mode):
        from core.bootstrap_search_models import format_search_preset_download_failure

        raise RuntimeError(format_search_preset_download_failure(mode))

    on_progress(label, filename, 100, source_display)


def download_embedding_preset_no_progress(mode_id: str | None) -> None:
    """Settings/background downloads without a splash progress callback."""

    def _noop(*_args: object) -> None:
        return None

    download_embedding_preset(mode_id, _noop)
