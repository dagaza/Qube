"""Resolve bootstrap download sizes from Hugging Face (source of truth when online)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import requests
from huggingface_hub import hf_hub_url

from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId

logger = logging.getLogger("Qube.Bootstrap.HFMetadata")

_KOKORO_FILES: tuple[tuple[str, str], ...] = (
    ("hexgrad/Kokoro-82M", "kokoro-v1.0.onnx"),
    ("hexgrad/Kokoro-82M", "voices-v1.0.bin"),
)


class BootstrapSizeSource(StrEnum):
    HUGGINGFACE = "huggingface"
    ESTIMATE = "estimate"


@dataclass(frozen=True)
class ResolvedBootstrapSize:
    model_id: BootstrapModelId
    size_bytes: int
    source: BootstrapSizeSource
    detail: str = ""


def _fetch_hf_file_size_bytes(repo_id: str, filename: str) -> int | None:
    try:
        from huggingface_hub import get_hf_file_metadata

        meta = get_hf_file_metadata(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        if meta.size is not None and int(meta.size) > 0:
            return int(meta.size)
    except Exception as exc:
        logger.debug("get_hf_file_metadata failed for %s/%s: %s", repo_id, filename, exc)

    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")
        with requests.head(url, timeout=(10, 30), allow_redirects=True) as resp:
            if resp.status_code == 200:
                raw = resp.headers.get("content-length")
                if raw:
                    return int(raw)
    except Exception as exc:
        logger.debug("HEAD size probe failed for %s/%s: %s", repo_id, filename, exc)
    return None


def resolve_bootstrap_size(model_id: BootstrapModelId) -> ResolvedBootstrapSize:
    spec = BOOTSTRAP_MODELS[model_id]
    fallback = int(spec.size_bytes)

    if model_id == BootstrapModelId.KOKORO_TTS:
        total = 0
        found = 0
        for repo, fname in _KOKORO_FILES:
            sz = _fetch_hf_file_size_bytes(repo, fname)
            if sz is not None:
                total += sz
                found += 1
        if found == len(_KOKORO_FILES):
            return ResolvedBootstrapSize(
                model_id=model_id,
                size_bytes=total,
                source=BootstrapSizeSource.HUGGINGFACE,
                detail="hexgrad/Kokoro-82M (onnx + voices)",
            )
        return ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=fallback,
            source=BootstrapSizeSource.ESTIMATE,
            detail="Kokoro bundle (offline estimate)",
        )

    if model_id == BootstrapModelId.WHISPER_SMALL:
        # faster-whisper pulls a CTranslate2 tree; published footprint is approximate.
        return ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=fallback,
            source=BootstrapSizeSource.ESTIMATE,
            detail="Systran/faster-whisper-small cache footprint (approximate)",
        )

    if model_id == BootstrapModelId.SEARCH_PRESET_BALANCED:
        from core.embedding_modes import DEFAULT_MODE, get_mode_spec

        mode_spec = get_mode_spec(DEFAULT_MODE)
        return ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=fallback,
            source=BootstrapSizeSource.ESTIMATE,
            detail=f"fastembed ONNX preset ({mode_spec.fastembed_model})",
        )

    if not spec.hf_repo or not spec.hf_filename:
        return ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=fallback,
            source=BootstrapSizeSource.ESTIMATE,
            detail="No Hugging Face GGUF mapping",
        )

    live = _fetch_hf_file_size_bytes(spec.hf_repo, spec.hf_filename)
    if live is not None and live > 0:
        return ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=live,
            source=BootstrapSizeSource.HUGGINGFACE,
            detail=f"{spec.hf_repo}/{spec.hf_filename}",
        )

    return ResolvedBootstrapSize(
        model_id=model_id,
        size_bytes=fallback,
        source=BootstrapSizeSource.ESTIMATE,
        detail=f"Offline estimate for {spec.hf_filename}",
    )


def resolve_all_bootstrap_sizes() -> dict[BootstrapModelId, ResolvedBootstrapSize]:
    return {model_id: resolve_bootstrap_size(model_id) for model_id in BootstrapModelId}


def size_map(resolved: dict[BootstrapModelId, ResolvedBootstrapSize]) -> dict[BootstrapModelId, int]:
    return {mid: entry.size_bytes for mid, entry in resolved.items()}


def format_bootstrap_size_tag_tooltip(entry: ResolvedBootstrapSize) -> str:
    """Hover text for Verified / Estimated chips in the bootstrap consent UI."""
    detail = entry.detail.strip() or "catalogue entry"
    if entry.source is BootstrapSizeSource.HUGGINGFACE:
        return (
            "Verified — exact download size\n"
            "We checked this file on Hugging Face, so the size shown should match "
            "what you download.\n\n"
            f"Technical: confirmed via Hugging Face file metadata ({detail}). "
            "Disk totals and feasibility use this byte count."
        )
    if detail.lower().startswith("loading"):
        return (
            "Estimated — checking online\n"
            "We're still contacting Hugging Face for exact file sizes. Totals use "
            "built-in estimates until verification finishes.\n\n"
            "Technical: offline catalogue sizes are shown while metadata loads."
        )
    return (
        "Estimated — approximate size\n"
        "We couldn't verify this file online (no connection, blocked access, or no "
        "direct file mapping). The size is a planning estimate and may differ slightly "
        "from the real download.\n\n"
        f"Technical: offline estimate ({detail}). Voice models (Whisper/Kokoro) use "
        "approximate cache footprints. Connect to the internet for live Hugging Face sizes."
    )
