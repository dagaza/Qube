"""Embedding mode presets — one active semantic space at a time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModeId = Literal["fast", "balanced", "power"]

DEFAULT_MODE: ModeId = "balanced"
MODE_IDS: tuple[ModeId, ...] = ("fast", "balanced", "power")


@dataclass(frozen=True)
class EmbeddingModeSpec:
    mode_id: ModeId
    label: str
    short_description: str
    fastembed_model: str
    vector_dim: int


EMBEDDING_MODES: dict[ModeId, EmbeddingModeSpec] = {
    "fast": EmbeddingModeSpec(
        mode_id="fast",
        label="Fast",
        short_description="Quickest searches, lighter on memory",
        fastembed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384,
    ),
    "balanced": EmbeddingModeSpec(
        mode_id="balanced",
        label="Balanced",
        short_description="Recommended balance of speed and quality",
        fastembed_model="jinaai/jina-embeddings-v2-small-en",
        vector_dim=512,
    ),
    "power": EmbeddingModeSpec(
        mode_id="power",
        label="Power",
        short_description="Best retrieval quality, uses more memory",
        # bge-m3 is not available in fastembed; bge-large-en-v1.5 is the
        # supported 1024-d ONNX preset in the same model family.
        fastembed_model="BAAI/bge-large-en-v1.5",
        vector_dim=1024,
    ),
}


def normalize_mode_id(value: str | None) -> ModeId:
    key = str(value or "").strip().lower()
    if key in EMBEDDING_MODES:
        return key  # type: ignore[return-value]
    return DEFAULT_MODE


def get_mode_spec(mode_id: str | None = None) -> EmbeddingModeSpec:
    return EMBEDDING_MODES[normalize_mode_id(mode_id)]


def list_mode_specs() -> list[EmbeddingModeSpec]:
    return [EMBEDDING_MODES[m] for m in MODE_IDS]
