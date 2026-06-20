"""Embedding mode registry tests."""
from __future__ import annotations

from core.embedding_modes import (
    DEFAULT_MODE,
    EMBEDDING_MODES,
    get_mode_spec,
    normalize_mode_id,
)


def test_default_mode_is_balanced():
    assert DEFAULT_MODE == "balanced"
    assert normalize_mode_id(None) == "balanced"
    assert normalize_mode_id("unknown") == "balanced"


def test_mode_dimensions():
    assert get_mode_spec("fast").vector_dim == 384
    assert get_mode_spec("balanced").vector_dim == 512
    assert get_mode_spec("power").vector_dim == 1024


def test_all_modes_registered():
    assert set(EMBEDDING_MODES) == {"fast", "balanced", "power"}


def test_all_mode_models_are_fastembed_supported():
    from fastembed import TextEmbedding

    supported = {entry["model"] for entry in TextEmbedding.list_supported_models()}
    for spec in EMBEDDING_MODES.values():
        assert spec.fastembed_model in supported, spec.mode_id
        listed_dim = next(
            entry["dim"]
            for entry in TextEmbedding.list_supported_models()
            if entry["model"] == spec.fastembed_model
        )
        assert spec.vector_dim == listed_dim, spec.mode_id
