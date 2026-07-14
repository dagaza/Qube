"""Fastembed backend batch parsing tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from core.embedding_modes import get_mode_spec
from rag.backends.fastembed_backend import FastembedBackend


class _FakeFastembedModel:
    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, texts, batch_size=8):
        for index, _text in enumerate(texts):
            base = np.arange(self.dim, dtype=np.float32)
            yield base + float(index + 1)


def test_embed_texts_treats_each_yield_as_one_vector():
    spec = get_mode_spec("balanced")
    backend = FastembedBackend.__new__(FastembedBackend)
    backend._spec = spec
    backend.vector_dim = spec.vector_dim
    backend._model = _FakeFastembedModel(spec.vector_dim)

    vectors = backend._embed_texts(["alpha", "beta"], is_query=False)

    assert vectors.shape == (2, spec.vector_dim)
    assert abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(vectors[1])) - 1.0) < 1e-5
    assert not np.allclose(vectors[0], vectors[1])


def test_embed_query_returns_single_normalized_vector():
    spec = get_mode_spec("fast")
    backend = FastembedBackend.__new__(FastembedBackend)
    backend._spec = spec
    backend.vector_dim = spec.vector_dim
    backend._model = _FakeFastembedModel(spec.vector_dim)

    vector = backend.embed_query("hello")

    assert vector.shape == (spec.vector_dim,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


@patch("rag.backends.fastembed_backend.FastembedBackend._load")
def test_power_mode_embed_query_integration(mock_load):
    mock_load.return_value = None
    backend = FastembedBackend(get_mode_spec("power"))
    backend._model = _FakeFastembedModel(backend.vector_dim)

    vector = backend.embed_query("What is this about?")

    assert vector.shape == (1024,)
