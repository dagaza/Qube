"""Reindex worker unit tests (mocked embedder/store)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.reindex_state import is_reindex_in_progress, set_reindex_in_progress
from mcp.cognitive_router import CognitiveRouterV4
from workers.reindex_worker import ReindexWorker


@pytest.fixture(autouse=True)
def _reset_reindex_flag():
    set_reindex_in_progress(False)
    yield
    set_reindex_in_progress(False)


def test_reindex_worker_reembeds_exported_rows_and_resets_router():
    store = MagicMock()
    store.export_all_rows.return_value = [
        {"text": "hello world", "source": "doc.txt", "chunk_id": 0},
        {"text": "memory fact", "source": "qube_memory::pref::1", "chunk_id": 0},
    ]

    embedder = MagicMock()
    embedder.vector_dim = 512
    embedder.embed.return_value = np.array(
        [[0.1] * 512, [0.2] * 512],
        dtype=np.float32,
    )
    embedder.reload = MagicMock()

    router = CognitiveRouterV4()
    router.set_rag_centroid(np.array([1.0, 0.0], dtype=np.float32))

    worker = ReindexWorker(
        embedder=embedder,
        store=store,
        cognitive_router=router,
        target_mode="balanced",
        reload_embedder=True,
    )

    with patch(
        "workers.reindex_worker.install_router_centroids",
    ) as install_mock, patch(
        "workers.reindex_worker.clear_router_embedding_state",
    ) as clear_mock:
        worker.run()

    embedder.reload.assert_called_once_with(mode_id="balanced")
    store.recreate_for_dim.assert_called_once_with(512)
    store.rebuild_fts_index.assert_called_once()
    assert store.add_chunks.call_count >= 1
    clear_mock.assert_called_once_with(router)
    install_mock.assert_called_once()
    assert is_reindex_in_progress() is False


def test_reindex_worker_clears_flag_on_failure():
    store = MagicMock()
    store.export_all_rows.side_effect = RuntimeError("boom")

    worker = ReindexWorker(
        embedder=MagicMock(vector_dim=512),
        store=store,
        reload_embedder=False,
    )

    errors: list[str] = []
    worker.error_occurred.connect(errors.append)
    worker.run()

    assert errors == ["boom"]
    assert is_reindex_in_progress() is False
