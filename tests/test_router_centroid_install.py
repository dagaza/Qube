"""Router centroid lifecycle tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from core.router_centroid_install import clear_router_embedding_state, install_router_centroids
from mcp.cognitive_router import CognitiveRouterV4
from mcp.router_lane_stats import LaneStatsRegistry, RouteFeedbackEvent


class _FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_clear_router_embedding_state_resets_centroids_and_tracker():
    router = CognitiveRouterV4()
    router.set_recall_centroid(np.array([1.0, 0.0], dtype=np.float32))
    router.last_intent_vector = np.array([0.5, 0.5], dtype=np.float32)
    router.stability_tracker.observe(np.array([1.0, 0.0], dtype=np.float32), "rag")
    seeded_stats = router.lane_stats
    router.lane_stats.update(
        RouteFeedbackEvent(
            route="memory",
            top_intent="memory",
            top_source="embedding",
            confidence_margin=0.2,
            latency_ms=1.0,
            success=True,
            drift=False,
            per_lane_hits={"memory": 1},
        )
    )

    clear_router_embedding_state(router)

    assert router.recall_centroid is None
    assert router.last_intent_vector is None
    assert router.stability_tracker._dim is None
    assert router.lane_stats is not seeded_stats
    assert isinstance(router.lane_stats, LaneStatsRegistry)


def test_install_router_centroids_builds_all_five():
    router = CognitiveRouterV4()
    install_router_centroids(router, _FakeEmbedder(), force=True)

    assert router.recall_centroid is not None
    assert router.chat_centroid is not None
    assert router.memory_centroid is not None
    assert router.rag_centroid is not None
    assert router.web_centroid is not None


def test_install_router_centroids_skips_when_present_without_force():
    router = CognitiveRouterV4()
    existing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    router.set_recall_centroid(existing)
    router.set_chat_centroid(existing)
    router.set_memory_centroid(existing)
    router.set_rag_centroid(existing)
    router.set_web_centroid(existing)
    embedder = MagicMock()
    embedder.embed_query.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    install_router_centroids(router, embedder, force=False)

    assert np.allclose(router.rag_centroid, existing)
    embedder.embed_query.assert_not_called()
