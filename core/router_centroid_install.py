"""Shared cognitive-router centroid lifecycle helpers."""
from __future__ import annotations

import logging
from typing import Any

from core.router_centroid_examples import (
    CHAT_INTENT_EXAMPLES,
    MEMORY_INTENT_EXAMPLES,
    RAG_INTENT_EXAMPLES,
    RECALL_INTENT_EXAMPLES,
    WEB_INTENT_EXAMPLES,
)
from mcp.cognitive_router import CognitiveRouterV4
from mcp.router_lane_stats import LaneStatsRegistry

logger = logging.getLogger("Qube.RouterCentroids")


def clear_router_embedding_state(router: CognitiveRouterV4) -> None:
    """Drop embedding-bound router state after an embedder swap or reindex."""
    router.recall_centroid = None
    router.chat_centroid = None
    router.memory_centroid = None
    router.rag_centroid = None
    router.web_centroid = None
    router.last_intent_vector = None
    router.stability_tracker.reset()
    router.lane_stats = LaneStatsRegistry()


def install_router_centroids(
    router: CognitiveRouterV4,
    embedder: Any,
    *,
    force: bool = False,
) -> None:
    """Build and install semantic centroids from the active embedder."""
    from workers.intent_router import build_centroid

    def _maybe_set(current, setter, examples, label: str) -> None:
        if not force and current is not None:
            return
        try:
            setter(build_centroid(embedder, list(examples)))
            logger.info("[RouterCentroids] %s centroid installed.", label)
        except Exception:
            logger.exception("[RouterCentroids] Failed to install %s centroid.", label)

    _maybe_set(
        router.recall_centroid,
        router.set_recall_centroid,
        RECALL_INTENT_EXAMPLES,
        "recall",
    )
    _maybe_set(
        router.chat_centroid,
        router.set_chat_centroid,
        CHAT_INTENT_EXAMPLES,
        "chat",
    )
    _maybe_set(
        router.memory_centroid,
        router.set_memory_centroid,
        MEMORY_INTENT_EXAMPLES,
        "memory",
    )
    _maybe_set(
        router.rag_centroid,
        router.set_rag_centroid,
        RAG_INTENT_EXAMPLES,
        "rag",
    )
    _maybe_set(
        router.web_centroid,
        router.set_web_centroid,
        WEB_INTENT_EXAMPLES,
        "web",
    )
