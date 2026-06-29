"""Persist and query session knowledge graphs."""

from __future__ import annotations

import logging
from typing import Any

from core.knowledge.graph.build import (
    build_graph_from_bundle,
    graph_from_json,
    graph_to_json,
    merge_graphs,
)
from core.knowledge.graph.bundle_codec import bundle_to_dict
from core.knowledge.graph.entities import extract_entity_keys_from_bundle
from core.knowledge.types import EvidenceBundle

logger = logging.getLogger("Qube.KnowledgeGraph")


def record_bundle_in_session_graph(
    db: Any,
    *,
    session_id: str,
    bundle: EvidenceBundle,
    message_id: str | None = None,
) -> dict[str, Any] | None:
    """Append bundle nodes/edges to the session graph and store a bundle snapshot."""
    sid = str(session_id or "").strip()
    if not sid or bundle is None or not bundle.sources:
        return None
    try:
        delta = build_graph_from_bundle(bundle, message_id=message_id)
        existing = db.get_session_knowledge_graph(sid)
        merged = merge_graphs(existing, delta)
        db.save_session_knowledge_graph(sid, graph_to_json(merged))
        entity_keys = extract_entity_keys_from_bundle(bundle)
        db.save_evidence_bundle_snapshot(
            bundle_id=bundle.bundle_id,
            session_id=sid,
            message_id=message_id,
            query_resolved=bundle.query_resolved,
            knowledge_service=bundle.knowledge_service,
            entity_keys=entity_keys,
            bundle_json=graph_to_json({"bundle": bundle_to_dict(bundle)}),
        )
        return merged
    except Exception:
        logger.debug("Failed to record knowledge graph for session %s", sid, exc_info=True)
        return None


def find_prior_bundles_by_entities(
    db: Any,
    *,
    entity_keys: tuple[str, ...],
    exclude_session_id: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Read-only cross-session suggestions sharing entity keys."""
    keys = {k for k in entity_keys if k}
    if not keys:
        return []
    try:
        return db.find_evidence_bundle_snapshots_by_entities(
            entity_keys=keys,
            exclude_session_id=exclude_session_id,
            limit=limit,
        )
    except Exception:
        logger.debug("Prior bundle lookup failed", exc_info=True)
        return []


def load_session_graph(db: Any, session_id: str) -> dict[str, Any]:
    sid = str(session_id or "").strip()
    if not sid:
        return {"version": 1, "nodes": [], "edges": []}
    try:
        raw = db.get_session_knowledge_graph_json(sid)
        if not raw:
            return {"version": 1, "nodes": [], "edges": []}
        return graph_from_json(raw)
    except Exception:
        logger.debug("Failed to load session graph %s", sid, exc_info=True)
        return {"version": 1, "nodes": [], "edges": []}


def export_session_graph(db: Any, session_id: str) -> dict[str, Any]:
    return load_session_graph(db, session_id)


def import_session_graph(db: Any, session_id: str, graph: dict[str, Any]) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    db.save_session_knowledge_graph(sid, graph_to_json(graph))
