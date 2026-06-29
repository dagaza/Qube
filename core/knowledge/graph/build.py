"""Build and merge session knowledge graphs from evidence bundles."""

from __future__ import annotations

import json
from typing import Any

from core.knowledge.graph.entities import (
    entity_id_for_key,
    extract_entity_keys_from_bundle,
    extract_entity_keys_from_source,
    extract_topic_keys_from_query,
)
from core.knowledge.types import EvidenceBundle, EvidenceConflict, EvidenceObject

GRAPH_VERSION = 1


def _node(
    node_id: str,
    *,
    kind: str,
    label: str,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": node_id, "kind": kind, "label": label}
    row.update(extra)
    return row


def _edge(
    from_id: str,
    to_id: str,
    *,
    kind: str,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"from": from_id, "to": to_id, "kind": kind}
    row.update(extra)
    return row


def _source_node_id(bundle_id: str, source: EvidenceObject) -> str:
    return f"source:{bundle_id}:{source.id}"


def _query_node_id(bundle_id: str) -> str:
    return f"query:{bundle_id}"


def _entity_nodes_from_keys(keys: set[str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for key in sorted(keys):
        if not key.startswith("entity:"):
            continue
        parts = key.split(":", 2)
        entity_kind = parts[1] if len(parts) > 1 else "term"
        label = parts[2].replace("-", " ") if len(parts) > 2 else key
        nodes.append(
            _node(
                key,
                kind="entity",
                label=label,
                entity_type=entity_kind,
            )
        )
    return nodes


def _conflict_edges(
    bundle_id: str,
    sources: tuple[EvidenceObject, ...],
    conflicts: tuple[EvidenceConflict, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not conflicts or len(sources) < 2:
        return [], []

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    pos_ids = [_source_node_id(bundle_id, s) for s in sources[:2]]
    for conflict in conflicts:
        conflict_id = entity_id_for_key("conflict", conflict.topic or bundle_id)
        nodes.append(
            _node(
                conflict_id,
                kind="entity",
                label=conflict.topic or "conflict",
                entity_type="conflict",
                severity=conflict.severity,
            )
        )
        if pos_ids:
            edges.append(
                _edge(
                    pos_ids[0],
                    conflict_id,
                    kind="conflicts",
                    topic=conflict.topic,
                    severity=conflict.severity,
                )
            )
        if len(pos_ids) > 1:
            edges.append(
                _edge(
                    pos_ids[1],
                    conflict_id,
                    kind="conflicts",
                    topic=conflict.topic,
                    severity=conflict.severity,
                )
            )
    return nodes, edges


def build_graph_from_bundle(
    bundle: EvidenceBundle,
    *,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Derive a graph fragment for one evidence bundle."""
    bundle_id = bundle.bundle_id
    query_id = _query_node_id(bundle_id)
    nodes: list[dict[str, Any]] = [
        _node(
            query_id,
            kind="query",
            label=bundle.query_resolved or bundle.query_raw,
            bundle_id=bundle_id,
            knowledge_service=bundle.knowledge_service,
            message_id=message_id,
        )
    ]
    edges: list[dict[str, Any]] = []

    entity_keys: set[str] = set(extract_entity_keys_from_bundle(bundle))
    entity_keys.update(extract_topic_keys_from_query(bundle.query_resolved))
    nodes.extend(_entity_nodes_from_keys(entity_keys))

    for topic_key in sorted(extract_topic_keys_from_query(bundle.query_resolved)):
        edges.append(_edge(query_id, topic_key, kind="about"))

    for source in bundle.sources:
        source_id = _source_node_id(bundle_id, source)
        nodes.append(
            _node(
                source_id,
                kind="source",
                label=source.title or source.id,
                bundle_id=bundle_id,
                evidence_id=source.id,
                adapter=source.adapter,
                doi=source.doi,
                document_type=source.document_type,
            )
        )
        edges.append(_edge(query_id, source_id, kind="supports"))
        for entity_key in sorted(extract_entity_keys_from_source(source)):
            edges.append(_edge(source_id, entity_key, kind="mentions"))

    conflict_nodes, conflict_edges = _conflict_edges(
        bundle_id, bundle.sources, bundle.conflicts
    )
    nodes.extend(conflict_nodes)
    edges.extend(conflict_edges)

    return _normalize_graph({"version": GRAPH_VERSION, "nodes": nodes, "edges": edges})


def _normalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_rows = [n for n in nodes if isinstance(n, dict) and n.get("id")]
    edge_rows = [e for e in edges if isinstance(e, dict) and e.get("from") and e.get("to")]

    dedup_nodes: dict[str, dict[str, Any]] = {}
    for node in sorted(node_rows, key=lambda n: str(n["id"])):
        dedup_nodes[str(node["id"])] = node

    dedup_edges: dict[tuple[str, ...], dict[str, Any]] = {}
    for edge in edge_rows:
        key = (
            str(edge["from"]),
            str(edge["to"]),
            str(edge.get("kind") or ""),
            str(edge.get("topic") or ""),
        )
        dedup_edges[key] = edge

    return {
        "version": int(graph.get("version") or GRAPH_VERSION),
        "nodes": [dedup_nodes[k] for k in sorted(dedup_nodes)],
        "edges": [dedup_edges[k] for k in sorted(dedup_edges)],
    }


def merge_graphs(existing: dict[str, Any] | None, delta: dict[str, Any]) -> dict[str, Any]:
    """Merge graph fragments; node/edge ids dedupe deterministically."""
    base = _normalize_graph(existing or {"version": GRAPH_VERSION, "nodes": [], "edges": []})
    incoming = _normalize_graph(delta)
    return _normalize_graph(
        {
            "version": GRAPH_VERSION,
            "nodes": (base.get("nodes") or []) + (incoming.get("nodes") or []),
            "edges": (base.get("edges") or []) + (incoming.get("edges") or []),
        }
    )


def graph_to_json(graph: dict[str, Any]) -> str:
    normalized = _normalize_graph(graph)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def graph_from_json(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        return {"version": GRAPH_VERSION, "nodes": [], "edges": []}
    return _normalize_graph(parsed)


def subgraph_for_bundle(graph: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    """Return nodes/edges tied to a specific bundle (for focused Research map view)."""
    normalized = _normalize_graph(graph)
    bundle_id = str(bundle_id or "")
    node_ids = {
        str(n["id"])
        for n in normalized.get("nodes") or []
        if isinstance(n, dict)
        and (
            str(n.get("bundle_id") or "") == bundle_id
            or str(n.get("id") or "") == f"query:{bundle_id}"
        )
    }
    for edge in normalized.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if str(edge.get("from") or "").startswith(f"query:{bundle_id}"):
            node_ids.add(str(edge["from"]))
            node_ids.add(str(edge.get("to") or ""))
        if str(edge.get("from") or "").startswith(f"source:{bundle_id}:"):
            node_ids.add(str(edge["from"]))
            node_ids.add(str(edge.get("to") or ""))

    nodes = [
        n
        for n in normalized.get("nodes") or []
        if isinstance(n, dict) and str(n.get("id") or "") in node_ids
    ]
    edges = [
        e
        for e in normalized.get("edges") or []
        if isinstance(e, dict)
        and str(e.get("from") or "") in node_ids
        and str(e.get("to") or "") in node_ids
    ]
    return _normalize_graph({"version": GRAPH_VERSION, "nodes": nodes, "edges": edges})
