"""Copy a filtered session knowledge graph when branching a conversation."""

from __future__ import annotations

from typing import Any

from core.knowledge.graph.build import merge_graphs, subgraph_for_bundle
from core.knowledge.graph.service import load_session_graph


def _bundle_ids_for_branch_prefix(
    parent_graph: dict[str, Any],
    prefix_messages: list[dict],
) -> set[str]:
    old_message_ids = {
        str(row.get("id") or "").strip()
        for row in prefix_messages
        if str(row.get("id") or "").strip()
    }
    bundle_ids: set[str] = set()
    for row in prefix_messages:
        bid = str(row.get("evidence_bundle_id") or "").strip()
        if bid:
            bundle_ids.add(bid)

    for node in parent_graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if str(node.get("kind") or "") != "query":
            continue
        mid = str(node.get("message_id") or "").strip()
        if mid and mid in old_message_ids:
            bid = str(node.get("bundle_id") or "").strip()
            if bid:
                bundle_ids.add(bid)
    return bundle_ids


def _remap_graph_message_ids(
    graph: dict[str, Any], message_id_map: dict[str, str]
) -> dict[str, Any]:
    if not message_id_map:
        return graph
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        mid = node.get("message_id")
        if mid is None:
            continue
        key = str(mid).strip()
        if key in message_id_map:
            node["message_id"] = message_id_map[key]
    return graph


def build_branch_session_graph(
    db: Any,
    *,
    source_session_id: str,
    prefix_messages: list[dict],
    message_id_map: dict[str, str],
) -> dict[str, Any] | None:
    """
    Build a knowledge graph fragment for a branched prefix: union of per-bundle
    subgraphs for bundles referenced by copied messages (and query nodes in range).
    """
    sid = str(source_session_id or "").strip()
    if not sid or not prefix_messages:
        return None

    parent = load_session_graph(db, sid)
    if not parent.get("nodes"):
        return None

    bundle_ids = _bundle_ids_for_branch_prefix(parent, prefix_messages)
    if not bundle_ids:
        return None

    merged: dict[str, Any] | None = None
    for bid in sorted(bundle_ids):
        fragment = subgraph_for_bundle(parent, bid)
        if not fragment.get("nodes"):
            continue
        merged = fragment if merged is None else merge_graphs(merged, fragment)

    if merged is None:
        return None
    return _remap_graph_message_ids(merged, message_id_map)


def copy_branch_session_graph(
    db: Any,
    *,
    source_session_id: str,
    target_session_id: str,
    prefix_messages: list[dict],
    message_id_map: dict[str, str],
) -> bool:
    """Persist a branch-filtered graph on ``target_session_id`` when non-empty."""
    from core.knowledge.graph.build import graph_to_json

    graph = build_branch_session_graph(
        db,
        source_session_id=source_session_id,
        prefix_messages=prefix_messages,
        message_id_map=message_id_map,
    )
    if graph is None:
        return False
    tid = str(target_session_id or "").strip()
    if not tid:
        return False
    db.save_session_knowledge_graph(tid, graph_to_json(graph))
    return True
