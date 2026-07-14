"""Session-local knowledge graph derived from evidence bundles (Phase 6 Slice 4)."""

from core.knowledge.graph.build import (
    build_graph_from_bundle,
    graph_from_json,
    graph_to_json,
    merge_graphs,
)
from core.knowledge.graph.bundle_codec import bundle_from_dict, bundle_to_dict
from core.knowledge.graph.service import (
    find_prior_bundles_by_entities,
    record_bundle_in_session_graph,
)

__all__ = [
    "build_graph_from_bundle",
    "bundle_from_dict",
    "bundle_to_dict",
    "find_prior_bundles_by_entities",
    "graph_from_json",
    "graph_to_json",
    "merge_graphs",
    "record_bundle_in_session_graph",
]
