"""Tests for session knowledge graph (Phase 6 Slice 4)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.database import DatabaseManager  # noqa: E402
from core.knowledge.bundle_builder import build_general_web_bundle  # noqa: E402
from core.knowledge.graph.build import (  # noqa: E402
    build_graph_from_bundle,
    graph_from_json,
    graph_to_json,
    merge_graphs,
)
from core.knowledge.graph.bundle_codec import bundle_from_dict, bundle_to_dict  # noqa: E402
from core.knowledge.graph.service import (  # noqa: E402
    export_session_graph,
    import_session_graph,
    record_bundle_in_session_graph,
)
from core.knowledge.types import EvidenceConflict  # noqa: E402


def _sample_bundle():
    return build_general_web_bundle(
        query_raw="ACE inhibitors heart failure evidence",
        query_resolved="ACE inhibitors heart failure evidence",
        kept_rows=[
            {
                "title": "ACE inhibitors in HF — PubMed review",
                "snippet": "ACE inhibitors reduce mortality in heart failure.",
                "url": "https://example.org/ace-hf",
                "doi": "10.1000/ace.hf.review",
                "_web_token_overlap": 0.55,
            },
            {
                "title": "Conflicting observational HF study",
                "snippet": "No significant benefit from ACE inhibitors in this cohort.",
                "url": "https://example.org/hf-negative",
                "_web_token_overlap": 0.41,
            },
        ],
        rejected_count=0,
        latency_ms=42.0,
    )


class TestKnowledgeGraph(unittest.TestCase):
    def test_bundle_codec_round_trip(self) -> None:
        bundle = _sample_bundle()
        restored = bundle_from_dict(bundle_to_dict(bundle))
        self.assertEqual(restored.bundle_id, bundle.bundle_id)
        self.assertEqual(len(restored.sources), len(bundle.sources))
        self.assertEqual(restored.sources[0].title, bundle.sources[0].title)

    def test_graph_golden_deterministic(self) -> None:
        bundle = _sample_bundle()
        bundle = bundle_from_dict(bundle_to_dict(bundle))
        graph_a = build_graph_from_bundle(bundle)
        graph_b = build_graph_from_bundle(bundle_from_dict(bundle_to_dict(bundle)))
        self.assertEqual(graph_to_json(graph_a), graph_to_json(graph_b))
        self.assertGreaterEqual(len(graph_a["nodes"]), 3)
        self.assertGreaterEqual(len(graph_a["edges"]), 2)
        kinds = {n["kind"] for n in graph_a["nodes"]}
        self.assertIn("query", kinds)
        self.assertIn("source", kinds)

    def test_merge_graphs_two_bundles(self) -> None:
        b1 = _sample_bundle()
        b2 = build_general_web_bundle(
            query_raw="SGLT2 inhibitors heart failure",
            query_resolved="SGLT2 inhibitors heart failure",
            kept_rows=[
                {
                    "title": "SGLT2 meta-analysis",
                    "snippet": "SGLT2 inhibitors improve outcomes in HF.",
                    "url": "https://example.org/sglt2",
                    "_web_token_overlap": 0.5,
                }
            ],
            rejected_count=0,
            latency_ms=10.0,
        )
        merged = merge_graphs(
            build_graph_from_bundle(b1),
            build_graph_from_bundle(b2),
        )
        query_nodes = [n for n in merged["nodes"] if n.get("kind") == "query"]
        self.assertEqual(len(query_nodes), 2)

    def test_session_graph_export_import_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            db = DatabaseManager(db_path)
            db.init_db()
            session_id = db.create_session("Graph test")

            bundle1 = _sample_bundle()
            bundle2 = build_general_web_bundle(
                query_raw="Second evidence query",
                query_resolved="Second evidence query",
                kept_rows=[
                    {"title": "Follow-up source", "snippet": "More context.", "url": "https://x.test"}
                ],
                rejected_count=0,
                latency_ms=5.0,
            )
            record_bundle_in_session_graph(
                db, session_id=session_id, bundle=bundle1, message_id="m1"
            )
            record_bundle_in_session_graph(
                db, session_id=session_id, bundle=bundle2, message_id="m2"
            )

            exported = export_session_graph(db, session_id)
            self.assertGreaterEqual(len(exported.get("nodes") or []), 4)

            import_session_graph(db, session_id, {"version": 1, "nodes": [], "edges": []})
            cleared = export_session_graph(db, session_id)
            self.assertEqual(cleared.get("nodes"), [])

            import_session_graph(db, session_id, exported)
            round_trip = export_session_graph(db, session_id)
            self.assertEqual(graph_to_json(exported), graph_to_json(round_trip))

    def test_graph_json_round_trip(self) -> None:
        graph = build_graph_from_bundle(_sample_bundle())
        raw = graph_to_json(graph)
        restored = graph_from_json(raw)
        self.assertEqual(graph_to_json(restored), raw)

    def test_conflict_nodes_when_present(self) -> None:
        bundle = _sample_bundle()
        bundle = bundle_from_dict(bundle_to_dict(bundle))
        bundle = bundle.__class__(
            **{
                **bundle.__dict__,
                "conflicts": (
                    EvidenceConflict(
                        topic="ACE inhibitor efficacy",
                        positions=(("supports", "Positive trial"), ("contradicts", "Negative trial")),
                        severity="material",
                    ),
                ),
            }
        )
        graph = build_graph_from_bundle(bundle)
        entity_types = {
            n.get("entity_type")
            for n in graph["nodes"]
            if n.get("kind") == "entity"
        }
        self.assertIn("conflict", entity_types)


if __name__ == "__main__":
    unittest.main()
