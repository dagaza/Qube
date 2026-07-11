"""Tests for knowledge platform evolution (Phases 0–4)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.knowledge.explain_preset import build_explain_preset, format_explain_preset_text
from core.knowledge.observability import RETRIEVAL_TRACE_SCHEMA_VERSION, build_retrieval_trace
from core.knowledge.pipeline_graph import build_pipeline_graph_from_trace
from core.knowledge.presets import KnowledgePreset, save_preset
from core.knowledge.registry import (
    SERVICE_PRESET_KNOWLEDGE,
    resolve_turn_knowledge_service,
    resolve_turn_preset_id,
)
from core.knowledge.retrieval_profiles import (
    get_profile_spec,
    normalize_profile_id,
    order_adapter_ids,
)
from core.knowledge.retrieval_records import RetrievalContextFingerprint
from core.knowledge.retrieval_replay import compare_traces
from core.composer_attachments import is_web_composer_tool


class TestKnowledgePlatformEvolution(unittest.TestCase):
    def test_user_preset_is_web_composer_tool(self) -> None:
        self.assertTrue(is_web_composer_tool("user:biology"))
        self.assertTrue(is_web_composer_tool("source:pubmed"))

    def test_preset_routing_user_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("core.knowledge.presets.user_data_root", return_value=root):
                save_preset(
                    KnowledgePreset(
                        id="biology",
                        label="Biology",
                        adapters=["pubmed"],
                    )
                )
                self.assertEqual(
                    resolve_turn_knowledge_service(composer_tool="user:biology"),
                    SERVICE_PRESET_KNOWLEDGE,
                )
                self.assertEqual(resolve_turn_preset_id("user:biology"), "biology")

    def test_retrieval_profile_normalization(self) -> None:
        self.assertEqual(normalize_profile_id("FAST"), "fast")
        self.assertEqual(normalize_profile_id("unknown"), "balanced")
        fast = get_profile_spec("fast")
        self.assertEqual(fast.max_parallel_adapters, 2)

    def test_local_first_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("core.knowledge.configured_sources.user_data_root", return_value=root):
                from core.knowledge.configured_sources import ConfiguredSource, save_configured_source

                save_configured_source(
                    ConfiguredSource(
                        id="localdb",
                        label="Local DB",
                        connector_type="sqlite",
                        config={"database_path": "/tmp/x.db", "query": "SELECT 1"},
                    )
                )
                profile = get_profile_spec("local_first")
                ordered = order_adapter_ids(("pubmed", "localdb"), profile=profile)
                self.assertEqual(ordered[0], "localdb")

    def test_context_fingerprint_roundtrip(self) -> None:
        fp = RetrievalContextFingerprint(
            query_raw="q",
            query_resolved="q",
            knowledge_service="scientific_evidence",
            preset_id="bio",
            adapter_filter=("pubmed",),
            retrieval_profile="balanced",
            connector_config_hashes=(),
        )
        restored = RetrievalContextFingerprint.from_dict(fp.to_dict())
        self.assertEqual(restored.preset_id, "bio")
        self.assertEqual(restored.adapter_filter, ("pubmed",))

    def test_trace_schema_v3_fields(self) -> None:
        from core.knowledge.bundle_builder import build_general_web_bundle

        bundle = build_general_web_bundle(
            query_raw="test",
            query_resolved="test",
            kept_rows=[{"title": "T", "snippet": "S"}],
            rejected_count=0,
            latency_ms=10.0,
        )
        trace = build_retrieval_trace(
            bundle,
            preset_id="biology",
            retrieval_profile="fast",
            context_fingerprint={"query_raw": "test"},
            pipeline_stages=[{"stage": "plan", "latency_ms": 1.0}],
        )
        self.assertEqual(trace.schema_version, RETRIEVAL_TRACE_SCHEMA_VERSION)
        self.assertEqual(trace.preset_id, "biology")
        self.assertEqual(trace.retrieval_profile, "fast")
        self.assertEqual(len(trace.pipeline_stages), 1)

    def test_pipeline_graph_from_trace(self) -> None:
        trace = {
            "query_raw": "CRISPR",
            "knowledge_service": "preset_knowledge",
            "retrieval_strategy": "preset:biology",
            "adapter_calls": ["pubmed"],
            "evidence_ids_kept": ["e1"],
            "candidates_rejected_count": 0,
            "coverage": "adequate",
            "confidence": 0.8,
            "pipeline_stages": [{"stage": "plan", "latency_ms": 2}],
        }
        nodes = build_pipeline_graph_from_trace(trace)
        labels = [n["label"] for n in nodes]
        self.assertIn("Question", labels)
        self.assertIn("EvidenceBundle", labels)

    def test_compare_traces(self) -> None:
        cmp = compare_traces(
            {"evidence_ids_kept": ["a", "b"], "coverage": "adequate", "latency_ms": 100},
            {"evidence_ids_kept": ["b", "c"], "coverage": "excellent", "latency_ms": 80},
        )
        self.assertEqual(cmp["evidence_removed"], ["a"])
        self.assertEqual(cmp["evidence_added"], ["c"])
        self.assertEqual(cmp["evidence_unchanged"], ["b"])

    def test_explain_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("core.knowledge.presets.user_data_root", return_value=root):
                save_preset(
                    KnowledgePreset(
                        id="biology",
                        label="Biology",
                        adapters=["pubmed"],
                        description="Life sciences bundle",
                    )
                )
                explain = build_explain_preset("biology")
                text = format_explain_preset_text(explain)
                self.assertIn("Biology", text)


if __name__ == "__main__":
    unittest.main()
