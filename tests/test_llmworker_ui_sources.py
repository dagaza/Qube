"""T13 — LLMWorker main-path UI sources via EvidenceBundle (P8/KI1)."""

from __future__ import annotations

import unittest

from core.integrations.capabilities.model import NormalizedHit
from core.integrations.capabilities.urn import CapabilityURN
from core.knowledge.bundle_builder import build_generic_bundle
from core.knowledge.ui_adapter import append_turn_evidence_bundle_sources, bundle_to_ui_sources


def _apply_sequential_source_ids(sources: list, start: int = 1) -> None:
    """Mirror LLMWorker._apply_sequential_source_ids (memory → RAG → bundle/web)."""
    for i, src in enumerate(sources, start=start):
        if isinstance(src, dict):
            src["id"] = i


def _cap_row(urn: CapabilityURN) -> dict:
    hit = NormalizedHit(
        title="Cap hit",
        snippet="From MCP.",
        source_cap=urn,
        url="https://example.test/cap",
    )
    row = hit.to_evidence_dict()
    row["_adapter"] = "configured_mcp"
    return row


class TestLLMWorkerUiSourcesMerge(unittest.TestCase):
    def setUp(self):
        self.urn = CapabilityURN.build("mcp", "github", "search-issues", "2")

    def test_bundle_append_carries_source_capability(self):
        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[_cap_row(self.urn)],
            rejected_count=0,
            latency_ms=1.0,
            knowledge_service="configured",
        )
        all_ui_sources: list[dict] = []
        append_turn_evidence_bundle_sources(all_ui_sources, bundle)
        self.assertEqual(len(all_ui_sources), 1)
        self.assertEqual(all_ui_sources[0]["source_capability"], str(self.urn))
        self.assertEqual(all_ui_sources[0]["retrieval_method"], "mcp")

    def test_mem_rag_and_bundle_ids_stay_unique_after_renumber(self):
        mem_rag = [
            {"id": 99, "filename": "Memory fact", "content": "x", "type": "memory"},
            {"id": 100, "filename": "Doc chunk", "content": "y", "type": "library"},
        ]
        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[
                _cap_row(self.urn),
                {"title": "Web page", "snippet": "plain web", "_adapter": "web"},
            ],
            rejected_count=0,
            latency_ms=1.0,
            knowledge_service="configured",
        )
        merged = list(mem_rag)
        append_turn_evidence_bundle_sources(merged, bundle)
        _apply_sequential_source_ids(merged)
        ids = [row["id"] for row in merged]
        self.assertEqual(ids, [1, 2, 3, 4])
        cap_row = next(r for r in merged if r.get("source_capability"))
        self.assertEqual(cap_row["source_capability"], str(self.urn))
        plain = next(r for r in merged if r.get("filename") == "Web page")
        self.assertNotIn("source_capability", plain)

    def test_empty_bundle_falls_back_to_manual_web_pattern(self):
        """When the bundle has no sources, callers keep the legacy web-item loop."""
        web_items = [{"title": "Result", "snippet": "text", "url": "https://x.test"}]
        all_ui_sources: list[dict] = []
        append_turn_evidence_bundle_sources(all_ui_sources, None)
        for item in web_items:
            all_ui_sources.append(
                {
                    "filename": item["title"],
                    "content": item["snippet"],
                    "type": "web",
                    "url": item["url"],
                }
            )
        _apply_sequential_source_ids(all_ui_sources)
        self.assertEqual(len(all_ui_sources), 1)
        self.assertNotIn("source_capability", all_ui_sources[0])

    def test_bundle_to_ui_sources_non_cap_has_no_capability_key(self):
        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[{"title": "Plain", "snippet": "no cap", "_adapter": "generic"}],
            rejected_count=0,
            latency_ms=1.0,
            knowledge_service="configured",
        )
        rows = bundle_to_ui_sources(bundle)
        self.assertNotIn("source_capability", rows[0])


if __name__ == "__main__":
    unittest.main()
