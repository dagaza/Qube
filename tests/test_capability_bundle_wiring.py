"""T10 — cap: provenance survives NormalizedHit -> EvidenceBundle -> UI (P8/KI1).

Proves the end-to-end wiring on the canonical EvidenceBundle path: a capability
hit's ``cap:`` URN reaches ``EvidenceObject.raw_metadata`` (via the generic
bundle builder) and then the UI source row (via ``evidence_to_ui_source`` /
``bundle_to_ui_sources``), which is what INSPECT / Sources render.
"""

from __future__ import annotations

import unittest

from core.integrations.capabilities.model import NormalizedHit
from core.integrations.capabilities.urn import CapabilityURN
from core.knowledge.bundle_builder import build_generic_bundle
from core.knowledge.ui_adapter import bundle_to_ui_sources, evidence_to_ui_source


def _cap_row(urn: CapabilityURN, adapter_id: str = "configured_mcp") -> dict:
    """Mimic the McpConnector row: NormalizedHit dict with a short _adapter id."""
    hit = NormalizedHit(
        title="Reactor safety",
        snippet="A summary about reactors.",
        source_cap=urn,
        url="https://example.test/1",
    )
    row = hit.to_evidence_dict()
    row["_adapter"] = adapter_id  # connector overlays the short id (KI2)
    return row


class TestBundleProvenanceWiring(unittest.TestCase):
    def setUp(self):
        self.urn = CapabilityURN.build("mcp", "github", "search-issues", "2")

    def test_to_evidence_dict_carries_capability(self):
        row = _cap_row(self.urn)
        self.assertEqual(row["_capability"], str(self.urn))
        self.assertEqual(row["retrieval_method"], "mcp")
        self.assertEqual(row["_adapter"], "configured_mcp")

    def test_generic_bundle_preserves_capability_in_raw_metadata(self):
        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[_cap_row(self.urn)],
            rejected_count=0,
            latency_ms=1.0,
            knowledge_service="configured",
        )
        self.assertEqual(len(bundle.sources), 1)
        obj = bundle.sources[0]
        self.assertEqual(obj.raw_metadata.get("capability"), str(self.urn))
        self.assertEqual(obj.adapter, "configured_mcp")

    def test_capability_reaches_ui_source_row(self):
        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[_cap_row(self.urn)],
            rejected_count=0,
            latency_ms=1.0,
            knowledge_service="configured",
        )
        ui_rows = bundle_to_ui_sources(bundle)
        self.assertEqual(len(ui_rows), 1)
        self.assertEqual(ui_rows[0]["source_capability"], str(self.urn))
        self.assertEqual(ui_rows[0]["retrieval_method"], "mcp")

    def test_non_capability_row_has_no_capability_key(self):
        bundle = build_generic_bundle(
            query_raw="q",
            query_resolved="q",
            kept_rows=[{"title": "Plain", "snippet": "no cap", "_adapter": "generic"}],
            rejected_count=0,
            latency_ms=1.0,
            knowledge_service="configured",
        )
        ui_row = evidence_to_ui_source(bundle.sources[0], ui_id=1)
        self.assertNotIn("source_capability", ui_row)


if __name__ == "__main__":
    unittest.main()
