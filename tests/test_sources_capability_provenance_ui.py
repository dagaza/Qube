"""T18 — Sources UI shows cap: provenance via source_capability labels."""

from __future__ import annotations

import unittest

from core.integrations.capabilities.urn import CapabilityURN
from core.knowledge.ui_adapter import (
    source_provenance_metadata_parts,
    source_type_label_for_row,
)


class TestSourcesCapabilityProvenanceUi(unittest.TestCase):
    def test_display_label_from_urn(self):
        urn = CapabilityURN.build("mcp", "github", "search-issues", "2")
        self.assertEqual(urn.display_label, "Github — Search Issues")

    def test_type_label_prefers_source_capability(self):
        urn = CapabilityURN.build("mcp", "github", "search-issues")
        row = {
            "source_capability": str(urn),
            "source_adapter": "configured_mcp",
            "type": "web",
        }
        self.assertEqual(source_type_label_for_row(row), "Github — Search Issues")

    def test_type_label_falls_back_to_adapter_when_no_capability(self):
        row = {"source_adapter": "pubmed", "type": "web"}
        self.assertEqual(source_type_label_for_row(row), "Pubmed")

    def test_type_label_ki2_cap_adapter_fallback(self):
        urn = CapabilityURN.build("mcp", "github", "search-issues")
        row = {"source_adapter": str(urn), "type": "web"}
        self.assertEqual(source_type_label_for_row(row), "Github — Search Issues")

    def test_type_label_non_cap_type_fallbacks_unchanged(self):
        self.assertEqual(source_type_label_for_row({"type": "memory"}), "Memory")
        self.assertEqual(source_type_label_for_row({"type": "web"}), "Web")
        self.assertEqual(source_type_label_for_row({"type": "library"}), "Library")

    def test_provenance_metadata_includes_method_and_urn_body(self):
        urn = CapabilityURN.build("mcp", "github", "search-issues")
        row = {
            "source_capability": str(urn),
            "retrieval_method": "mcp",
        }
        self.assertEqual(
            source_provenance_metadata_parts(row),
            ["mcp", "mcp:github/search-issues"],
        )

    def test_provenance_metadata_empty_for_plain_web_row(self):
        row = {"source_adapter": "web", "type": "web"}
        self.assertEqual(source_provenance_metadata_parts(row), [])


if __name__ == "__main__":
    unittest.main()
