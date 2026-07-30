"""Tests for capability drift, grant review, consent export, and MCP registry."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capabilities.persistence import ConsentStore, save_descriptor_cache
from core.integrations.capability_drift import (
    diff_namespace_capabilities,
    format_drift_summary,
    has_material_drift,
)
from core.integrations.consent_controller import ConsentUIState, IntegrationsConsentController
from core.integrations.consent_export import export_integration_consents, import_integration_consents
from core.integrations.grant_review import (
    GrantReviewChange,
    apply_grant_review_rows,
    build_grant_review_rows,
    suggest_capability_presets,
)
from core.integrations.mcp_server_registry import list_mcp_server_summaries
from core.knowledge.configured_sources import ConfiguredSource, save_configured_source
from core.knowledge.knowledge_pack import export_knowledge_pack, import_knowledge_pack


def _descriptors(tools):
    group = CapabilityMapper().map_tools("mcp", "github", tools)
    return list(group.capabilities)


_TOOLS_V1 = [
    RawTool("search_issues", "Find open GitHub issues", {"type": "object"}),
    RawTool("create_issue", "Open a new issue", {"type": "object"}),
]

_TOOLS_V2 = [
    RawTool("search_issues", "Find open GitHub issues", {"type": "object", "properties": {"q": {}}}),
    RawTool("create_issue", "Open a new issue", {"type": "object"}),
    RawTool("delete_branch", "Delete a branch", {"type": "object"}),
]


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs
        import core.knowledge.presets as presets

        self._orig_cs = cs.user_data_root
        self._orig_presets = presets.user_data_root
        cs.user_data_root = lambda: self._root  # type: ignore[assignment]
        presets.user_data_root = lambda: self._root  # type: ignore[assignment]

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs
        import core.knowledge.presets as presets

        cs.user_data_root = self._orig_cs  # type: ignore[assignment]
        presets.user_data_root = self._orig_presets  # type: ignore[assignment]
        self._tmp.cleanup()


class TestCapabilityDrift(_TmpRootTestCase):
    def test_diff_detects_additions_and_schema_change(self):
        before = _descriptors(_TOOLS_V1)
        after = _descriptors(_TOOLS_V2)
        diff = diff_namespace_capabilities(before, after, namespace="github")
        self.assertTrue(has_material_drift(diff))
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(diff.added[0].action, "delete-branch")
        self.assertEqual(len(diff.changed), 1)

    def test_format_drift_summary_pluralization(self):
        before = _descriptors(_TOOLS_V1)
        after = _descriptors(_TOOLS_V2)
        diff = diff_namespace_capabilities(before, after, namespace="github")
        summary = format_drift_summary(diff)
        self.assertIn("capabilities", summary)
        self.assertNotIn("capability(ies)", summary)

        shrink_diff = diff_namespace_capabilities(before, before[:1], namespace="github")
        self.assertEqual(format_drift_summary(shrink_diff), "1 removed capability")


class TestGrantReview(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        self.descs = _descriptors(_TOOLS_V1)
        save_descriptor_cache("mcp", self.descs)

    def test_first_connect_defaults_read_on_write_off(self):
        rows = build_grant_review_rows(
            "mcp",
            self.descs,
            namespace="github",
            first_connect=True,
            drift=None,
        )
        by_action = {row.descriptor.action: row for row in rows}
        self.assertTrue(by_action["search-issues"].checked)
        self.assertFalse(by_action["create-issue"].checked)

    def test_apply_grant_review_persists(self):
        rows = build_grant_review_rows(
            "mcp",
            self.descs,
            namespace="github",
            first_connect=True,
            drift=None,
        )
        apply_grant_review_rows("mcp", rows)
        controller = IntegrationsConsentController("mcp")
        granted = {
            row.descriptor.action: row.ui_state
            for row in controller.list_capability_rows()
        }
        self.assertEqual(granted["search-issues"], ConsentUIState.ALLOWED)
        self.assertEqual(granted["create-issue"], ConsentUIState.DENIED)

    def test_suggest_presets(self):
        presets = suggest_capability_presets(
            namespace="github",
            server_label="GitHub MCP",
            descriptors=self.descs,
        )
        self.assertTrue(presets)
        self.assertTrue(all(preset.capability_urns for preset in presets))

    def test_capability_in_preset(self):
        from core.integrations.grant_review import capability_in_preset

        presets = suggest_capability_presets(
            namespace="github",
            server_label="GitHub MCP",
            descriptors=self.descs,
        )
        minimal = next(p for p in presets if p.preset_id.endswith("-minimal"))
        read_desc = next(d for d in self.descs if d.action == "search-issues")
        write_desc = next(d for d in self.descs if d.action == "create-issue")
        self.assertTrue(capability_in_preset(read_desc, minimal))
        self.assertFalse(capability_in_preset(write_desc, minimal))


class TestConsentExport(_TmpRootTestCase):
    def test_export_import_roundtrip(self):
        descs = _descriptors(_TOOLS_V1)
        read = next(d for d in descs if d.action == "search-issues")
        IntegrationsConsentController("mcp").grant_capability(read)
        exported = export_integration_consents()
        self.assertIn("mcp", exported.get("providers", {}))

        other_root = self._root / "other"
        other_root.mkdir()
        P.user_data_root = lambda: other_root  # type: ignore[assignment]
        summary = import_integration_consents(exported, merge=False)
        self.assertEqual(summary["providers_imported"], 1)
        self.assertGreaterEqual(summary["grants_imported"], 1)

    def test_knowledge_pack_includes_consents(self):
        descs = _descriptors(_TOOLS_V1)
        read = next(d for d in descs if d.action == "search-issues")
        IntegrationsConsentController("mcp").grant_capability(read)
        pack = export_knowledge_pack(include_sources=False, include_presets=False)
        self.assertIn("integration_consents", pack)
        self.assertIn("mcp", pack["integration_consents"].get("providers", {}))


class TestMcpServerRegistry(_TmpRootTestCase):
    def test_list_summaries_for_configured_mcp_source(self):
        save_configured_source(
            ConfiguredSource(
                id="github-mcp",
                label="GitHub MCP",
                connector_type="mcp",
                config={"command": ["github-mcp.cmd"], "namespace": "github"},
            )
        )
        save_descriptor_cache("mcp", _descriptors(_TOOLS_V1))
        summaries = list_mcp_server_summaries()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].namespace, "github")
        self.assertEqual(summaries[0].capability_count, 2)
        self.assertEqual(summaries[0].health_label, "Permissions pending")


if __name__ == "__main__":
    unittest.main()
