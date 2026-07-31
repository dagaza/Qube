"""Tests for MCP integration cache reconciliation with Knowledge sources."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.persistence import ConsentStore, save_descriptor_cache
from core.integrations.consent_controller import IntegrationsConsentController
from core.integrations.descriptor_cache import reconcile_mcp_integration_state
from core.knowledge.configured_sources import ConfiguredSource, save_configured_source


def _descriptor(raw_name: str, *, namespace: str = "filesystem"):
    group = CapabilityMapper().map_tools(
        "mcp",
        namespace,
        [RawTool(raw_name, raw_name, {"type": "object"})],
    )
    return group.capabilities[0]


class TestMcpIntegrationReconcile(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        self._orig_cs = cs.user_data_root
        cs.user_data_root = lambda: self._root  # type: ignore[assignment]

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        cs.user_data_root = self._orig_cs  # type: ignore[assignment]
        self._tmp.cleanup()

    def test_reconcile_removes_orphan_descriptors_and_consent(self):
        desc = _descriptor("search_files")
        save_descriptor_cache("mcp", [desc])
        IntegrationsConsentController("mcp").grant_capability(desc)

        summary = reconcile_mcp_integration_state()
        self.assertGreater(summary["descriptors_removed"], 0)
        self.assertGreater(summary["grants_removed"], 0)
        self.assertEqual(IntegrationsConsentController("mcp").list_capability_rows(), [])

    def test_reconcile_keeps_configured_namespace(self):
        save_configured_source(
            ConfiguredSource(
                id="fs-mcp",
                label="Filesystem",
                connector_type="mcp",
                config={
                    "command": ["tool.cmd", "/data"],
                    "namespace": "filesystem",
                },
            )
        )
        desc = _descriptor("search_files")
        save_descriptor_cache("mcp", [desc])
        IntegrationsConsentController("mcp").grant_capability(desc)

        summary = reconcile_mcp_integration_state()
        self.assertEqual(summary["descriptors_removed"], 0)
        self.assertEqual(summary["grants_removed"], 0)
        self.assertEqual(len(IntegrationsConsentController("mcp").list_capability_rows()), 1)

    def test_reconcile_after_source_file_removed(self):
        save_configured_source(
            ConfiguredSource(
                id="fs-mcp",
                label="Filesystem",
                connector_type="mcp",
                config={"command": ["tool.cmd"], "namespace": "filesystem"},
            )
        )
        desc = _descriptor("search_files")
        save_descriptor_cache("mcp", [desc])
        IntegrationsConsentController("mcp").grant_capability(desc)

        source_path = self._root / "knowledge" / "sources" / "fs-mcp.json"
        source_path.unlink()

        summary = reconcile_mcp_integration_state()
        self.assertGreater(summary["descriptors_removed"], 0)
        self.assertEqual(IntegrationsConsentController("mcp").list_capability_rows(), [])


if __name__ == "__main__":
    unittest.main()
