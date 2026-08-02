"""Tests for Pro MCP Filesystem integration gating."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.capabilities import EditionTier, invalidate_capabilities_cache, resolve_capabilities
from core.integrations.capability_availability import (
    CapabilityBlockReason,
    resolve_capability_availability,
)
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.mcp_configured_source import resolve_configured_mcp_binding
from core.mcp_filesystem_pro_features import (
    FILESYSTEM_MCP_NAMESPACE,
    is_mcp_filesystem_config,
    is_mcp_filesystem_source,
    require_pro_mcp_filesystem_for_config,
)


@dataclass
class _FakeSource:
    id: str
    label: str
    connector_type: str
    config: dict = field(default_factory=dict)


class McpFilesystemProFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        invalidate_capabilities_cache()

    def tearDown(self) -> None:
        invalidate_capabilities_cache()

    def test_home_tier_denies_mcp_filesystem(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.HOME, source="test")
        self.assertFalse(caps.has("pro.mcp_filesystem"))

    def test_pro_tier_grants_mcp_filesystem(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.PRO, source="test")
        self.assertTrue(caps.has("pro.mcp_filesystem"))

    def test_detects_filesystem_namespace(self) -> None:
        self.assertTrue(
            is_mcp_filesystem_config({"namespace": FILESYSTEM_MCP_NAMESPACE})
        )

    def test_detects_filesystem_command(self) -> None:
        self.assertTrue(
            is_mcp_filesystem_config(
                {
                    "namespace": "projects",
                    "command": ["mcp-server-filesystem", "/tmp"],
                }
            )
        )

    def test_non_filesystem_mcp_not_detected(self) -> None:
        self.assertFalse(
            is_mcp_filesystem_config(
                {
                    "namespace": "github",
                    "command": ["mcp-server-github"],
                }
            )
        )

    def test_require_pro_blocks_filesystem_without_license(self) -> None:
        source = _FakeSource(
            id="local-fs",
            label="Local FS",
            connector_type="mcp",
            config={
                "namespace": FILESYSTEM_MCP_NAMESPACE,
                "command": ["mcp-server-filesystem", "/tmp"],
            },
        )
        with patch(
            "core.mcp_filesystem_pro_features.user_has_pro_mcp_filesystem",
            return_value=False,
        ):
            self.assertTrue(is_mcp_filesystem_source(source))
            with self.assertRaises(ValueError):
                from core.mcp_filesystem_pro_features import (
                    require_pro_mcp_filesystem_for_source,
                )

                require_pro_mcp_filesystem_for_source(source)

    def test_binding_hidden_without_license(self) -> None:
        fake_source = _FakeSource(
            id="local-fs",
            label="Local FS",
            connector_type="mcp",
            config={
                "namespace": FILESYSTEM_MCP_NAMESPACE,
                "command": ["mcp-server-filesystem", "/tmp"],
            },
        )
        with patch(
            "core.knowledge.configured_sources.list_configured_sources",
            return_value=[fake_source],
        ), patch(
            "core.mcp_filesystem_pro_features.user_has_pro_mcp_filesystem",
            return_value=False,
        ):
            self.assertIsNone(resolve_configured_mcp_binding(FILESYSTEM_MCP_NAMESPACE))

    def test_capability_availability_reports_license_required(self) -> None:
        urn = CapabilityURN.build("mcp", FILESYSTEM_MCP_NAMESPACE, "search-files")
        fake_source = _FakeSource(
            id="local-fs",
            label="Local FS",
            connector_type="mcp",
            config={
                "namespace": FILESYSTEM_MCP_NAMESPACE,
                "command": ["mcp-server-filesystem", "/tmp"],
            },
        )
        with patch(
            "core.integrations.capability_availability.inspect_configured_mcp_namespace",
            return_value=("ok", "local-fs", ""),
        ), patch(
            "core.knowledge.configured_sources.load_configured_source",
            return_value=fake_source,
        ), patch(
            "core.mcp_filesystem_pro_features.user_has_pro_mcp_filesystem",
            return_value=False,
        ):
            availability = resolve_capability_availability(urn)
            self.assertFalse(availability.available)
            self.assertEqual(availability.reason, CapabilityBlockReason.LICENSE_REQUIRED)


if __name__ == "__main__":
    unittest.main()
