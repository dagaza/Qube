"""Tests for Knowledge → Integrations capability availability."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.persistence import ConsentStore, save_descriptor_cache
from core.integrations.capability_availability import (
    CapabilityBlockReason,
    resolve_capability_availability,
)
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.descriptor_cache import (
    merge_descriptor_cache_for_namespace,
    remove_descriptor_cache_namespace,
)
from core.knowledge.configured_sources import ConfiguredSource, save_configured_source


def _descriptor(raw_name: str, *, namespace: str = "filesystem"):
    group = CapabilityMapper().map_tools(
        "mcp",
        namespace,
        [RawTool(raw_name, raw_name, {"type": "object"})],
    )
    return group.capabilities[0]


class TestCapabilityAvailability(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        self._orig_cs = cs.user_data_root
        cs.user_data_root = lambda: self._root  # type: ignore[assignment]
        self._pro_mcp_patch = patch(
            "core.mcp_filesystem_pro_features.user_has_pro_mcp_filesystem",
            return_value=True,
        )
        self._pro_mcp_patch.start()

    def tearDown(self):
        self._pro_mcp_patch.stop()
        P.user_data_root = self._orig  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        cs.user_data_root = self._orig_cs  # type: ignore[assignment]
        self._tmp.cleanup()

    def test_missing_source_message(self):
        urn = CapabilityURN.build("mcp", "filesystem", "search-files")
        availability = resolve_capability_availability(urn)
        self.assertFalse(availability.available)
        self.assertEqual(availability.reason, CapabilityBlockReason.SOURCE_MISSING)
        self.assertIn("Custom sources", availability.user_message)

    def test_not_granted_when_source_and_descriptor_exist(self):
        save_configured_source(
            ConfiguredSource(
                id="fs-mcp",
                label="Filesystem",
                connector_type="mcp",
                config={
                    "command": ["tool.cmd", "/data"],
                    "namespace": "filesystem",
                    "adapter_id": "fs-mcp",
                },
            )
        )
        desc = _descriptor("search_files")
        save_descriptor_cache("mcp", [desc])
        urn = CapabilityURN.build("mcp", "filesystem", "search-files")
        availability = resolve_capability_availability(urn)
        self.assertFalse(availability.available)
        self.assertEqual(availability.reason, CapabilityBlockReason.NOT_GRANTED)

    def test_ok_when_granted(self):
        save_configured_source(
            ConfiguredSource(
                id="fs-mcp",
                label="Filesystem",
                connector_type="mcp",
                config={
                    "command": ["tool.cmd", "/data"],
                    "namespace": "filesystem",
                    "adapter_id": "fs-mcp",
                },
            )
        )
        desc = _descriptor("search_files")
        save_descriptor_cache("mcp", [desc])
        ConsentStore("mcp").grant(desc)
        urn = CapabilityURN.build("mcp", "filesystem", "search-files")
        availability = resolve_capability_availability(urn)
        self.assertTrue(availability.available)

    def test_invalid_json_source(self):
        bad_path = self._root / "knowledge" / "sources" / "fs-mcp.json"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_text(
            json.dumps(
                {
                    "id": "fs-mcp",
                    "label": "Filesystem",
                    "connector_type": "mcp",
                    "config": {"namespace": "filesystem"},
                }
            ),
            encoding="utf-8",
        )
        urn = CapabilityURN.build("mcp", "filesystem", "search-files")
        availability = resolve_capability_availability(urn)
        self.assertEqual(availability.reason, CapabilityBlockReason.SOURCE_INVALID)

    def test_license_required_for_filesystem_without_pro(self):
        save_configured_source(
            ConfiguredSource(
                id="fs-mcp",
                label="Filesystem",
                connector_type="mcp",
                config={
                    "command": ["tool.cmd", "/data"],
                    "namespace": "filesystem",
                    "adapter_id": "fs-mcp",
                },
            )
        )
        desc = _descriptor("search_files")
        save_descriptor_cache("mcp", [desc])
        urn = CapabilityURN.build("mcp", "filesystem", "search-files")
        self._pro_mcp_patch.stop()
        with patch(
            "core.mcp_filesystem_pro_features.user_has_pro_mcp_filesystem",
            return_value=False,
        ):
            availability = resolve_capability_availability(urn)
        self.assertFalse(availability.available)
        self.assertEqual(availability.reason, CapabilityBlockReason.LICENSE_REQUIRED)

    def test_merge_and_remove_namespace(self):
        a = _descriptor("search_files", namespace="filesystem")
        b = _descriptor("search_docs", namespace="github")
        merge_descriptor_cache_for_namespace("mcp", "filesystem", [a])
        merge_descriptor_cache_for_namespace("mcp", "github", [b])
        from core.integrations.consent_controller import load_cached_descriptors

        names = {d.urn.namespace for d in load_cached_descriptors("mcp")}
        self.assertEqual(names, {"filesystem", "github"})
        remove_descriptor_cache_namespace("mcp", "filesystem")
        names = {d.urn.namespace for d in load_cached_descriptors("mcp")}
        self.assertEqual(names, {"github"})


if __name__ == "__main__":
    unittest.main()
