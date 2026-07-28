"""Custom sources form logic — disk load, MCP preserve, command parsing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.knowledge.configured_sources import (
    ConfiguredSource,
    list_configured_sources,
    save_configured_source,
)
from core.knowledge.custom_source_editor import (
    build_configured_source_from_fields,
    configured_source_to_field_values,
    parse_mcp_command,
)


class TestCustomSourcesForm(unittest.TestCase):
    def test_parse_mcp_command_json_array(self):
        parsed = parse_mcp_command(
            '["mcp-server-filesystem.cmd", "C:\\\\Data\\\\Projects"]'
        )
        self.assertEqual(parsed, ["mcp-server-filesystem.cmd", "C:\\Data\\Projects"])

    def test_field_values_from_mcp_source(self):
        source = ConfiguredSource(
            id="fs-mcp",
            label="Filesystem MCP",
            connector_type="mcp",
            config={
                "command": ["mcp-server-filesystem.cmd", "/data"],
                "namespace": "filesystem",
                "tool_name": "search_files",
                "adapter_id": "fs-mcp",
            },
        )
        values = configured_source_to_field_values(source)
        self.assertEqual(values["connector_type"], "mcp")
        self.assertEqual(values["id"], "fs-mcp")
        self.assertIn("mcp-server-filesystem.cmd", values["mcp_command"])
        self.assertEqual(values["mcp_namespace"], "filesystem")
        self.assertEqual(values["mcp_tool_name"], "search_files")

    def test_save_preserves_mcp_config_when_editing(self):
        loaded = ConfiguredSource(
            id="fs-mcp",
            label="Filesystem MCP",
            connector_type="mcp",
            trust_policy="enterprise",
            config={
                "command": ["server.cmd", "/docs"],
                "namespace": "filesystem",
                "tool_name": "search_files",
                "adapter_id": "fs-mcp",
                "env": {"FOO": "bar"},
            },
        )
        built = build_configured_source_from_fields(
            source_id="fs-mcp",
            label="Filesystem MCP",
            connector_type="mcp",
            mcp_command=json.dumps(["server.cmd", "/docs"]),
            mcp_namespace="filesystem",
            mcp_tool_name="search_files",
            loaded=loaded,
        )
        self.assertEqual(built.connector_type, "mcp")
        self.assertEqual(built.config["command"], ["server.cmd", "/docs"])
        self.assertEqual(built.config["env"], {"FOO": "bar"})

    def test_new_source_id_does_not_inherit_loaded_mcp_env(self):
        loaded = ConfiguredSource(
            id="fs-mcp",
            label="Filesystem MCP",
            connector_type="mcp",
            config={
                "command": ["server.cmd", "/docs"],
                "namespace": "filesystem",
                "tool_name": "search_files",
                "adapter_id": "fs-mcp",
                "env": {"FOO": "bar"},
            },
        )
        built = build_configured_source_from_fields(
            source_id="other-mcp",
            label="Other MCP",
            connector_type="mcp",
            mcp_command=json.dumps(["other.cmd", "/tmp"]),
            mcp_namespace="other",
            mcp_tool_name="search_files",
            loaded=loaded,
        )
        self.assertEqual(built.id, "other-mcp")
        self.assertNotIn("env", built.config)

    def test_list_picks_up_externally_created_source_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            import core.knowledge.configured_sources as configured

            configured.user_data_root = lambda: root  # type: ignore[assignment]
            save_configured_source(
                ConfiguredSource(
                    id="external",
                    label="External Source",
                    connector_type="mcp",
                    config={"command": ["tool.cmd"], "adapter_id": "external"},
                )
            )
            sources = list_configured_sources()
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].connector_type, "mcp")


if __name__ == "__main__":
    unittest.main()
