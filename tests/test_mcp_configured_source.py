"""Configured MCP source binding for composer invoke."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.persistence import ConsentStore, save_descriptor_cache
from core.integrations.capability_invoke import invoke_gated_capability
from core.integrations.mcp_configured_source import (
    augment_spawn_env_for_command,
    build_search_invoke_arg_sets,
    build_tool_call_arguments,
    extract_search_names,
    merge_mcp_factory_kwargs,
    resolve_configured_mcp_binding,
)
from core.integrations.registry.provider_registry import reset_registry_for_tests
from core.knowledge.configured_sources import ConfiguredSource, save_configured_source

_MOCK_SERVER = Path(__file__).resolve().parent / "fixtures" / "mock_mcp_server.py"


def _descriptor(raw_name: str, *, schema: dict | None = None):
    group = CapabilityMapper().map_tools(
        "mcp",
        "filesystem",
        [RawTool(raw_name, raw_name, schema or {"type": "object"})],
    )
    return group.capabilities[0]


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        self._orig_cs_root = cs.user_data_root
        cs.user_data_root = lambda: self._root  # type: ignore[assignment]
        reset_registry_for_tests()

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        import core.knowledge.configured_sources as cs

        cs.user_data_root = self._orig_cs_root  # type: ignore[assignment]
        cs._configured_search_fn.cache_clear()
        self._tmp.cleanup()
        reset_registry_for_tests()


class TestMcpConfiguredBinding(_TmpRootTestCase):
    def _fixture_root(self) -> str:
        return str(self._root / "workspace")

    def _save_fs_source(self) -> None:
        root = self._fixture_root()
        save_configured_source(
            ConfiguredSource(
                id="fs-mcp",
                label="Filesystem MCP",
                connector_type="mcp",
                config={
                    "command": [sys.executable, str(_MOCK_SERVER), root],
                    "namespace": "filesystem",
                    "adapter_id": "fs-mcp",
                    "tool_name": "search_docs",
                },
            )
        )

    def test_resolve_binding_by_namespace(self):
        self._save_fs_source()
        binding = resolve_configured_mcp_binding("filesystem")
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding.adapter_id, "fs-mcp")
        self.assertEqual(binding.root_path, self._fixture_root())
        self.assertEqual(binding.command[0], sys.executable)

    def test_merge_factory_kwargs_adds_command(self):
        self._save_fs_source()
        kwargs, binding = merge_mcp_factory_kwargs("mcp", "filesystem", {})
        self.assertIn("command", kwargs)
        self.assertIsNotNone(binding)
        self.assertIn("nodejs", kwargs["env"].get("PATH", "").lower())

    def test_augment_spawn_env_adds_nodejs_on_windows_cmd(self):
        env = augment_spawn_env_for_command(
            [r"C:\Program Files\nodejs\mcp-server-filesystem.cmd"],
            {"PATH": r"C:\Windows\System32"},
        )
        self.assertIn("nodejs", env["PATH"].lower())

    def test_build_list_directory_args_use_root(self):
        desc = _descriptor(
            "list_directory",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )
        args = build_tool_call_arguments(
            desc,
            "Search the workspace for files named Alpha or Beta",
            binding=resolve_configured_mcp_binding("filesystem")
            or type(
                "B",
                (),
                {"root_path": self._fixture_root()},
            )(),
        )
        self.assertEqual(args["path"], self._fixture_root())

    def test_build_search_args_extract_pattern(self):
        desc = _descriptor(
            "search_files",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
            },
        )
        args = build_tool_call_arguments(
            desc,
            "files whose names contain Alpha or Beta",
            binding=type("B", (), {"root_path": "/docs"})(),
        )
        self.assertEqual(args["path"], "/docs")
        self.assertEqual(args["pattern"], "**/*Alpha*")

    def test_build_search_invoke_arg_sets_for_or_names(self):
        desc = _descriptor(
            "search_files",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
            },
        )
        arg_sets = build_search_invoke_arg_sets(
            desc,
            "files whose names contain Alpha or Beta",
            binding=type("B", (), {"root_path": "/docs"})(),
        )
        self.assertEqual(
            arg_sets,
            [
                {"path": "/docs", "pattern": "**/*Alpha*"},
                {"path": "/docs", "pattern": "**/*Beta*"},
            ],
        )

    def test_extract_search_names_without_match_returns_empty(self):
        self.assertEqual(extract_search_names("list all pdf files"), [])

    def test_glob_pattern_titlecases_lowercase_names(self):
        from core.integrations.mcp_configured_source import _glob_pattern_for_name

        self.assertEqual(_glob_pattern_for_name("alpha"), "**/*Alpha*")

    def test_invoke_uses_configured_command_with_grant(self):
        self._save_fs_source()
        read = _descriptor(
            "search_docs",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        save_descriptor_cache("mcp", [read])
        ConsentStore("mcp").grant(read)

        result = invoke_gated_capability(
            "cap:mcp:filesystem/search-docs",
            "reactor safety",
            live_descriptors=[read],
        )
        self.assertTrue(result.allowed, result.reason)
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["_adapter"], "fs-mcp")

    def test_invoke_denies_without_grant_even_with_command(self):
        self._save_fs_source()
        read = _descriptor(
            "search_docs",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        save_descriptor_cache("mcp", [read])

        result = invoke_gated_capability(
            "cap:mcp:filesystem/search-docs",
            "reactor safety",
            live_descriptors=[read],
        )
        self.assertFalse(result.allowed)
        self.assertIn("grant", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
