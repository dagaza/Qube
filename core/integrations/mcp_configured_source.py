"""Bridge composer MCP invokes to configured knowledge sources (stdio command).

Terminal ``McpConnector.execute`` reads ``command`` from ``fs-mcp.json`` (etc.).
Composer ``@[cap:mcp:…]`` must resolve the same binding by namespace before
constructing :class:`~core.integrations.providers.mcp.McpCapabilityProvider`.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from core.integrations.capabilities.model import CapabilityDescriptor

__all__ = [
    "McpConfiguredBinding",
    "augment_spawn_env_for_command",
    "build_search_invoke_arg_sets",
    "build_tool_call_arguments",
    "configured_mcp_namespaces",
    "extract_search_names",
    "merge_mcp_factory_kwargs",
    "resolve_configured_mcp_binding",
]

_NAME_FILTER_RE = re.compile(
    r"(?:contain(?:ing)?|named|matching|with)\s+['\"]?([A-Za-z0-9._-]+)['\"]?"
    r"(?:\s+or\s+['\"]?([A-Za-z0-9._-]+)['\"]?)?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class McpConfiguredBinding:
    """Resolved MCP server launch config from a configured source."""

    command: list[str]
    namespace: str
    adapter_id: str
    root_path: str | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None


def augment_spawn_env_for_command(
    command: list[str],
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Ensure Windows GUI launches can find Node when spawning ``.cmd`` MCP servers."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    if sys.platform != "win32" or not command:
        return merged
    launcher = str(command[0] or "").lower()
    if not launcher.endswith((".cmd", ".bat")):
        return merged
    for node_dir in (
        r"C:\Program Files\nodejs",
        r"C:\Program Files (x86)\nodejs",
    ):
        if not os.path.isdir(node_dir):
            continue
        path = merged.get("PATH", "")
        if node_dir.lower() in path.lower():
            return merged
        merged["PATH"] = f"{node_dir};{path}" if path else node_dir
        return merged
    return merged


def configured_mcp_namespaces() -> frozenset[str]:
    """Return MCP namespaces that have a configured Knowledge custom source."""
    from core.knowledge.configured_sources import list_configured_sources
    from core.mcp_filesystem_pro_features import (
        is_mcp_filesystem_source,
        user_has_pro_mcp_filesystem,
    )

    namespaces: set[str] = set()
    for source in list_configured_sources():
        if source.connector_type != "mcp":
            continue
        if is_mcp_filesystem_source(source) and not user_has_pro_mcp_filesystem():
            continue
        cfg = dict(source.config or {})
        ns = str(cfg.get("namespace") or cfg.get("adapter_id") or source.id).strip().lower()
        if ns:
            namespaces.add(ns)
    return frozenset(namespaces)


def resolve_configured_mcp_binding(namespace: str) -> McpConfiguredBinding | None:
    """Find an MCP configured source whose ``namespace`` matches ``cap:mcp:<ns>/…``."""
    from core.knowledge.configured_sources import list_configured_sources

    want = (namespace or "").strip().lower()
    if not want:
        return None

    for source in list_configured_sources():
        if source.connector_type != "mcp":
            continue
        cfg = dict(source.config or {})
        src_ns = str(cfg.get("namespace") or cfg.get("adapter_id") or source.id).strip().lower()
        if src_ns != want:
            continue
        command = cfg.get("command")
        if not isinstance(command, list) or not command:
            continue
        env_raw = cfg.get("env")
        env = dict(env_raw) if isinstance(env_raw, dict) else None
        cwd = str(cfg.get("cwd") or "").strip() or None
        binding = McpConfiguredBinding(
            command=[str(part) for part in command],
            namespace=src_ns,
            adapter_id=str(cfg.get("adapter_id") or source.id),
            root_path=_root_path_from_command(command),
            env=env,
            cwd=cwd,
        )
        from core.mcp_filesystem_pro_features import mcp_filesystem_integration_allowed

        if not mcp_filesystem_integration_allowed(binding=binding):
            continue
        return binding
    return None


def merge_mcp_factory_kwargs(
    provider_id: str,
    namespace: str,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], McpConfiguredBinding | None]:
    """Augment provider factory kwargs with a configured MCP source when missing."""
    pid = (provider_id or "").strip().lower()
    if pid != "mcp":
        return kwargs, None
    if kwargs.get("transport") is not None or kwargs.get("command"):
        binding = None
        if isinstance(kwargs.get("command"), list):
            binding = McpConfiguredBinding(
                command=[str(x) for x in kwargs["command"]],
                namespace=(namespace or "").strip().lower(),
                adapter_id=str(kwargs.get("adapter_id") or namespace),
                root_path=_root_path_from_command(kwargs["command"]),
            )
        return kwargs, binding

    binding = resolve_configured_mcp_binding(namespace)
    if binding is None:
        return kwargs, None

    merged = dict(kwargs)
    merged["command"] = list(binding.command)
    merged["namespace"] = binding.namespace
    spawn_env = augment_spawn_env_for_command(
        list(binding.command),
        dict(binding.env) if binding.env else None,
    )
    merged["env"] = spawn_env
    if binding.cwd and "cwd" not in merged:
        merged["cwd"] = binding.cwd
    return merged, binding


def build_tool_call_arguments(
    descriptor: CapabilityDescriptor,
    query: str,
    *,
    max_results: int = 5,
    binding: McpConfiguredBinding | None = None,
) -> dict[str, Any]:
    """Map a user query + descriptor schema to MCP ``tools/call`` arguments."""
    schema = descriptor.input_schema if isinstance(descriptor.input_schema, dict) else {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}

    args: dict[str, Any] = {}
    q = (query or "").strip()
    root = (binding.root_path if binding else None) or "."
    action = str(descriptor.action or "").strip().lower()
    raw_ref = str(descriptor.raw_ref or "").strip().lower()

    if "query" in props:
        args["query"] = q
    if "path" in props:
        if action in {"list-directory", "list_directory"} or raw_ref == "list_directory":
            args["path"] = root
        elif "search" in action or raw_ref in {"search_files", "search-files"}:
            args["path"] = root
            if "pattern" in props:
                names = extract_search_names(q)
                if names:
                    args["pattern"] = _glob_pattern_for_name(names[0])
                else:
                    args["pattern"] = _search_pattern_from_query(q) or "**/*"
        elif action.startswith("read") or raw_ref.startswith("read"):
            args["path"] = _read_path_from_query(q, root)
        else:
            args["path"] = root
    if "pattern" in props and "pattern" not in args:
        names = extract_search_names(q)
        if names:
            args["pattern"] = _glob_pattern_for_name(names[0])
        else:
            args["pattern"] = _search_pattern_from_query(q) or "**/*"
    if "max_results" in props:
        args["max_results"] = max(1, int(max_results))
    if not args:
        args["query"] = q
    return args


def extract_search_names(query: str) -> list[str]:
    """Pull filename tokens from natural-language search requests."""
    match = _NAME_FILTER_RE.search(query or "")
    names: list[str] = []
    if match:
        for group in (match.group(1), match.group(2)):
            text = (group or "").strip()
            if text and text not in names:
                names.append(text)
    return names


def build_search_invoke_arg_sets(
    descriptor: CapabilityDescriptor,
    query: str,
    *,
    max_results: int = 5,
    binding: McpConfiguredBinding | None = None,
) -> list[dict[str, Any]]:
    """Build one or more MCP ``tools/call`` payloads for filename search."""
    action = str(descriptor.action or "").strip().lower()
    raw_ref = str(descriptor.raw_ref or "").strip().lower()
    if "search" not in action and raw_ref not in {"search_files", "search-files"}:
        return [
            build_tool_call_arguments(
                descriptor,
                query,
                max_results=max_results,
                binding=binding,
            )
        ]
    root = (binding.root_path if binding else None) or "."
    names = extract_search_names(query)
    if not names:
        return [
            build_tool_call_arguments(
                descriptor,
                query,
                max_results=max_results,
                binding=binding,
            )
        ]
    return [{"path": root, "pattern": _glob_pattern_for_name(name)} for name in names]


def _glob_pattern_for_name(name: str) -> str:
    token = (name or "").strip()
    if not token:
        return "**/*"
    # Filesystem MCP globs are case-sensitive on Windows; title-case bare lowercase tokens.
    if token.islower() and token.isalpha():
        token = token.title()
    return f"**/*{token}*"


def _root_path_from_command(command: list[Any]) -> str | None:
    for part in reversed(command):
        text = str(part or "").strip()
        if not text:
            continue
        if text.startswith(("/", "~")) or (len(text) > 2 and text[1] == ":"):
            return text
    return None


def _search_pattern_from_query(query: str) -> str | None:
    match = _NAME_FILTER_RE.search(query or "")
    if match:
        first = (match.group(1) or "").strip()
        second = (match.group(2) or "").strip()
        if first and second:
            return _glob_pattern_for_name(first)
        if first:
            return _glob_pattern_for_name(first)
    return None


def _read_path_from_query(query: str, root: str) -> str:
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", query or "")
    for candidate in reversed(quoted):
        text = candidate.strip()
        if text:
            return text
    return root
