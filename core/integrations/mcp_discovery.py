"""Discover MCP capabilities from a Knowledge custom source config."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.integrations.capabilities.model import CapabilityDescriptor
from core.integrations.descriptor_cache import merge_descriptor_cache_for_namespace
from core.integrations.mcp_configured_source import augment_spawn_env_for_command

logger = logging.getLogger("Qube.Integrations.MCP.Discovery")

PROVIDER_ID = "mcp"


def _run(coro):
    return asyncio.run(coro)


def discover_mcp_capabilities(
    config: dict[str, Any],
    *,
    namespace: str,
) -> tuple[list[CapabilityDescriptor], str | None]:
    """Run ``tools/list`` for one configured MCP source. Returns (descriptors, error)."""
    command = config.get("command")
    ns = (namespace or config.get("namespace") or config.get("adapter_id") or "").strip()
    if not isinstance(command, list) or not command:
        return [], "MCP command is not configured"
    if not ns:
        return [], "MCP namespace is not configured"

    from core.integrations.providers.mcp import McpCapabilityProvider

    cmd = [str(part) for part in command]
    spawn_env = augment_spawn_env_for_command(
        cmd,
        dict(config.get("env") or {}) if isinstance(config.get("env"), dict) else None,
    )
    provider = McpCapabilityProvider(
        command=cmd,
        namespace=ns.lower(),
        env=spawn_env,
        cwd=str(config.get("cwd") or "").strip() or None,
    )
    try:
        descriptors = _run(provider.discover())
    except Exception as exc:
        logger.warning("[MCP] discover failed for namespace=%s: %s", ns, exc)
        return [], str(exc)
    finally:
        provider.close()
    return list(descriptors or []), None


def discover_and_cache_mcp_source(
    config: dict[str, Any],
    *,
    namespace: str,
) -> tuple[int, str | None]:
    """Discover capabilities and merge them into the provider descriptor cache."""
    descriptors, error = discover_mcp_capabilities(config, namespace=namespace)
    if error:
        return 0, error
    ns = (namespace or config.get("namespace") or config.get("adapter_id") or "").strip().lower()
    merge_descriptor_cache_for_namespace(PROVIDER_ID, ns, descriptors)
    return len(descriptors), None
