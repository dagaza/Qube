"""Discover MCP capabilities from a Knowledge custom source config."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from core.integrations.capabilities.model import CapabilityDescriptor
from core.integrations.capability_drift import (
    CapabilityDriftDiff,
    descriptors_for_namespace,
    diff_namespace_capabilities,
    has_material_drift,
)
from core.integrations.consent_controller import load_cached_descriptors
from core.integrations.descriptor_cache import merge_descriptor_cache_for_namespace

logger = logging.getLogger("Qube.Integrations.MCP.Discovery")

PROVIDER_ID = "mcp"

__all__ = [
    "McpDiscoveryResult",
    "discover_and_cache_mcp_source",
    "discover_mcp_capabilities",
]


@dataclass(frozen=True, slots=True)
class McpDiscoveryResult:
    count: int
    error: str | None
    namespace: str
    first_connect: bool
    drift: CapabilityDriftDiff | None
    descriptors: tuple[CapabilityDescriptor, ...]


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

    from core.mcp_filesystem_pro_features import require_pro_mcp_filesystem_for_config

    try:
        require_pro_mcp_filesystem_for_config(dict(config or {}), namespace=ns)
    except ValueError as exc:
        return [], str(exc)

    from core.integrations.providers.mcp import McpCapabilityProvider

    cmd = [str(part) for part in command]
    from core.integrations.mcp_configured_source import augment_spawn_env_for_command

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
) -> McpDiscoveryResult:
    """Discover capabilities, merge cache, and report drift for grant review."""
    ns = (namespace or config.get("namespace") or config.get("adapter_id") or "").strip().lower()
    before = load_cached_descriptors(PROVIDER_ID)
    first_connect = not descriptors_for_namespace(before, ns)

    descriptors, error = discover_mcp_capabilities(config, namespace=ns)
    if error:
        return McpDiscoveryResult(
            count=0,
            error=error,
            namespace=ns,
            first_connect=first_connect,
            drift=None,
            descriptors=(),
        )

    drift = diff_namespace_capabilities(before, descriptors, namespace=ns)
    merge_descriptor_cache_for_namespace(PROVIDER_ID, ns, descriptors)
    return McpDiscoveryResult(
        count=len(descriptors),
        error=None,
        namespace=ns,
        first_connect=first_connect,
        drift=drift if has_material_drift(drift) else None,
        descriptors=tuple(descriptors),
    )
