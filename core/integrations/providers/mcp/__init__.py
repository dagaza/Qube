"""MCP capability provider.

MCP is *a* provider behind the provider-agnostic Capability Plane, not the
architecture. Everything MCP-specific (JSON-RPC lifecycle, stdio transport,
result shapes) is confined to this package (P6); the rest of Qube depends only
on :class:`core.integrations.capabilities.CapabilityProvider`.
"""

from __future__ import annotations

from core.integrations.providers.mcp.client import (
    PROVIDER_ID,
    McpCapabilityProvider,
)

__all__ = ["McpCapabilityProvider", "PROVIDER_ID"]
