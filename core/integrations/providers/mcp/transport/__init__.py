"""MCP transports.

The transport owns the JSON-RPC request/response plumbing for one MCP session.
``stdio`` is implemented here; an HTTP transport can be added later behind the
same :class:`Transport` protocol without touching the provider client.
"""

from __future__ import annotations

from core.integrations.providers.mcp.transport.base import (
    McpTransportError,
    McpTimeoutError,
    Transport,
)
from core.integrations.providers.mcp.transport.stdio import StdioTransport

__all__ = ["Transport", "McpTransportError", "McpTimeoutError", "StdioTransport"]
