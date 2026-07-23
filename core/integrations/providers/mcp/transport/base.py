"""Transport contract for the MCP provider.

A transport hides *how* JSON-RPC messages travel (stdio subprocess today, HTTP
later) behind a small synchronous request/notify surface. The provider client
depends only on this protocol, so the same client works over any transport and
tests can inject an in-memory fake.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["Transport", "McpTransportError", "McpTimeoutError"]


class McpTransportError(RuntimeError):
    """A transport-level failure (spawn error, closed pipe, protocol violation)."""


class McpTimeoutError(McpTransportError):
    """A request exceeded its deadline before the peer responded."""


@runtime_checkable
class Transport(Protocol):
    """One MCP session's message channel.

    Implementations are synchronous; the async :class:`CapabilityProvider`
    methods drive them from within the runtime's retrieval task. Correlation of
    responses to requests (by JSON-RPC id) is the transport's responsibility.
    """

    @property
    def is_connected(self) -> bool:
        """True once the underlying channel is live and not yet closed."""
        ...

    def connect(self) -> None:
        """Start the channel (spawn the process / open the socket). Idempotent."""
        ...

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return its ``result`` object.

        Raises :class:`McpTimeoutError` if no response arrives within
        ``timeout_s`` and :class:`McpTransportError` (or
        :class:`~core.integrations.providers.mcp.jsonrpc.JsonRpcError`) on a
        transport/protocol failure or a JSON-RPC ``error`` response.
        """
        ...

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response)."""
        ...

    def close(self) -> None:
        """Shut the channel down gracefully. Idempotent and never raises."""
        ...
