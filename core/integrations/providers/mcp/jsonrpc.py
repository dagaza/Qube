"""Minimal JSON-RPC 2.0 helpers for the MCP stdio transport.

Deliberately tiny and dependency-free: MCP speaks JSON-RPC 2.0 over
newline-delimited JSON on stdio, and the official ``mcp`` SDK is *not* a
dependency of Qube. Keeping a hand-rolled encoder/decoder here (inside
``providers/mcp/``) honours P6 — no MCP protocol code leaks outside this
package — and avoids adding a heavy transitive dependency.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

__all__ = [
    "JsonRpcError",
    "next_id",
    "encode_request",
    "encode_notification",
    "decode_message",
]

# Process-wide monotonic id source. Ids only need to be unique per live session
# for request/response correlation; a global counter is simplest and safe.
_ID_COUNTER = itertools.count(1)


class JsonRpcError(RuntimeError):
    """A JSON-RPC ``error`` object returned by the peer.

    Carries the numeric ``code`` and optional ``data`` so callers can decide how
    to surface it, while ``str(err)`` yields a human-readable message.
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.rpc_message = message
        self.data = data


def next_id() -> int:
    """Return the next monotonic JSON-RPC request id."""
    return next(_ID_COUNTER)


def encode_request(method: str, params: dict[str, Any] | None, *, request_id: int) -> str:
    """Encode a JSON-RPC request as a single NDJSON line (no trailing newline)."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg, separators=(",", ":"))


def encode_notification(method: str, params: dict[str, Any] | None) -> str:
    """Encode a JSON-RPC notification (no id, no response expected)."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg, separators=(",", ":"))


def decode_message(line: str) -> dict[str, Any]:
    """Parse one NDJSON line into a JSON-RPC message dict.

    Raises :class:`ValueError` if the line is not a JSON object.
    """
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON-RPC object, got {type(obj).__name__}")
    return obj
