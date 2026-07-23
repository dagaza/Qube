"""A tiny, deterministic stdio MCP server for transport tests.

Speaks just enough of the MCP JSON-RPC lifecycle (newline-delimited JSON on
stdio) to exercise :class:`StdioTransport` and :class:`McpCapabilityProvider`
without a network or the real ``mcp`` SDK:

* ``initialize``            -> minimal server info
* ``notifications/initialized`` (notification, no reply)
* ``tools/list``            -> a fixed set of tools
* ``tools/call``            -> a text content block echoing the query; the
                               ``search_slow`` tool sleeps so a client can prove
                               its per-request timeout fires.

Launch it with ``[sys.executable, <this file>]`` so it runs on any OS (no
shebang reliance), which matters on the Windows CI runner.
"""

from __future__ import annotations

import json
import sys
import time

_TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the documentation",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "create_doc",
        "description": "Create a documentation page",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "search_slow",
        "description": "Search but respond slowly (for timeout tests)",
        "inputSchema": {"type": "object"},
    },
]


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _result(request_id, result: dict) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _handle(msg: dict) -> None:
    method = msg.get("method")
    msg_id = msg.get("id")

    # Notifications carry no id and get no response.
    if msg_id is None:
        return

    if method == "initialize":
        _result(msg_id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "mock-mcp", "version": "0"},
            "capabilities": {"tools": {}},
        })
    elif method == "tools/list":
        _result(msg_id, {"tools": _TOOLS})
    elif method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "search_slow":
            time.sleep(30)  # exceeds any sane test timeout
        query = str(args.get("query") or "")
        _result(msg_id, {
            "content": [
                {
                    "type": "text",
                    "title": f"Result for {query}",
                    "text": f"You searched for: {query}",
                    "url": "https://example.test/doc/1",
                }
            ]
        })
    else:
        _error(msg_id, -32601, f"method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict):
            _handle(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
