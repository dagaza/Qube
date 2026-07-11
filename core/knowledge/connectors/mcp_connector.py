"""MCP connector — subprocess boundary for local MCP servers."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

logger = logging.getLogger("Qube.Knowledge.Connectors.MCP")

_MAX_OUTPUT_BYTES = 524_288
_DEFAULT_TIMEOUT_SEC = 15.0


class McpConnector:
    id = "mcp"
    trust_policy = "enterprise"

    def execute(
        self,
        query: str,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        max_results: int = 3,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        _ = auth, egress_policy
        command = config.get("command")
        tool_name = str(config.get("tool_name") or "search")
        adapter_id = str(config.get("adapter_id") or "configured_mcp")
        if not isinstance(command, list) or not command:
            return []

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": {"query": query, "max_results": max_results}},
        }
        try:
            proc = subprocess.run(
                [str(x) for x in command],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=min(timeout, _DEFAULT_TIMEOUT_SEC),
            )
        except Exception as exc:
            logger.warning("[MCP] subprocess failed: %s", exc)
            return []

        if proc.returncode != 0:
            logger.warning("[MCP] non-zero exit: %s", proc.stderr[:500])
            return []
        if len(proc.stdout.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            logger.warning("[MCP] output too large")
            return []

        try:
            response = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []

        content = response.get("result", {}).get("content")
        if not isinstance(content, list):
            return []

        rows: list[dict[str, Any]] = []
        for item in content[: max(1, max_results)]:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("snippet") or "")
                title = str(item.get("title") or text[:120])
                url = item.get("url")
            else:
                text = str(item)
                title = text[:120]
                url = None
            if not text:
                continue
            rows.append(
                {
                    "title": title,
                    "snippet": text[:600],
                    "full_text": None,
                    "url": str(url) if url else None,
                    "_adapter": adapter_id,
                    "retrieval_method": "mcp",
                }
            )
        return rows

    def test_connection(
        self,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        _ = auth, egress_policy
        command = config.get("command")
        if not isinstance(command, list) or not command:
            return False, "MCP command not configured"
        try:
            proc = subprocess.run(
                [str(x) for x in command],
                input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
                capture_output=True,
                text=True,
                timeout=min(timeout, _DEFAULT_TIMEOUT_SEC),
            )
        except Exception as exc:
            return False, str(exc)
        if proc.returncode != 0:
            return False, proc.stderr[:300] or "MCP initialize failed"
        return True, "OK — MCP server responded"
