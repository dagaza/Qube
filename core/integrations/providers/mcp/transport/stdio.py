"""Persistent stdio JSON-RPC transport for local MCP servers.

Spawns the server once with :class:`subprocess.Popen`, keeps the pipes open for
the life of the session, and correlates responses to requests by JSON-RPC id via
a dedicated reader thread. This replaces the proof-of-concept one-shot
``subprocess.run`` path (which spawned a fresh process per call and could not
honour the MCP ``initialize`` handshake) with a real session.

Framing is MCP stdio's newline-delimited JSON: one JSON-RPC object per line on
stdin, one per line on stdout. Output is capped (per-line and cumulative) so a
runaway server cannot exhaust memory, and every request carries a deadline.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Any

from core.integrations.providers.mcp.jsonrpc import (
    JsonRpcError,
    decode_message,
    encode_notification,
    encode_request,
    next_id,
)
from core.integrations.providers.mcp.transport.base import (
    McpTimeoutError,
    McpTransportError,
)

logger = logging.getLogger("Qube.Integrations.MCP.Stdio")

_DEFAULT_MAX_OUTPUT_BYTES = 524_288
_SHUTDOWN_GRACE_S = 2.0


class _Pending:
    """A slot awaiting a response for one request id."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: Exception | None = None


def _no_window_flags() -> int:
    """Avoid a console window flashing on Windows when spawning the server."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


class StdioTransport:
    """A single persistent stdio MCP session.

    Not safe for concurrent use by multiple threads issuing requests; the runtime
    drives one invocation at a time per provider instance. The reader thread runs
    independently and is the only reader of ``stdout``.
    """

    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        if not isinstance(command, list) or not command:
            raise McpTransportError("MCP stdio command must be a non-empty list")
        self._command = [str(part) for part in command]
        self._cwd = cwd
        self._env = env
        self._max_output_bytes = max_output_bytes

        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, _Pending] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._bytes_seen = 0

    # -- lifecycle --------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and not self._closed
        )

    def connect(self) -> None:
        if self.is_connected:
            return
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered
                cwd=self._cwd,
                env=self._env,
                creationflags=_no_window_flags(),
            )
        except Exception as exc:  # spawn failure (bad command, permission, ...)
            raise McpTransportError(f"failed to start MCP server: {exc}") from exc

        self._closed = False
        self._bytes_seen = 0
        self._reader = threading.Thread(
            target=self._read_loop,
            name="mcp-stdio-reader",
            daemon=True,
        )
        self._reader.start()

    # -- reader thread ----------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                self._bytes_seen += len(line.encode("utf-8", "ignore"))
                if self._bytes_seen > self._max_output_bytes:
                    self._fail_all(McpTransportError("MCP output exceeded cap"))
                    return
                try:
                    msg = decode_message(line)
                except ValueError:
                    # A non-JSON line (e.g. a stray server log) — ignore it.
                    continue
                self._dispatch(msg)
        except Exception as exc:  # pipe closed / decode crash
            self._fail_all(McpTransportError(f"MCP reader stopped: {exc}"))
            return
        # stdout closed: the server exited. Wake anything still waiting.
        self._fail_all(McpTransportError("MCP server closed the connection"))

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        if msg_id is None:
            # Server-initiated request or notification — not supported yet; ignore.
            return
        with self._lock:
            pending = self._pending.pop(int(msg_id), None)
        if pending is None:
            return
        if "error" in msg and msg["error"] is not None:
            err = msg["error"] or {}
            pending.error = JsonRpcError(
                int(err.get("code", -1)),
                str(err.get("message", "unknown error")),
                err.get("data"),
            )
        else:
            result = msg.get("result")
            pending.result = result if isinstance(result, dict) else {"value": result}
        pending.event.set()

    def _fail_all(self, error: Exception) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot.error = error
            slot.event.set()

    # -- messaging --------------------------------------------------------

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        if not self.is_connected:
            raise McpTransportError("transport is not connected")
        request_id = next_id()
        slot = _Pending()
        with self._lock:
            self._pending[request_id] = slot
        self._write(encode_request(method, params, request_id=request_id))

        if not slot.event.wait(timeout=max(0.0, timeout_s)):
            with self._lock:
                self._pending.pop(request_id, None)
            raise McpTimeoutError(f"MCP request {method!r} timed out after {timeout_s}s")
        if slot.error is not None:
            raise slot.error
        return slot.result or {}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self.is_connected:
            raise McpTransportError("transport is not connected")
        self._write(encode_notification(method, params))

    def _write(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpTransportError("transport stdin is not available")
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise McpTransportError(f"failed to write to MCP server: {exc}") from exc

    # -- shutdown ---------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_SHUTDOWN_GRACE_S)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=_SHUTDOWN_GRACE_S)
                    except Exception:
                        pass
        finally:
            self._fail_all(McpTransportError("transport closed"))
            self._proc = None
