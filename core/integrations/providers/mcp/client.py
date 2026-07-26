"""``McpCapabilityProvider`` — the first real :class:`CapabilityProvider`.

This is MCP-as-a-provider: it speaks the MCP JSON-RPC lifecycle
(``initialize`` -> ``notifications/initialized`` -> ``tools/list`` ->
``tools/call``) over a pluggable :class:`Transport`, maps discovered tools into
provider-agnostic :class:`CapabilityDescriptor` objects via the shared
:class:`CapabilityMapper`, and returns provenance-bearing :class:`NormalizedHit`
results from an invocation.

All MCP-specific knowledge (protocol methods, transport, result shapes) stays
inside this package (P6). The rest of Qube depends only on the
``CapabilityProvider`` protocol and the capability value objects (P5).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from core.integrations.capabilities import (
    CapabilityDescriptor,
    CapabilityInvocationError,
    CapabilityMapper,
    CapabilityTier,
    CapabilityURN,
    HealthState,
    HealthStatus,
    InvokeContext,
    NormalizedHit,
    RawTool,
    fingerprint_descriptors,
)
from core.integrations.capabilities.mapper import CapabilityMappingError
from core.integrations.providers.mcp.transport import (
    McpTimeoutError,
    McpTransportError,
    StdioTransport,
    Transport,
)

logger = logging.getLogger("Qube.Integrations.MCP.Client")


def _is_filesystem_search_capability(source_cap: CapabilityURN) -> bool:
    action = str(source_cap.action or "").strip().lower()
    return "search" in action


def _split_search_paths(text: str) -> list[str]:
    paths: list[str] = []
    for line in (text or "").splitlines():
        candidate = line.strip()
        if not candidate or candidate.lower() == "no matches found":
            continue
        paths.append(candidate)
    return paths


def _path_hit_title(path: str) -> str:
    normalized = path.rstrip("\\/")
    base = os.path.basename(normalized)
    return base or normalized

PROVIDER_ID = "mcp"
_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "qube", "version": "1"}
# Deadline for control-plane calls (handshake / tools-list / health probe).
_CONTROL_TIMEOUT_S = 15.0


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class McpCapabilityProvider:
    """A :class:`CapabilityProvider` backed by one MCP server session.

    One instance == one MCP server (one ``namespace``). Construct with either an
    explicit ``transport`` (injected in tests) or a ``command`` to spawn a local
    stdio server. ``discover`` connects and lists tools; ``invoke`` calls one
    mapped tool; the runtime is responsible for having checked the permission
    grant before calling ``invoke`` (see the persistence/consent layer).
    """

    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        namespace: str,
        command: list[str] | None = None,
        transport: Transport | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        tier_overrides: dict[str, CapabilityTier] | None = None,
    ) -> None:
        if transport is None and not command:
            raise ValueError("McpCapabilityProvider requires a command or a transport")
        self._namespace = namespace
        self._tier_overrides = tier_overrides
        self._transport: Transport = transport or StdioTransport(
            command or [], cwd=cwd, env=env
        )
        self._descriptors: list[CapabilityDescriptor] = []
        self._by_base_urn: dict[str, CapabilityDescriptor] = {}
        self._initialized = False
        self._last_error: str | None = None
        self._last_success_at: str | None = None
        self._last_invocation_at: str | None = None

    # -- discovery --------------------------------------------------------

    async def discover(self) -> list[CapabilityDescriptor]:
        """Run the MCP handshake, list tools, and map them to capabilities."""
        self._handshake()
        result = self._transport.request(
            "tools/list", {}, timeout_s=_CONTROL_TIMEOUT_S
        )
        raw_tools = self._parse_tools(result)
        try:
            group = CapabilityMapper().map_tools(
                self.provider_id,
                self._namespace,
                raw_tools,
                group_label=self._namespace,
                tier_overrides=self._tier_overrides,
            )
        except CapabilityMappingError as exc:
            raise CapabilityInvocationError(str(exc)) from exc
        self._descriptors = list(group.capabilities)
        self._by_base_urn = {str(d.urn.base): d for d in self._descriptors}
        self._last_success_at = _iso_now()
        return self._descriptors

    def _handshake(self) -> None:
        if not self._transport.is_connected:
            self._transport.connect()
        if self._initialized:
            return
        try:
            self._transport.request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
                timeout_s=_CONTROL_TIMEOUT_S,
            )
            self._transport.notify("notifications/initialized")
        except (McpTransportError, McpTimeoutError) as exc:
            self._last_error = str(exc)
            raise CapabilityInvocationError(f"MCP initialize failed: {exc}") from exc
        self._initialized = True

    @staticmethod
    def _parse_tools(result: dict[str, Any]) -> list[RawTool]:
        tools = result.get("tools")
        if not isinstance(tools, list):
            return []
        raw: list[RawTool] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            schema = tool.get("inputSchema")
            raw.append(
                RawTool(
                    name=name,
                    description=str(tool.get("description") or ""),
                    input_schema=dict(schema) if isinstance(schema, dict) else {},
                )
            )
        return raw

    # -- invocation -------------------------------------------------------

    async def invoke(
        self,
        urn: CapabilityURN,
        args: dict[str, Any],
        *,
        ctx: InvokeContext,
    ) -> list[NormalizedHit]:
        """Invoke exactly the tool mapped to ``urn`` and normalize its result."""
        if urn.provider != self.provider_id:
            raise CapabilityInvocationError(
                f"{urn} does not belong to provider {self.provider_id!r}"
            )
        descriptor = self._by_base_urn.get(str(urn.base))
        if descriptor is None:
            raise CapabilityInvocationError(f"unknown capability: {urn}")

        # Least-privilege: a write/destructive capability must not perform side
        # effects during a preview. The runtime sets ctx.dry_run before consent.
        if ctx.dry_run and descriptor.tier is not CapabilityTier.READ:
            return [
                NormalizedHit(
                    title=f"[dry-run] {urn.action}",
                    snippet=(
                        f"{descriptor.tier.value} capability {urn} would run with "
                        f"{sorted(args)} (no side effects performed)"
                    ),
                    source_cap=urn,
                )
            ]

        started = time.monotonic()
        try:
            result = self._transport.request(
                "tools/call",
                {"name": descriptor.raw_ref, "arguments": dict(args or {})},
                timeout_s=ctx.timeout_s,
            )
        except McpTimeoutError as exc:
            self._last_error = str(exc)
            raise CapabilityInvocationError(f"MCP call timed out: {exc}") from exc
        except (McpTransportError, Exception) as exc:  # includes JsonRpcError
            self._last_error = str(exc)
            raise CapabilityInvocationError(f"MCP call failed: {exc}") from exc

        self._last_invocation_at = _iso_now()
        self._last_success_at = self._last_invocation_at
        logger.debug("[MCP] invoked %s in %.1fms", urn, (time.monotonic() - started) * 1000)
        return self._normalize(result, source_cap=urn, max_results=ctx.max_results)

    @staticmethod
    def _normalize(
        result: dict[str, Any],
        *,
        source_cap: CapabilityURN,
        max_results: int,
    ) -> list[NormalizedHit]:
        content = result.get("content")
        if not isinstance(content, list):
            return []
        hits: list[NormalizedHit] = []
        limit = max(1, max_results)
        path_search = _is_filesystem_search_capability(source_cap)
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("snippet") or "")
                default_title = str(item.get("title") or text[:120])
                url = item.get("url")
            else:
                text = str(item)
                default_title = text[:120]
                url = None
            if not text:
                continue
            if text.strip().lower() == "no matches found":
                continue
            if path_search:
                for path in _split_search_paths(text):
                    hits.append(
                        NormalizedHit(
                            title=_path_hit_title(path),
                            snippet=path,
                            source_cap=source_cap,
                            url=str(url) if url else None,
                            full_text=None,
                        )
                    )
                    if len(hits) >= limit:
                        return hits
                continue
            hits.append(
                NormalizedHit(
                    title=default_title,
                    snippet=text[:600],
                    source_cap=source_cap,
                    url=str(url) if url else None,
                    full_text=None,
                )
            )
            if len(hits) >= limit:
                return hits
        return hits

    # -- observability ----------------------------------------------------

    async def health(self) -> HealthStatus:
        if not self._transport.is_connected:
            return HealthStatus(state=HealthState.DOWN, last_error=self._last_error)
        try:
            started = time.monotonic()
            self._transport.request("tools/list", {}, timeout_s=_CONTROL_TIMEOUT_S)
            latency_ms = (time.monotonic() - started) * 1000
            self._last_success_at = _iso_now()
            return HealthStatus(
                state=HealthState.OK,
                latency_ms=round(latency_ms, 2),
                last_success_at=self._last_success_at,
                last_invocation_at=self._last_invocation_at,
            )
        except Exception as exc:
            self._last_error = str(exc)
            return HealthStatus(
                state=HealthState.DEGRADED,
                last_error=str(exc),
                last_success_at=self._last_success_at,
                last_invocation_at=self._last_invocation_at,
            )

    def fingerprint(self) -> str:
        return fingerprint_descriptors(self._descriptors)

    # -- lifecycle helpers ------------------------------------------------

    @property
    def descriptors(self) -> list[CapabilityDescriptor]:
        return list(self._descriptors)

    def close(self) -> None:
        """Tear down the underlying transport. Safe to call more than once."""
        self._initialized = False
        try:
            self._transport.close()
        except Exception:  # pragma: no cover - close must never raise
            pass
