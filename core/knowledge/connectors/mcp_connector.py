"""MCP connector — the bridge from a *configured source* to the MCP provider.

This connector no longer speaks JSON-RPC itself. It is the single, sanctioned
delegation point (per the capability architecture) from the legacy configured-
source retrieval spine into the real :class:`McpCapabilityProvider`, which owns
the persistent stdio session and the ``initialize`` -> ``tools/list`` ->
``tools/call`` lifecycle. Keeping one path (delegate, not fork) means the
handshake, timeouts, output caps, and provenance live in exactly one place.

Least privilege (P7): configuring a source is the user's explicit opt-in for its
*read* search tool, so read capabilities run. Anything write/destructive (or a
low-confidence ``needs_review`` classification) is default-denied here unless an
explicit consent grant is on record — it is never silently invoked.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger("Qube.Knowledge.Connectors.MCP")

_DEFAULT_TIMEOUT_SEC = 15.0

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async provider coroutine from this synchronous connector.

    The connector runs on a worker thread with no active event loop, so a fresh
    loop per call is correct and simplest; the provider's transport is
    thread/subprocess based and is not bound to any particular loop.
    """
    return asyncio.run(coro)


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
        namespace = str(config.get("namespace") or adapter_id)
        if not isinstance(command, list) or not command:
            return []

        # Imported lazily and locally: this is the one place the configured-source
        # spine reaches into the MCP provider package (P6 stays inside providers/mcp/).
        from core.integrations.capabilities import (
            CapabilityTier,
            ConsentStore,
            InvokeContext,
            evaluate_access,
            save_descriptor_cache,
        )
        from core.integrations.providers.mcp import McpCapabilityProvider
        from core.integrations.mcp_configured_source import augment_spawn_env_for_command

        cmd = [str(part) for part in command]
        spawn_env = augment_spawn_env_for_command(
            cmd,
            dict(config.get("env") or {}) if isinstance(config.get("env"), dict) else None,
        )
        provider = McpCapabilityProvider(
            command=cmd,
            namespace=namespace,
            env=spawn_env,
            cwd=str(config.get("cwd") or "").strip() or None,
        )
        try:
            descriptors = _run(provider.discover())
            if not descriptors:
                return []
            try:
                save_descriptor_cache(provider.provider_id, descriptors)
            except Exception as exc:  # cache is best-effort, never fatal
                logger.debug("[MCP] descriptor cache skipped: %s", exc)

            descriptor = self._resolve(descriptors, tool_name)
            if descriptor is None:
                logger.info("[MCP] no capability matched tool_name=%r", tool_name)
                return []

            if not self._is_permitted(descriptor, provider.provider_id, evaluate_access, ConsentStore, CapabilityTier):
                logger.warning(
                    "[MCP] denied %s (tier=%s needs_review=%s) — no grant",
                    descriptor.urn, descriptor.tier.value, descriptor.needs_review,
                )
                return []

            ctx = InvokeContext(
                query=query,
                max_results=max_results,
                timeout_s=min(timeout, _DEFAULT_TIMEOUT_SEC),
            )
            hits = _run(provider.invoke(descriptor.urn, {"query": query, "max_results": max_results}, ctx=ctx))
        except Exception as exc:
            logger.warning("[MCP] execute failed: %s", exc)
            return []
        finally:
            provider.close()

        rows: list[dict[str, Any]] = []
        for hit in hits:
            row = hit.to_evidence_dict()
            # Keep the short, stable configured id for authority/diversity keying
            # while preserving the full cap: URN as provenance (KI2).
            row["_adapter"] = adapter_id
            rows.append(row)
        return rows

    @staticmethod
    def _resolve(descriptors: list[Any], tool_name: str) -> Any | None:
        """Pick the capability for ``tool_name`` (exact raw tool, then action)."""
        for d in descriptors:
            if d.raw_ref == tool_name:
                return d
        for d in descriptors:
            if d.action == tool_name:
                return d
        # Fall back to a single obvious read search capability if present.
        reads = [d for d in descriptors if d.tier.value == "read" and not d.needs_review]
        if len(reads) == 1:
            return reads[0]
        return None

    @staticmethod
    def _is_permitted(descriptor, provider_id, evaluate_access, consent_store_cls, tier_cls) -> bool:
        """Configured-source opt-in routes all tiers through ``evaluate_access``."""
        from core.integrations.capabilities import PermissionGrant
        from core.integrations.capabilities.persistence import capability_fingerprint

        grant = consent_store_cls(provider_id).get(descriptor.urn)
        if (
            grant is None
            and descriptor.tier is tier_cls.READ
            and not descriptor.needs_review
        ):
            grant = PermissionGrant(
                urn=descriptor.urn.base,
                tier=descriptor.tier,
                granted=True,
                fingerprint=capability_fingerprint(descriptor),
            )
        return evaluate_access(descriptor, grant).allowed

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
        namespace = str(config.get("namespace") or config.get("adapter_id") or "configured_mcp")
        if not isinstance(command, list) or not command:
            return False, "MCP command not configured"

        from core.integrations.providers.mcp import McpCapabilityProvider
        from core.integrations.mcp_configured_source import augment_spawn_env_for_command

        cmd = [str(part) for part in command]
        spawn_env = augment_spawn_env_for_command(
            cmd,
            dict(config.get("env") or {}) if isinstance(config.get("env"), dict) else None,
        )
        provider = McpCapabilityProvider(
            command=cmd,
            namespace=namespace,
            env=spawn_env,
            cwd=str(config.get("cwd") or "").strip() or None,
        )
        try:
            descriptors = _run(provider.discover())
        except Exception as exc:
            return False, str(exc)
        finally:
            provider.close()
        return True, f"OK — MCP server responded ({len(descriptors)} capabilities)"
