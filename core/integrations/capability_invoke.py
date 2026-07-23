"""Gated capability invocation for composer ``@[cap:…]`` attachments (T14).

Composer attach→invoke uses strict :func:`evaluate_access` (attach ≠ grant; no
ephemeral READ). The configured-source :class:`~core.knowledge.connectors.mcp_connector.McpConnector`
path synthesizes an ephemeral READ grant when none is stored — see
``.cursor/starfall/decisions.md`` (2026-07-23).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from core.integrations.capabilities import (
    CapabilityDescriptor,
    CapabilityURN,
    ConsentStore,
    InvokeContext,
    evaluate_access,
)
from core.integrations.capabilities.persistence import AccessDecision
from core.integrations.consent_controller import load_cached_descriptors
from core.integrations.registry.provider_registry import (
    UnknownCapabilityProvider,
    create_capability_provider,
    ensure_providers_registered,
)

logger = logging.getLogger("Qube.Integrations.Invoke")

__all__ = [
    "CapabilityInvokeResult",
    "evaluate_invoke_access",
    "parse_composer_capability_urn",
    "resolve_descriptor_for_urn",
    "invoke_gated_capability",
]


@dataclass(frozen=True, slots=True)
class CapabilityInvokeResult:
    """Outcome of a gated capability invocation."""

    allowed: bool
    reason: str
    rows: tuple[dict[str, Any], ...] = ()
    descriptor: CapabilityDescriptor | None = None
    urn: CapabilityURN | None = None

    @property
    def hits(self) -> list[dict[str, Any]]:
        return list(self.rows)


def evaluate_invoke_access(
    descriptor: CapabilityDescriptor,
    grant,
) -> AccessDecision:
    """Strict access check for composer attach→invoke (no synthetic grants)."""
    return evaluate_access(descriptor, grant)


def parse_composer_capability_urn(token_id: str) -> CapabilityURN | None:
    """Parse a composer cap token body into a :class:`CapabilityURN`."""
    raw = (token_id or "").strip()
    if not raw:
        return None
    if not raw.startswith("cap:"):
        raw = f"cap:{raw}"
    return CapabilityURN.try_parse(raw)


def resolve_descriptor_for_urn(
    urn: CapabilityURN,
    *,
    live_descriptors: list[CapabilityDescriptor] | None = None,
) -> CapabilityDescriptor | None:
    """Resolve a descriptor for ``urn`` from live or cached discovery."""
    sources: list[CapabilityDescriptor] = []
    if live_descriptors:
        sources.extend(live_descriptors)
    sources.extend(load_cached_descriptors(urn.provider))
    for descriptor in sources:
        if descriptor.urn.base != urn.base:
            continue
        if urn.version and descriptor.urn.version and descriptor.urn.version != urn.version:
            continue
        return descriptor
    return None


def _run(coro):
    return asyncio.run(coro)


def invoke_gated_capability(
    urn: CapabilityURN | str,
    query: str,
    *,
    max_results: int = 5,
    timeout_s: float = 15.0,
    provider_factory_kwargs: dict[str, Any] | None = None,
    adapter_id: str | None = None,
    live_descriptors: list[CapabilityDescriptor] | None = None,
) -> CapabilityInvokeResult:
    """Invoke a capability after strict consent evaluation."""
    if isinstance(urn, str):
        parsed = parse_composer_capability_urn(urn)
        if parsed is None:
            return CapabilityInvokeResult(False, "invalid capability URN")
        urn = parsed

    descriptor = resolve_descriptor_for_urn(urn, live_descriptors=live_descriptors)
    if descriptor is None:
        return CapabilityInvokeResult(
            False, f"capability not found: {urn}", urn=urn
        )

    store = ConsentStore(urn.provider)
    decision = evaluate_invoke_access(descriptor, store.get(descriptor.urn))
    if not decision.allowed:
        logger.info("[CapabilityInvoke] denied %s: %s", urn, decision.reason)
        return CapabilityInvokeResult(
            False, decision.reason, descriptor=descriptor, urn=urn
        )

    ensure_providers_registered()
    kwargs = dict(provider_factory_kwargs or {})
    kwargs.setdefault("namespace", urn.namespace)

    try:
        provider = create_capability_provider(urn.provider, **kwargs)
    except UnknownCapabilityProvider:
        return CapabilityInvokeResult(
            False,
            f"unknown provider {urn.provider!r}",
            descriptor=descriptor,
            urn=urn,
        )
    except Exception as exc:
        return CapabilityInvokeResult(
            False,
            f"provider init failed: {exc}",
            descriptor=descriptor,
            urn=urn,
        )

    try:
        if live_descriptors is None:
            discovered = _run(provider.discover())
            refreshed = resolve_descriptor_for_urn(urn, live_descriptors=discovered)
            if refreshed is not None:
                descriptor = refreshed
                decision = evaluate_invoke_access(
                    descriptor, store.get(descriptor.urn)
                )
                if not decision.allowed:
                    return CapabilityInvokeResult(
                        False,
                        decision.reason,
                        descriptor=descriptor,
                        urn=urn,
                    )

        invoke_urn = urn if urn.version else descriptor.urn
        ctx = InvokeContext(
            query=query,
            max_results=max_results,
            timeout_s=timeout_s,
        )
        hits = _run(
            provider.invoke(
                invoke_urn,
                {"query": query, "max_results": max_results},
                ctx=ctx,
            )
        )
    except Exception as exc:
        logger.warning("[CapabilityInvoke] invoke failed for %s: %s", urn, exc)
        return CapabilityInvokeResult(
            False, str(exc), descriptor=descriptor, urn=urn
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    short_adapter = adapter_id or urn.namespace
    rows: list[dict[str, Any]] = []
    for hit in hits:
        row = hit.to_evidence_dict()
        row["_adapter"] = short_adapter
        rows.append(row)

    return CapabilityInvokeResult(
        True, "ok", rows=tuple(rows), descriptor=descriptor, urn=urn
    )
