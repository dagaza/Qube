"""Gated capability invocation for composer ``@[cap:…]`` attachments (T14).

Composer attach→invoke uses strict :func:`evaluate_access` (attach ≠ grant; no
ephemeral READ). The configured-source :class:`~core.knowledge.connectors.mcp_connector.McpConnector`
path synthesizes an ephemeral READ grant when none is stored — see
``.cursor/starfall/decisions.md`` (2026-07-23).

Phase 3 (#61) adds agent-scope enforcement (P1), per-step approval for
write/destructive tiers, dry-run preview, and session egress recording.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from core.integrations.agent_scope import AgentScope, agent_scope_store
from core.integrations.capabilities import (
    CapabilityDescriptor,
    CapabilityURN,
    ConsentStore,
    InvokeContext,
    evaluate_access,
)
from core.integrations.capabilities.persistence import AccessDecision
from core.integrations.consent_controller import load_cached_descriptors
from core.integrations.egress_summary import include_raw_tools_in_egress
from core.integrations.mcp_configured_source import (
    McpConfiguredBinding,
    build_tool_call_arguments,
    build_search_invoke_arg_sets,
    merge_mcp_factory_kwargs,
)
from core.integrations.registry.provider_registry import (
    UnknownCapabilityProvider,
    create_capability_provider,
    ensure_providers_registered,
)
from core.integrations.session_egress import build_egress_record, session_egress_ledger
from core.integrations.step_approval import requires_step_approval, step_approval_store

logger = logging.getLogger("Qube.Integrations.Invoke")

__all__ = [
    "CapabilityInvokeResult",
    "evaluate_invoke_access",
    "parse_composer_capability_urn",
    "preview_gated_capability",
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
    dry_run: bool = False

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


def _user_facing_invoke_denial(urn: CapabilityURN, *, fallback: str = "") -> str:
    from core.integrations.capability_availability import resolve_capability_availability

    message = resolve_capability_availability(urn).user_message.strip()
    return message or fallback


def _record_egress(
    *,
    session_id: str | None,
    turn_id: str | None,
    urn: CapabilityURN,
    descriptor: CapabilityDescriptor | None,
    allowed: bool,
    reason: str,
    latency_ms: float,
    dry_run: bool,
) -> None:
    if not session_id or not turn_id:
        return
    record = build_egress_record(
        session_id=session_id,
        turn_id=turn_id,
        urn=urn,
        descriptor=descriptor,
        allowed=allowed,
        reason=reason,
        latency_ms=latency_ms,
        dry_run=dry_run,
        include_raw_tool=include_raw_tools_in_egress(),
    )
    session_egress_ledger.record(record)


def _check_agent_scope(
    urn: CapabilityURN,
    *,
    session_id: str | None,
    agent_scope: AgentScope | None,
    enforce_scope: bool,
) -> tuple[bool, str]:
    if not enforce_scope or not session_id:
        return True, "ok"
    scope = agent_scope
    if scope is None:
        scope = agent_scope_store.get_scope(session_id)
    if scope is None:
        return True, "ok"
    return scope.check(urn)


def _check_step_approval(
    descriptor: CapabilityDescriptor,
    urn: CapabilityURN,
    *,
    session_id: str | None,
    turn_id: str | None,
    step_approved: bool,
) -> tuple[bool, str]:
    if not requires_step_approval(descriptor):
        return True, "ok"
    if step_approved:
        return True, "ok"
    if session_id and turn_id and step_approval_store.has_approval(
        session_id, turn_id, urn
    ):
        return True, "ok"
    return False, "step approval required for write/destructive capability"


def preview_gated_capability(
    urn: CapabilityURN | str,
    query: str,
    *,
    max_results: int = 5,
    timeout_s: float = 15.0,
    provider_factory_kwargs: dict[str, Any] | None = None,
    live_descriptors: list[CapabilityDescriptor] | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> CapabilityInvokeResult:
    """Dry-run preview for write/destructive capabilities (no side effects)."""
    return invoke_gated_capability(
        urn,
        query,
        max_results=max_results,
        timeout_s=timeout_s,
        provider_factory_kwargs=provider_factory_kwargs,
        live_descriptors=live_descriptors,
        session_id=session_id,
        turn_id=turn_id,
        dry_run=True,
        enforce_scope=False,
        record_egress=False,
    )


def invoke_gated_capability(
    urn: CapabilityURN | str,
    query: str,
    *,
    max_results: int = 5,
    timeout_s: float = 15.0,
    provider_factory_kwargs: dict[str, Any] | None = None,
    adapter_id: str | None = None,
    live_descriptors: list[CapabilityDescriptor] | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    dry_run: bool = False,
    step_approved: bool = False,
    agent_scope: AgentScope | None = None,
    enforce_scope: bool = True,
    record_egress: bool = True,
) -> CapabilityInvokeResult:
    """Invoke a capability after strict consent, scope, and step-approval checks."""
    t0 = time.time()
    if isinstance(urn, str):
        parsed = parse_composer_capability_urn(urn)
        if parsed is None:
            return CapabilityInvokeResult(False, "invalid capability URN")
        urn = parsed

    descriptor = resolve_descriptor_for_urn(urn, live_descriptors=live_descriptors)
    if descriptor is None:
        reason = _user_facing_invoke_denial(urn, fallback=f"capability not found: {urn}")
        result = CapabilityInvokeResult(
            False, reason, urn=urn, dry_run=dry_run
        )
        if record_egress and session_id and turn_id:
            _record_egress(
                session_id=session_id,
                turn_id=turn_id,
                urn=urn,
                descriptor=None,
                allowed=False,
                reason=result.reason,
                latency_ms=(time.time() - t0) * 1000.0,
                dry_run=dry_run,
            )
        return result

    scope_ok, scope_reason = _check_agent_scope(
        urn,
        session_id=session_id,
        agent_scope=agent_scope,
        enforce_scope=enforce_scope,
    )
    if not scope_ok:
        logger.info("[CapabilityInvoke] out of scope %s: %s", urn, scope_reason)
        result = CapabilityInvokeResult(
            False, scope_reason, descriptor=descriptor, urn=urn, dry_run=dry_run
        )
        if record_egress and session_id and turn_id:
            _record_egress(
                session_id=session_id,
                turn_id=turn_id,
                urn=urn,
                descriptor=descriptor,
                allowed=False,
                reason=scope_reason,
                latency_ms=(time.time() - t0) * 1000.0,
                dry_run=dry_run,
            )
        return result

    store = ConsentStore(urn.provider)
    decision = evaluate_invoke_access(descriptor, store.get(descriptor.urn))
    if not decision.allowed:
        logger.info("[CapabilityInvoke] denied %s: %s", urn, decision.reason)
        reason = _user_facing_invoke_denial(urn, fallback=decision.reason)
        result = CapabilityInvokeResult(
            False,
            reason,
            descriptor=descriptor,
            urn=urn,
            dry_run=dry_run,
        )
        if record_egress and session_id and turn_id:
            _record_egress(
                session_id=session_id,
                turn_id=turn_id,
                urn=urn,
                descriptor=descriptor,
                allowed=False,
                reason=decision.reason,
                latency_ms=(time.time() - t0) * 1000.0,
                dry_run=dry_run,
            )
        return result

    if not dry_run:
        step_ok, step_reason = _check_step_approval(
            descriptor,
            urn,
            session_id=session_id,
            turn_id=turn_id,
            step_approved=step_approved,
        )
        if not step_ok:
            logger.info("[CapabilityInvoke] step approval required %s", urn)
            result = CapabilityInvokeResult(
                False,
                step_reason,
                descriptor=descriptor,
                urn=urn,
                dry_run=False,
            )
            if record_egress and session_id and turn_id:
                _record_egress(
                    session_id=session_id,
                    turn_id=turn_id,
                    urn=urn,
                    descriptor=descriptor,
                    allowed=False,
                    reason=step_reason,
                    latency_ms=(time.time() - t0) * 1000.0,
                    dry_run=False,
                )
            return result

    ensure_providers_registered()
    kwargs = dict(provider_factory_kwargs or {})
    kwargs.setdefault("namespace", urn.namespace)
    kwargs, mcp_binding = merge_mcp_factory_kwargs(urn.provider, urn.namespace, kwargs)
    if mcp_binding is not None and adapter_id is None:
        adapter_id = mcp_binding.adapter_id

    try:
        provider = create_capability_provider(urn.provider, **kwargs)
    except UnknownCapabilityProvider:
        result = CapabilityInvokeResult(
            False,
            f"unknown provider {urn.provider!r}",
            descriptor=descriptor,
            urn=urn,
            dry_run=dry_run,
        )
        if record_egress and session_id and turn_id:
            _record_egress(
                session_id=session_id,
                turn_id=turn_id,
                urn=urn,
                descriptor=descriptor,
                allowed=False,
                reason=result.reason,
                latency_ms=(time.time() - t0) * 1000.0,
                dry_run=dry_run,
            )
        return result
    except Exception as exc:
        result = CapabilityInvokeResult(
            False,
            f"provider init failed: {exc}",
            descriptor=descriptor,
            urn=urn,
            dry_run=dry_run,
        )
        if record_egress and session_id and turn_id:
            _record_egress(
                session_id=session_id,
                turn_id=turn_id,
                urn=urn,
                descriptor=descriptor,
                allowed=False,
                reason=result.reason,
                latency_ms=(time.time() - t0) * 1000.0,
                dry_run=dry_run,
            )
        return result

    try:
        should_discover = live_descriptors is None or mcp_binding is not None
        if should_discover:
            discovered = _run(provider.discover())
            if discovered and mcp_binding is not None:
                try:
                    from core.integrations.descriptor_cache import (
                        merge_descriptor_cache_for_namespace,
                    )

                    merge_descriptor_cache_for_namespace(
                        urn.provider,
                        urn.namespace,
                        list(discovered),
                    )
                except Exception as exc:
                    logger.debug("[CapabilityInvoke] descriptor cache skipped: %s", exc)
            refreshed = resolve_descriptor_for_urn(urn, live_descriptors=discovered)
            if refreshed is not None:
                descriptor = refreshed
                decision = evaluate_invoke_access(
                    descriptor, store.get(descriptor.urn)
                )
                if not decision.allowed:
                    reason = _user_facing_invoke_denial(urn, fallback=decision.reason)
                    result = CapabilityInvokeResult(
                        False,
                        reason,
                        descriptor=descriptor,
                        urn=urn,
                        dry_run=dry_run,
                    )
                    if record_egress and session_id and turn_id:
                        _record_egress(
                            session_id=session_id,
                            turn_id=turn_id,
                            urn=urn,
                            descriptor=descriptor,
                            allowed=False,
                            reason=decision.reason,
                            latency_ms=(time.time() - t0) * 1000.0,
                            dry_run=dry_run,
                        )
                    return result
            elif mcp_binding is not None:
                reason = _user_facing_invoke_denial(
                    urn,
                    fallback=f"capability not available on MCP server: {urn}",
                )
                result = CapabilityInvokeResult(
                    False,
                    reason,
                    descriptor=descriptor,
                    urn=urn,
                    dry_run=dry_run,
                )
                if record_egress and session_id and turn_id:
                    _record_egress(
                        session_id=session_id,
                        turn_id=turn_id,
                        urn=urn,
                        descriptor=descriptor,
                        allowed=False,
                        reason=result.reason,
                        latency_ms=(time.time() - t0) * 1000.0,
                        dry_run=dry_run,
                    )
                return result

        invoke_urn = urn if urn.version else descriptor.urn
        invoke_arg_sets = build_search_invoke_arg_sets(
            descriptor,
            query,
            max_results=max_results,
            binding=mcp_binding,
        )
        ctx = InvokeContext(
            query=query,
            max_results=max_results,
            timeout_s=timeout_s,
            conversation_id=session_id,
            turn_id=str(turn_id) if turn_id is not None else None,
            dry_run=dry_run,
        )
        hits = []
        seen_snippets: set[str] = set()
        for invoke_args in invoke_arg_sets:
            batch = _run(
                provider.invoke(
                    invoke_urn,
                    invoke_args,
                    ctx=ctx,
                )
            )
            for hit in batch:
                key = str(hit.snippet or hit.title or "").strip()
                if not key or key in seen_snippets:
                    continue
                seen_snippets.add(key)
                hits.append(hit)
                if len(hits) >= max_results:
                    break
            if len(hits) >= max_results:
                break
    except Exception as exc:
        logger.warning("[CapabilityInvoke] invoke failed for %s: %s", urn, exc)
        result = CapabilityInvokeResult(
            False, str(exc), descriptor=descriptor, urn=urn, dry_run=dry_run
        )
        if record_egress and session_id and turn_id:
            _record_egress(
                session_id=session_id,
                turn_id=turn_id,
                urn=urn,
                descriptor=descriptor,
                allowed=False,
                reason=str(exc),
                latency_ms=(time.time() - t0) * 1000.0,
                dry_run=dry_run,
            )
        return result
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

    latency_ms = (time.time() - t0) * 1000.0
    result = CapabilityInvokeResult(
        True, "ok", rows=tuple(rows), descriptor=descriptor, urn=urn, dry_run=dry_run
    )
    if record_egress and session_id and turn_id:
        _record_egress(
            session_id=session_id,
            turn_id=turn_id,
            urn=urn,
            descriptor=descriptor,
            allowed=True,
            reason="ok",
            latency_ms=latency_ms,
            dry_run=dry_run,
        )
    return result
