"""The single abstraction the Qube runtime knows about: ``CapabilityProvider``.

Every source of capabilities — the MCP client, Live Sources, local tools,
knowledge packs, future cloud/enterprise connectors — implements this protocol.
The registry, cognitive router, composer, evidence pipeline, and INSPECT depend
**only** on this interface and on the value objects in ``model.py``; they never
import a concrete provider (principle P6). Adding a new provider is therefore a
new folder under ``core/integrations/providers/`` implementing four methods,
with no changes required elsewhere (principle P5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    HealthStatus,
    NormalizedHit,
)
from core.integrations.capabilities.urn import CapabilityURN

__all__ = ["InvokeContext", "CapabilityProvider", "CapabilityInvocationError"]


class CapabilityInvocationError(RuntimeError):
    """Raised by a provider when an invocation fails in a well-defined way.

    Providers should raise this (rather than leaking transport-specific errors)
    so the runtime can render a clear, provider-agnostic message and record it in
    INSPECT / health. Permission denials, timeouts, and protocol errors are all
    expressed through this type with a descriptive message.
    """


@dataclass(frozen=True, slots=True)
class InvokeContext:
    """Per-invocation context passed from the runtime into a provider.

    Carries the information a provider needs to execute one capability call and
    the metadata the observability plane needs to attribute it. It never carries
    secrets; credentials are resolved by the provider from configured env-var
    names in the registry record.
    """

    query: str
    conversation_id: str | None = None
    turn_id: str | None = None
    max_results: int = 3
    timeout_s: float = 15.0
    # When True the provider must not perform side effects; used to preview a
    # write/destructive capability before the user confirms it (agent mode).
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CapabilityProvider(Protocol):
    """Contract implemented by every capability source.

    Implementations live under ``core/integrations/providers/<provider_id>/``.
    All methods that touch a transport are async; the runtime awaits them within
    its existing retrieval task orchestration.
    """

    #: Short, lowercase provider id. Must match the ``provider`` segment of the
    #: URNs this provider emits (e.g. ``"mcp"`` -> ``cap:mcp:...``).
    provider_id: str

    async def discover(self) -> list[CapabilityDescriptor]:
        """Connect if needed and return the provider's current capabilities.

        For MCP this performs ``initialize`` -> ``tools/list`` and maps raw tools
        into grouped :class:`CapabilityDescriptor` objects. Results are cached by
        the registry; ``discover`` may be called again on reconnect to detect
        drift (see :meth:`fingerprint`).
        """
        ...

    async def invoke(
        self,
        urn: CapabilityURN,
        args: dict[str, Any],
        *,
        ctx: InvokeContext,
    ) -> list[NormalizedHit]:
        """Invoke one capability and return normalized results.

        The runtime guarantees ``urn`` was granted and belongs to this provider
        before calling. Implementations must:

        * invoke only the raw tool(s) mapped to ``urn`` (never a broader set),
        * honour ``ctx.dry_run`` for write/destructive capabilities,
        * raise :class:`CapabilityInvocationError` on failure.
        """
        ...

    async def health(self) -> HealthStatus:
        """Return current connection health (feeds the Source Status UI)."""
        ...

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the currently discovered capabilities.

        Used for drift detection and to bind permission grants to the capability
        contract they were granted against. See
        ``model.fingerprint_descriptors``.
        """
        ...
