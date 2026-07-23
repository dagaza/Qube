"""Concrete capability providers — the composition root.

Each provider lives in its own subpackage under this directory and implements
the :class:`core.integrations.capabilities.CapabilityProvider` protocol. The
rest of Qube depends only on that protocol and on the value objects in
``core/integrations/capabilities/`` — never on a concrete provider — so adding a
provider is a new folder here, with no changes required elsewhere (P5/P6).

This package ``__init__`` is the **only** module in Qube that imports concrete
provider classes. It exists so the provider-agnostic registry
(:mod:`core.integrations.registry`) can resolve providers by string id without
importing any of them. Provider-specific imports (transports, SDKs,
``provider == "..."`` branches) stay confined to the individual provider
subpackages.
"""

from __future__ import annotations

__all__: list[str] = ["register_builtin_providers"]


def register_builtin_providers() -> None:
    """Register every built-in capability provider with the registry.

    This is the single sanctioned place that imports a concrete provider class;
    the registry and the runtime resolve providers by ``provider_id`` and never
    import a provider (P5/P6). Idempotent — ``register_capability_provider``
    overwrites, so calling this repeatedly is safe.
    """
    from core.integrations.registry.provider_registry import (
        register_capability_provider,
    )
    from core.integrations.providers.mcp import (
        PROVIDER_ID as MCP_PROVIDER_ID,
        McpCapabilityProvider,
    )

    register_capability_provider(MCP_PROVIDER_ID, McpCapabilityProvider)
