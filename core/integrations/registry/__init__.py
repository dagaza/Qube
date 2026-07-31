"""Capability control-plane registry.

Provider-agnostic resolution of :class:`CapabilityProvider` implementations by
string ``provider_id``. The runtime imports from here and never from a concrete
provider package (principles P5/P6). See ``provider_registry`` for details and
``docs/mcp_capability_architecture_review.md`` (§4 control plane, §8 layout).
"""

from __future__ import annotations

from core.integrations.registry.provider_registry import (
    ProviderFactory,
    UnknownCapabilityProvider,
    create_capability_provider,
    ensure_providers_registered,
    get_capability_provider_factory,
    is_provider_registered,
    list_capability_providers,
    register_capability_provider,
    reset_registry_for_tests,
)

__all__ = [
    "ProviderFactory",
    "UnknownCapabilityProvider",
    "register_capability_provider",
    "get_capability_provider_factory",
    "create_capability_provider",
    "list_capability_providers",
    "is_provider_registered",
    "ensure_providers_registered",
    "reset_registry_for_tests",
]
