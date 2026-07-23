"""Provider-agnostic capability-provider registry (control plane).

Resolves a :class:`~core.integrations.capabilities.CapabilityProvider`
*implementation* by its string ``provider_id`` so the Qube runtime can construct
a provider without importing a concrete provider class (principles P5/P6).

Concrete providers are registered exactly once, from the **composition root**
(:mod:`core.integrations.providers`) — the only module in Qube allowed to import
a concrete provider. This module imports *nothing* provider-specific: no MCP SDK
import, no provider-equality branch, and no reference to a concrete provider
subpackage (principles P5/P6).

A *provider factory* is any callable returning a ``CapabilityProvider``. MCP's
provider is per-server (one instance == one MCP server / namespace), so the
registry stores the **factory** (the provider class itself is a valid factory)
and the caller supplies the per-server config as keyword arguments when it
constructs an instance via :func:`create_capability_provider`.
"""

from __future__ import annotations

import logging
from typing import Callable

from core.integrations.capabilities.protocol import CapabilityProvider

logger = logging.getLogger("Qube.Integrations.Registry")

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

#: A callable that returns a fresh :class:`CapabilityProvider` given per-instance
#: keyword config (e.g. the MCP ``command``/``namespace`` for one server).
ProviderFactory = Callable[..., CapabilityProvider]

_PROVIDERS: dict[str, ProviderFactory] = {}
_builtins_loaded = False


class UnknownCapabilityProvider(KeyError):
    """Raised when resolving/creating a provider id that is not registered."""

    def __init__(self, provider_id: str) -> None:
        super().__init__(provider_id)
        self.provider_id = provider_id

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"no capability provider registered for id {self.provider_id!r}"


def _norm(provider_id: str) -> str:
    return (provider_id or "").strip().lower()


def register_capability_provider(provider_id: str, factory: ProviderFactory) -> None:
    """Register a provider factory under ``provider_id`` (overwrites, idempotent).

    ``factory`` is any callable returning a :class:`CapabilityProvider` — the
    provider *class* is the common case. Only the composition root
    (:mod:`core.integrations.providers`) calls this with a concrete provider;
    this module never imports one.
    """
    pid = _norm(provider_id)
    if not pid:
        raise ValueError("provider_id is required")
    if not callable(factory):
        raise TypeError("provider factory must be callable")
    _PROVIDERS[pid] = factory
    logger.debug("[integrations] registered capability provider %r", pid)


def get_capability_provider_factory(provider_id: str) -> ProviderFactory | None:
    """Return the factory registered for ``provider_id``, or ``None``."""
    ensure_providers_registered()
    return _PROVIDERS.get(_norm(provider_id))


def create_capability_provider(provider_id: str, **kwargs) -> CapabilityProvider:
    """Construct a provider instance for ``provider_id`` from ``**kwargs`` config.

    Resolves the provider by string id only (no concrete import here) and calls
    its factory with the supplied per-server configuration. Raises
    :class:`UnknownCapabilityProvider` if the id is not registered.
    """
    factory = get_capability_provider_factory(provider_id)
    if factory is None:
        raise UnknownCapabilityProvider(_norm(provider_id))
    return factory(**kwargs)


def list_capability_providers() -> list[str]:
    """Return the sorted list of registered provider ids."""
    ensure_providers_registered()
    return sorted(_PROVIDERS)


def is_provider_registered(provider_id: str) -> bool:
    """Return whether ``provider_id`` resolves to a registered factory."""
    ensure_providers_registered()
    return _norm(provider_id) in _PROVIDERS


def ensure_providers_registered() -> None:
    """Load built-in providers once, via the composition root.

    Importing :mod:`core.integrations.providers` and calling its
    ``register_builtin_providers()`` is the single place that imports concrete
    provider classes. This module never imports a provider, so the
    provider-agnostic guarantee (P6) holds even though registration is triggered
    from here. Failures are non-fatal (logged) so a broken provider can't crash
    every registry lookup.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    try:
        from core.integrations.providers import register_builtin_providers

        register_builtin_providers()
    except Exception as exc:  # never let one provider crash all resolution
        _builtins_loaded = False  # allow a later retry
        logger.warning("[integrations] builtin provider registration failed: %s", exc)


def reset_registry_for_tests() -> None:
    """Clear all registrations and reload state (test helper).

    Mirrors the discovery registry's test affordance so a test can start from a
    clean slate and then either register fakes or trigger builtin registration.
    """
    global _builtins_loaded
    _PROVIDERS.clear()
    _builtins_loaded = False
