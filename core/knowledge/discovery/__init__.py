"""URL discovery subsystem — returns CandidateUrl, not evidence."""

from core.knowledge.discovery.registry import (
    default_discovery_provider,
    discover,
    discover_full,
    discover_full_with_fallback,
    fallback_discovery_provider,
    get_discovery_provider,
    list_discovery_providers,
    register_discovery_provider,
)
from core.knowledge.discovery.types import CandidateUrl, DiscoveryProvider, DiscoveryResult

__all__ = [
    "CandidateUrl",
    "DiscoveryProvider",
    "DiscoveryResult",
    "default_discovery_provider",
    "discover",
    "discover_full",
    "discover_full_with_fallback",
    "fallback_discovery_provider",
    "get_discovery_provider",
    "list_discovery_providers",
    "register_discovery_provider",
]
