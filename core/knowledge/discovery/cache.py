"""Short-lived in-memory cache for successful discovery results."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import replace

from core.knowledge.discovery.query_normalization import normalize_discovery_query
from core.knowledge.discovery.types import DiscoveryResult
from core.knowledge.retrieval_profiles import get_profile_spec, normalize_profile_id
from core.knowledge.search_outcome import SearchOutcomeKind

DEFAULT_DISCOVERY_CACHE_TTL_SECONDS = 300
DEFAULT_DISCOVERY_CACHE_AGGRESSIVE_TTL_SECONDS = 600
DEFAULT_DISCOVERY_CACHE_MAX_ENTRIES = 48

_lock = threading.Lock()
_entries: OrderedDict[str, tuple[float, DiscoveryResult]] = OrderedDict()


def discovery_cache_enabled() -> bool:
    raw = os.getenv("QUBE_DISCOVERY_CACHE")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def discovery_cache_ttl_seconds() -> int:
    raw = os.getenv("QUBE_DISCOVERY_CACHE_TTL")
    if raw is None:
        return DEFAULT_DISCOVERY_CACHE_TTL_SECONDS
    try:
        return max(0, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_DISCOVERY_CACHE_TTL_SECONDS


def discovery_cache_aggressive_ttl_seconds() -> int:
    raw = os.getenv("QUBE_DISCOVERY_CACHE_AGGRESSIVE_TTL")
    if raw is None:
        return DEFAULT_DISCOVERY_CACHE_AGGRESSIVE_TTL_SECONDS
    try:
        return max(0, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_DISCOVERY_CACHE_AGGRESSIVE_TTL_SECONDS


def discovery_cache_ttl_for_profile(profile_id: str | None) -> int:
    """Return SERP cache TTL from retrieval profile cache policy."""
    profile = get_profile_spec(normalize_profile_id(profile_id))
    if profile.cache_policy == "aggressive":
        return discovery_cache_aggressive_ttl_seconds()
    return discovery_cache_ttl_seconds()


def discovery_cache_max_entries() -> int:
    raw = os.getenv("QUBE_DISCOVERY_CACHE_MAX")
    if raw is None:
        return DEFAULT_DISCOVERY_CACHE_MAX_ENTRIES
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_DISCOVERY_CACHE_MAX_ENTRIES


def _cache_key(
    provider_id: str,
    query: str,
    *,
    max_results: int,
    site_bias: tuple[str, ...] | None,
) -> str:
    normalized_query = normalize_discovery_query(query)
    sites = ",".join(sorted(str(s).strip().lower() for s in (site_bias or ())))
    return "|".join(
        [
            (provider_id or "").strip().lower(),
            normalized_query,
            str(int(max_results)),
            sites,
        ]
    )


def _should_cache(result: DiscoveryResult) -> bool:
    outcome = result.search_outcome
    if outcome is None:
        return False
    if outcome.kind != SearchOutcomeKind.SERP_SUCCESS:
        return False
    return bool(result.candidates)


def get_cached_discovery(
    provider_id: str,
    query: str,
    *,
    max_results: int,
    site_bias: tuple[str, ...] | None,
    retrieval_profile: str | None = None,
) -> DiscoveryResult | None:
    if not discovery_cache_enabled():
        return None
    key = _cache_key(provider_id, query, max_results=max_results, site_bias=site_bias)
    ttl = discovery_cache_ttl_for_profile(retrieval_profile)
    if ttl <= 0:
        return None
    with _lock:
        row = _entries.get(key)
        if row is None:
            return None
        expires_at, result = row
        if time.time() >= expires_at:
            _entries.pop(key, None)
            return None
        _entries.move_to_end(key)
        return replace(result, discovery_cache_hit=True)


def store_cached_discovery(
    provider_id: str,
    query: str,
    *,
    max_results: int,
    site_bias: tuple[str, ...] | None,
    result: DiscoveryResult,
    retrieval_profile: str | None = None,
) -> None:
    if not discovery_cache_enabled() or not _should_cache(result):
        return
    ttl = discovery_cache_ttl_for_profile(retrieval_profile)
    if ttl <= 0:
        return
    key = _cache_key(provider_id, query, max_results=max_results, site_bias=site_bias)
    expires_at = time.time() + ttl
    with _lock:
        _entries[key] = (expires_at, result)
        _entries.move_to_end(key)
        max_entries = discovery_cache_max_entries()
        while len(_entries) > max_entries:
            _entries.popitem(last=False)


def reset_discovery_cache() -> None:
    """Clear cache entries (tests only)."""
    with _lock:
        _entries.clear()
