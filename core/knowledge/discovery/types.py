"""Discovery layer types — URLs only, not evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from core.knowledge.search_outcome import SearchOutcome


@dataclass(frozen=True)
class CandidateUrl:
    """A URL candidate from discovery; evidence exists only after fetch + extract."""

    url: str
    title: str | None
    snippet: str | None
    source: str  # "duckduckgo", "rss", "site_list", ...
    rank: int = 0


@dataclass(frozen=True)
class DiscoveryResult:
    """Discovery output including raw SERP rows and typed search outcome."""

    candidates: tuple[CandidateUrl, ...]
    raw_rows: tuple[dict, ...]
    search_outcome: SearchOutcome | None = None
    provider_id: str = "duckduckgo"
    discovery_cache_hit: bool = False
    discovery_pace_wait_ms: int = 0
    privacy_tier: str | None = None


class DiscoveryProvider(Protocol):
    """Pluggable URL discovery (parallel to adapter SearchFn)."""

    id: str

    def discover(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> list[CandidateUrl]: ...
