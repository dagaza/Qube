"""Retrieval profiles — user-facing orchestration policies (not domain ranking)."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.types import RetrievalBudget

PROFILE_FAST = "fast"
PROFILE_BALANCED = "balanced"
PROFILE_THOROUGH = "thorough"
PROFILE_EVIDENCE_FIRST = "evidence_first"
PROFILE_LOCAL_FIRST = "local_first"

DEFAULT_RETRIEVAL_PROFILE = PROFILE_BALANCED

VALID_PROFILE_IDS = frozenset(
    {
        PROFILE_FAST,
        PROFILE_BALANCED,
        PROFILE_THOROUGH,
        PROFILE_EVIDENCE_FIRST,
        PROFILE_LOCAL_FIRST,
    }
)

_LOCAL_CONNECTORS = frozenset({"sqlite", "filesystem"})


@dataclass(frozen=True)
class RetrievalProfileSpec:
    id: str
    label: str
    short_description: str
    budget: RetrievalBudget
    max_parallel_adapters: int
    cache_policy: str  # aggressive | default | bypass
    source_ordering: str  # default | local_before_remote
    ranking_profile_hint: str | None = None
    fetch_url_count: int = 0
    playwright_allowed: bool = False
    pagination_allowed: bool = False

    def materialize_budget(self, base: RetrievalBudget | None = None) -> RetrievalBudget:
        """Merge profile budget caps with service defaults."""
        if base is None:
            return self.budget
        return RetrievalBudget(
            max_results=min(base.max_results, self.budget.max_results)
            if self.budget.max_results
            else base.max_results,
            max_adapter_calls=max(base.max_adapter_calls, self.budget.max_adapter_calls)
            if self.budget.max_adapter_calls
            else base.max_adapter_calls,
            max_fetch_bytes=base.max_fetch_bytes or self.budget.max_fetch_bytes,
            max_latency_ms=min(base.max_latency_ms, self.budget.max_latency_ms)
            if self.budget.max_latency_ms
            else base.max_latency_ms,
        )


_PROFILES: dict[str, RetrievalProfileSpec] = {
    PROFILE_FAST: RetrievalProfileSpec(
        id=PROFILE_FAST,
        label="Fast",
        short_description="SERP snippets only — no page fetch (lowest latency).",
        budget=RetrievalBudget(max_results=2, max_adapter_calls=2, max_latency_ms=3500),
        max_parallel_adapters=2,
        cache_policy="aggressive",
        source_ordering="default",
        fetch_url_count=0,
    ),
    PROFILE_BALANCED: RetrievalProfileSpec(
        id=PROFILE_BALANCED,
        label="Balanced",
        short_description="Default orchestration — SERP plus fetch top page when relevant.",
        budget=RetrievalBudget(
            max_results=3,
            max_adapter_calls=3,
            max_latency_ms=8000,
            max_fetch_bytes=524_288,
        ),
        max_parallel_adapters=3,
        cache_policy="default",
        source_ordering="default",
        fetch_url_count=1,
    ),
    PROFILE_THOROUGH: RetrievalProfileSpec(
        id=PROFILE_THOROUGH,
        label="Thorough",
        short_description="Wider fan-out and fetch up to three pages for higher recall.",
        budget=RetrievalBudget(
            max_results=5,
            max_adapter_calls=6,
            max_latency_ms=15000,
            max_fetch_bytes=524_288,
        ),
        max_parallel_adapters=4,
        cache_policy="default",
        source_ordering="default",
        fetch_url_count=3,
    ),
    PROFILE_EVIDENCE_FIRST: RetrievalProfileSpec(
        id=PROFILE_EVIDENCE_FIRST,
        label="Evidence-first",
        short_description="Prioritize high-confidence sources and citation quality.",
        budget=RetrievalBudget(max_results=3, max_adapter_calls=4, max_latency_ms=10000),
        max_parallel_adapters=3,
        cache_policy="default",
        source_ordering="default",
        ranking_profile_hint="literature",
    ),
    PROFILE_LOCAL_FIRST: RetrievalProfileSpec(
        id=PROFILE_LOCAL_FIRST,
        label="Local-first",
        short_description="Query local sources before external APIs.",
        budget=RetrievalBudget(max_results=3, max_adapter_calls=4, max_latency_ms=8000),
        max_parallel_adapters=3,
        cache_policy="aggressive",
        source_ordering="local_before_remote",
    ),
}


def normalize_profile_id(profile_id: str | None) -> str:
    pid = (profile_id or DEFAULT_RETRIEVAL_PROFILE).strip().lower()
    return pid if pid in VALID_PROFILE_IDS else DEFAULT_RETRIEVAL_PROFILE


def get_profile_spec(profile_id: str | None) -> RetrievalProfileSpec:
    return _PROFILES[normalize_profile_id(profile_id)]


def list_profile_specs() -> list[RetrievalProfileSpec]:
    return [_PROFILES[pid] for pid in (
        PROFILE_FAST,
        PROFILE_BALANCED,
        PROFILE_THOROUGH,
        PROFILE_EVIDENCE_FIRST,
        PROFILE_LOCAL_FIRST,
    )]


def order_adapter_ids(
    adapter_ids: tuple[str, ...],
    *,
    profile: RetrievalProfileSpec,
) -> tuple[str, ...]:
    """Reorder adapters for local-first profile."""
    if profile.source_ordering != "local_before_remote":
        return adapter_ids

    from core.knowledge.configured_sources import load_configured_source

    local: list[str] = []
    remote: list[str] = []
    for aid in adapter_ids:
        source = load_configured_source(aid)
        if source is not None and source.connector_type in _LOCAL_CONNECTORS:
            local.append(aid)
        else:
            remote.append(aid)
    return tuple(local + remote)


def scientific_cache_enabled(profile: RetrievalProfileSpec) -> bool:
    return profile.cache_policy != "bypass"


def scientific_tier_breadth(profile: RetrievalProfileSpec) -> str:
    if profile.id == PROFILE_FAST:
        return "narrow"
    if profile.id == PROFILE_THOROUGH:
        return "wide"
    return "default"
