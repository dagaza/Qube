"""Deep research depth profiles — local orchestration limits for @research."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.types import RetrievalBudget

PROFILE_STANDARD = "standard"
PROFILE_THOROUGH = "thorough"

DEFAULT_DEEP_RESEARCH_PROFILE = PROFILE_STANDARD

VALID_DEEP_RESEARCH_PROFILE_IDS = frozenset({PROFILE_STANDARD, PROFILE_THOROUGH})


@dataclass(frozen=True)
class DeepResearchProfileSpec:
    id: str
    label: str
    short_description: str
    max_sub_queries: int
    budget: RetrievalBudget
    merged_source_cap: int
    synthesis_max_tokens: int
    context_char_budget: int


_PROFILES: dict[str, DeepResearchProfileSpec] = {
    PROFILE_STANDARD: DeepResearchProfileSpec(
        id=PROFILE_STANDARD,
        label="Standard",
        short_description=(
            "Default @research depth — up to three sub-queries with balanced "
            "local retrieval limits."
        ),
        max_sub_queries=3,
        budget=RetrievalBudget(max_results=5, max_adapter_calls=3, max_latency_ms=15000),
        merged_source_cap=10,
        synthesis_max_tokens=1400,
        context_char_budget=12000,
    ),
    PROFILE_THOROUGH: DeepResearchProfileSpec(
        id=PROFILE_THOROUGH,
        label="Thorough",
        short_description=(
            "Pro depth — more sub-queries, higher local adapter budgets, and "
            "longer synthesis. Does not change upstream API quotas."
        ),
        max_sub_queries=6,
        budget=RetrievalBudget(max_results=8, max_adapter_calls=5, max_latency_ms=45000),
        merged_source_cap=24,
        synthesis_max_tokens=2400,
        context_char_budget=20000,
    ),
}


def normalize_profile_id(profile_id: str | None) -> str:
    pid = (profile_id or DEFAULT_DEEP_RESEARCH_PROFILE).strip().lower()
    return pid if pid in VALID_DEEP_RESEARCH_PROFILE_IDS else DEFAULT_DEEP_RESEARCH_PROFILE


def get_profile_spec(profile_id: str | None) -> DeepResearchProfileSpec:
    return _PROFILES[normalize_profile_id(profile_id)]


def list_profile_specs() -> list[DeepResearchProfileSpec]:
    return [_PROFILES[PROFILE_STANDARD], _PROFILES[PROFILE_THOROUGH]]


# Backward-compatible alias for callers/tests that imported the old constant.
DEEP_BUDGET = _PROFILES[PROFILE_STANDARD].budget
