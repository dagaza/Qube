"""Data types for the skills layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SkillContext:
    """Read-only turn snapshot for activation + prompt build."""

    user_query: str
    clean_query: str
    execution_route: str
    has_retrieval_sources: bool
    source_count: int
    follow_up_active: bool
    explicit_remember_active: bool
    file_search_active: bool
    narrative_active: bool
    web_capability_blocked: bool = False
    explicit_web_empty_results: bool = False
    router_top_intent: str | None = None
    router_trace_summary: str | None = None
    query_embedding: Any | None = None


@dataclass(frozen=True)
class SkillActivation:
    skill_id: str
    score: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class SkillActivationResult:
    activations: tuple[SkillActivation, ...]
    prompt_block: str
    token_budget_applied: int
    skipped_reason: str | None = None
    forced_skill_ids: tuple[str, ...] = ()
    auto_skill_ids: tuple[str, ...] = ()

    def telemetry_dict(self) -> dict[str, Any]:
        forced = set(self.forced_skill_ids)
        return {
            "skills_active": [
                {
                    "id": a.skill_id,
                    "score": round(a.score, 4),
                    "signals": list(a.signals),
                    "forced": a.skill_id in forced,
                }
                for a in self.activations
            ],
            "skills_forced": list(self.forced_skill_ids),
            "skills_auto": list(self.auto_skill_ids),
            "skills_prompt_chars": len(self.prompt_block or ""),
            "skills_skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class SkillSettings:
    enabled: bool = False
    min_activation_score: float = 0.55
    max_active_skills: int = 3
    total_prompt_char_budget: int = 1200
    embedding_boost_enabled: bool = True
    debug_log_enabled: bool = False


@runtime_checkable
class Skill(Protocol):
    id: str
    name: str
    description: str
    version: str
    priority: int
    max_prompt_chars: int
    mutual_exclusion_group: str | None

    def score(self, ctx: SkillContext) -> float: ...

    def score_signals(self, ctx: SkillContext) -> tuple[float, tuple[str, ...]]: ...

    def build_prompt_fragment(self, ctx: SkillContext, score: float) -> str: ...

    def retrieval_framing_hint(self, ctx: SkillContext, score: float) -> str | None: ...
