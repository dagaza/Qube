"""Base implementation for built-in substring-scored skills."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from core.skills.types import SkillContext

ContextBoostFn = Callable[[SkillContext], tuple[float, str | None]]


def substring_trigger_score(query: str, triggers: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    """Normalized hit rate in [0, 1]; uses max(per-list, hits/3) so 2+ matches can activate."""
    if not triggers:
        return 0.0, ()
    q = (query or "").lower()
    hits = [t for t in triggers if t in q]
    if not hits:
        return 0.0, ()
    per_list = len(hits) / len(triggers)
    per_floor = len(hits) / min(3, len(triggers))
    score = min(1.0, max(per_list, per_floor))
    signals = tuple(f"substring:{t}" for t in hits[:5])
    return score, signals


def regex_pattern_score(
    query: str, patterns: tuple[re.Pattern[str], ...]
) -> tuple[float, tuple[str, ...]]:
    if not patterns:
        return 0.0, ()
    q = query or ""
    hits = [p.pattern for p in patterns if p.search(q)]
    if not hits:
        return 0.0, ()
    score = min(1.0, len(hits) / len(patterns))
    return score, tuple(f"regex:{h[:40]}" for h in hits[:3])


@dataclass
class BuiltinSkill:
    """Deterministic skill with substring/regex activation and static prompt fragment."""

    id: str
    name: str
    description: str
    version: str
    priority: int
    max_prompt_chars: int
    prompt_fragment: str
    activation_triggers: tuple[str, ...] = ()
    activation_patterns: tuple[re.Pattern[str], ...] = ()
    mutual_exclusion_group: str | None = None
    retrieval_hint: str | None = None
    context_boost_fns: tuple[ContextBoostFn, ...] = field(default_factory=tuple)

    def score_signals(self, ctx: SkillContext) -> tuple[float, tuple[str, ...]]:
        sub_score, sub_signals = substring_trigger_score(
            ctx.clean_query, self.activation_triggers
        )
        regex_score, regex_signals = regex_pattern_score(
            ctx.clean_query, self.activation_patterns
        )
        raw = max(sub_score, regex_score)
        signals: list[str] = []
        if sub_signals:
            signals.extend(sub_signals)
        if regex_signals:
            signals.extend(regex_signals)

        boost_total = 0.0
        for fn in self.context_boost_fns:
            boost, label = fn(ctx)
            if boost > 0 and label:
                boost_total += boost
                signals.append(label)
        boost_total = min(boost_total, 0.15)
        final = min(1.0, raw + boost_total)
        return final, tuple(signals)

    def score(self, ctx: SkillContext) -> float:
        return self.score_signals(ctx)[0]

    def build_prompt_fragment(self, ctx: SkillContext, score: float) -> str:
        fragment = (self.prompt_fragment or "").strip()
        if not fragment:
            return ""
        hint = self.retrieval_framing_hint(ctx, score)
        if hint:
            fragment = f"{fragment} {hint}"
        if len(fragment) > self.max_prompt_chars:
            fragment = fragment[: self.max_prompt_chars].rstrip() + "…"
        return fragment

    def retrieval_framing_hint(self, ctx: SkillContext, score: float) -> str | None:
        if not ctx.has_retrieval_sources or not self.retrieval_hint:
            return None
        return self.retrieval_hint
