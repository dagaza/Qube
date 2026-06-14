"""Skill activation engine — scoring, ranking, and prompt composition."""

from __future__ import annotations

import logging
from typing import Iterable

from core.skills.centroids import embedding_boost_score
from core.skills.prompt_compose import compose_skill_prompt_block
from core.skills.registry import get_skill, iter_skills
from core.skills.types import (
    Skill,
    SkillActivation,
    SkillActivationResult,
    SkillContext,
    SkillSettings,
)

logger = logging.getLogger("Qube.Skills")

_JACCARD_DUP_THRESHOLD = 0.7
_FORCED_SIGNAL = "forced:composer"


def _should_skip(ctx: SkillContext, settings: SkillSettings) -> str | None:
    if ctx.explicit_remember_active:
        return "explicit_remember"
    if ctx.web_capability_blocked:
        return "web_capability_blocked"
    if ctx.rag_capability_blocked:
        return "rag_capability_blocked"
    if ctx.explicit_web_empty_results:
        return "explicit_web_empty_results"
    return None


def _trigger_word_set(skill: Skill) -> set[str]:
    triggers = getattr(skill, "activation_triggers", ()) or ()
    return {t.lower() for t in triggers if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _is_forced(act: SkillActivation) -> bool:
    return _FORCED_SIGNAL in act.signals


def _apply_mutual_exclusion(
    ranked: list[tuple[Skill, SkillActivation]],
) -> list[tuple[Skill, SkillActivation]]:
    seen_groups: dict[str, tuple[Skill, SkillActivation]] = {}
    out: list[tuple[Skill, SkillActivation]] = []
    for skill, act in ranked:
        group = skill.mutual_exclusion_group
        if not group:
            out.append((skill, act))
            continue
        prev = seen_groups.get(group)
        if prev is None:
            seen_groups[group] = (skill, act)
            out.append((skill, act))
            continue
        prev_forced = _is_forced(prev[1])
        act_forced = _is_forced(act)
        if act_forced and not prev_forced:
            out = [(s, a) for s, a in out if s.mutual_exclusion_group != group]
            seen_groups[group] = (skill, act)
            out.append((skill, act))
        elif prev_forced and not act_forced:
            continue
        elif act.score > prev[1].score or (
            act.score == prev[1].score and skill.priority > prev[0].priority
        ):
            out = [(s, a) for s, a in out if s.mutual_exclusion_group != group]
            seen_groups[group] = (skill, act)
            out.append((skill, act))
    return out


def _apply_near_duplicate_suppression(
    ranked: list[tuple[Skill, SkillActivation]],
) -> list[tuple[Skill, SkillActivation]]:
    kept: list[tuple[Skill, SkillActivation]] = []
    kept_sets: list[tuple[set[str], int]] = []
    for skill, act in ranked:
        if _is_forced(act):
            kept.append((skill, act))
            kept_sets.append((_trigger_word_set(skill), skill.priority))
            continue
        words = _trigger_word_set(skill)
        drop = False
        for other_words, other_prio in kept_sets:
            if _jaccard(words, other_words) > _JACCARD_DUP_THRESHOLD:
                if skill.priority <= other_prio:
                    drop = True
                    break
        if drop:
            continue
        kept.append((skill, act))
        kept_sets.append((words, skill.priority))
    return kept


def _score_skill(
    skill: Skill,
    ctx: SkillContext,
    settings: SkillSettings,
) -> SkillActivation | None:
    base_score, signals = skill.score_signals(ctx)
    emb_score, emb_signal = embedding_boost_score(
        skill.id,
        ctx.query_embedding,
        enabled=settings.embedding_boost_enabled,
    )
    raw = max(base_score, emb_score)
    sig_list = list(signals)
    if emb_signal:
        sig_list.append(emb_signal)
    final = min(1.0, raw)
    if final < settings.min_activation_score:
        return None
    return SkillActivation(skill_id=skill.id, score=final, signals=tuple(sig_list))


def _resolve_forced_pairs(
    forced_skill_ids: tuple[str, ...],
    skill_by_id: dict[str, Skill],
) -> list[tuple[Skill, SkillActivation]]:
    pairs: list[tuple[Skill, SkillActivation]] = []
    for sid in forced_skill_ids:
        skill = skill_by_id.get(sid) or get_skill(sid)
        if skill is None:
            logger.warning("[Skills] Unknown enforced skill id=%s (ignored)", sid)
            continue
        pairs.append(
            (
                skill,
                SkillActivation(
                    skill_id=skill.id,
                    score=1.0,
                    signals=(_FORCED_SIGNAL,),
                ),
            )
        )
    return pairs


def activate_skills(
    ctx: SkillContext,
    *,
    settings: SkillSettings | None = None,
    skills: Iterable[Skill] | None = None,
    forced_skill_ids: tuple[str, ...] = (),
) -> SkillActivationResult:
    """Score, rank, and compose skill prompt injection for one turn."""
    if settings is None:
        from core.app_settings import get_skill_settings

        settings = get_skill_settings()

    forced_unique = tuple(dict.fromkeys(forced_skill_ids))
    skip = _should_skip(ctx, settings)
    if skip:
        return SkillActivationResult(
            activations=(),
            prompt_block="",
            token_budget_applied=0,
            skipped_reason=skip,
            forced_skill_ids=forced_unique,
        )

    if not settings.enabled and not forced_unique:
        return SkillActivationResult(
            activations=(),
            prompt_block="",
            token_budget_applied=0,
            skipped_reason="disabled",
            forced_skill_ids=forced_unique,
        )

    skill_list = list(skills) if skills is not None else list(iter_skills())
    skill_by_id = {s.id: s for s in skill_list}
    forced_pairs = _resolve_forced_pairs(forced_unique, skill_by_id)
    forced_ids = {s.id for s, _a in forced_pairs}

    auto_candidates: list[tuple[Skill, SkillActivation]] = []
    for skill in skill_list:
        if skill.id in forced_ids:
            continue
        act = _score_skill(skill, ctx, settings)
        if act is not None:
            auto_candidates.append((skill, act))

    auto_candidates.sort(
        key=lambda pair: (-pair[1].score, -pair[0].priority, pair[0].id)
    )
    slots = max(0, settings.max_active_skills - len(forced_pairs))
    merged = list(forced_pairs) + auto_candidates[:slots]
    merged = _apply_mutual_exclusion(merged)
    merged = _apply_near_duplicate_suppression(merged)
    merged = merged[: max(1, settings.max_active_skills)] if merged else []

    activations = tuple(act for _skill, act in merged)
    skills_by_id = {s.id: s for s, _a in merged}
    auto_ids = tuple(
        a.skill_id for a in activations if a.skill_id not in forced_ids
    )

    prompt_block, chars_used = compose_skill_prompt_block(
        activations=activations,
        skills_by_id=skills_by_id,
        ctx=ctx,
        total_char_budget=settings.total_prompt_char_budget,
    )

    if activations:
        logger.info(
            "[Skills] active=%s forced=%s chars=%s",
            [a.skill_id for a in activations],
            list(forced_ids),
            chars_used,
        )

    return SkillActivationResult(
        activations=activations,
        prompt_block=prompt_block,
        token_budget_applied=chars_used,
        skipped_reason=None,
        forced_skill_ids=forced_unique,
        auto_skill_ids=auto_ids,
    )
