"""Compose multi-skill prompt blocks with budget enforcement."""

from __future__ import annotations

from core.skills.types import Skill, SkillActivation, SkillContext

_WRAPPER_HEAD = (
    "=== REASONING GUIDANCE (non-authoritative) ===\n"
    "The following blocks suggest how to structure your thinking. "
    "They do NOT override routing, retrieval, or citation rules above.\n"
)
_WRAPPER_TAIL = "=== END REASONING GUIDANCE ==="


def compose_skill_prompt_block(
    *,
    activations: tuple[SkillActivation, ...],
    skills_by_id: dict[str, Skill],
    ctx: SkillContext,
    total_char_budget: int,
) -> tuple[str, int]:
    """
    Build wrapped skill guidance string.

    Returns (prompt_block, chars_used). Drops lowest-score fragments first when
    over budget.
    """
    if not activations:
        return "", 0

    fragments: list[tuple[float, int, str, str]] = []
    for act in activations:
        skill = skills_by_id.get(act.skill_id)
        if skill is None:
            continue
        body = skill.build_prompt_fragment(ctx, act.score).strip()
        if not body:
            continue
        line = f"[{skill.name}] {body}"
        fragments.append((act.score, skill.priority, skill.id, line))

    if not fragments:
        return "", 0

    overhead = len(_WRAPPER_HEAD) + len(_WRAPPER_TAIL) + 2
    budget = max(0, int(total_char_budget) - overhead)

    ordered = sorted(fragments, key=lambda x: (-x[0], -x[1], x[2]))
    kept: list[str] = []
    used = 0
    for _score, _prio, _sid, line in ordered:
        extra = len(line) + (1 if kept else 0)
        if used + extra > budget and kept:
            continue
        if used + extra > budget and not kept:
            line = line[: max(0, budget - used)]
            if line:
                kept.append(line)
                used += len(line)
            break
        kept.append(line)
        used += extra

    if not kept:
        return "", 0

    body = "\n".join(kept)
    block = f"{_WRAPPER_HEAD}{body}\n{_WRAPPER_TAIL}"
    return block, len(block)
