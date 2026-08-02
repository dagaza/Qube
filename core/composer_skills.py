"""Composer @-mention skill tokens (prompt enforcement only; never routes)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.composer_attachments import (
    ComposerAttachment,
    lift_bare_tool_mentions,
    parse_attachments,
    strip_tokens_for_display,
)
from core.skills.registry import get_skill

_SKILL_TOKEN_RE = re.compile(r"@\[skill:([^\]]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class ComposerSkillMention:
    """Palette selection for a built-in reasoning skill."""

    id: str
    label: str


def format_skill_token(skill_id: str) -> str:
    return f"@[skill:{skill_id.strip()}]"


def strip_skill_tokens(text: str) -> tuple[tuple[str, ...], str]:
    """Remove skill tokens; return deduped skill ids (in order) and remaining text."""
    seen: set[str] = set()
    ordered: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        sid = match.group(1).strip().lower()
        if sid and sid not in seen:
            seen.add(sid)
            ordered.append(sid)
        return ""

    remainder = _SKILL_TOKEN_RE.sub(_repl, text or "")
    remainder = re.sub(r"\s+", " ", remainder).strip()
    return tuple(ordered), remainder


def parse_composer_input(
    text: str,
    *,
    lift_bare_mentions: bool | None = None,
) -> tuple[str, list[ComposerAttachment], tuple[str, ...]]:
    """
    Parse composer text into clean prompt, routing attachments, and enforced skills.

    Skill tokens are stripped before attachment parsing and never affect routing.
    When bare-mention routing is enabled, a leading @tool shorthand (e.g. @research)
    is lifted into a routing attachment before formal `@[tool:…]` tokens are parsed.
    """
    if lift_bare_mentions is None:
        from core.app_settings import get_composer_bare_mention_routing_enabled

        lift_bare_mentions = get_composer_bare_mention_routing_enabled()

    enforced, without_skills = strip_skill_tokens(text)
    bare_attachments: list[ComposerAttachment] = []
    working = without_skills
    if lift_bare_mentions and "@[" not in working:
        working, bare_attachments = lift_bare_tool_mentions(working)

    clean, token_attachments = parse_attachments(working)
    attachments = bare_attachments + token_attachments
    return clean, attachments, enforced


def substantive_composer_prompt(text: str) -> str | None:
    """Return user-visible prompt text after stripping composer tokens, or None if empty."""
    clean, _attachments, _skills = parse_composer_input(text or "")
    clean = clean.strip()
    return clean or None


def strip_all_composer_tokens_for_display(text: str) -> str:
    """Remove attachment and skill tokens for compact display."""
    _, without_skills = strip_skill_tokens(text or "")
    return strip_tokens_for_display(without_skills)


def skill_mention_from_id(skill_id: str) -> ComposerSkillMention | None:
    skill = get_skill(skill_id)
    if skill is None:
        return None
    return ComposerSkillMention(id=skill.id, label=skill.name)


def list_skill_mentions_for_palette(
    *,
    query: str = "",
    limit: int = 80,
) -> list[ComposerSkillMention]:
    from core.skills.registry import iter_skills

    q = (query or "").strip().lower()
    rows: list[ComposerSkillMention] = []
    skills = sorted(
        iter_skills(),
        key=lambda s: (-int(s.priority), s.name.lower(), s.id),
    )
    for skill in skills:
        if q and q not in skill.id and q not in skill.name.lower():
            desc = (skill.description or "").lower()
            if q not in desc:
                continue
        rows.append(ComposerSkillMention(id=skill.id, label=skill.name))
        if len(rows) >= limit:
            break
    return rows
