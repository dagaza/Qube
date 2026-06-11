"""
Inference-time query rewriting for ambiguous deictic follow-ups.

Original user text stays in UI/DB; only the resolved query feeds routing/retrieval/prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from core.discourse_patterns import has_possessive_anaphor, is_deictic_prompt
from core.discourse_referent_policy import (
    rewrite_referent_target,
    validate_resolved_query,
)

if TYPE_CHECKING:
    from core.discourse_intent import FollowUpClassification
    from core.discourse_state import DiscourseState

REWRITE_CONFIDENCE_MIN = 0.70

RewriteReason = Literal[
    "possessive_substitution",
    "pronoun_substitution",
    "frame_template",
    "none",
]

_POSSESSIVE_HEAD = re.compile(
    r"\b(its|his|her|their|our|your)\s+(\w+(?:\s+\w+){0,3})",
    re.I,
)
_PRONOUN_SUBJ = re.compile(r"\b(he|she|they)\b", re.I)
_PRONOUN_OBJ = re.compile(r"\b(him|her|them)\b", re.I)
_LEADING_AND = re.compile(r"^\s*and\s+", re.I)
_POPULATION_FRAME = re.compile(
    r"^\s*(?:and\s+)?(?:what\s+is\s+)?(?:the\s+)?(?:size\s+of\s+)?(?:its|their|his|her)\s+population\b",
    re.I,
)
_POPULATION_SIZE = re.compile(
    r"^\s*(?:and\s+)?what\s+is\s+the\s+size\s+of\s+its\s+population\b",
    re.I,
)
_AREA_FRAME = re.compile(
    r"^\s*(?:and\s+)?what\s+is\s+(?:its|their|his|her)\s+area\b",
    re.I,
)
_EXPLICIT_ENTITY = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\"[^\"]{2,}\"|\'[^\']{2,}\'",
)


@dataclass(frozen=True)
class ResolvedUserQuery:
    original: str
    resolved: str
    substitutions: tuple[tuple[str, str], ...]
    confidence: float
    rewrite_reason: RewriteReason

    @property
    def succeeded(self) -> bool:
        return (
            self.rewrite_reason != "none"
            and self.resolved.strip() != self.original.strip()
            and self.confidence >= REWRITE_CONFIDENCE_MIN
        )


def _possessive_form(referent: str) -> str:
    r = referent.strip()
    if not r:
        return r
    if r.endswith(("s", "S", "z", "Z", "x", "X")):
        return f"{r}'"
    return f"{r}'s"


def _user_already_names_entity(prompt: str, referent: str) -> bool:
    p = (prompt or "").strip()
    r = (referent or "").strip()
    if not p or not r:
        return False
    if r.lower() in p.lower():
        return True
    return bool(_EXPLICIT_ENTITY.search(p)) and not (
        is_deictic_prompt(p) or has_possessive_anaphor(p)
    )


def resolve_ambiguous_user_query(
    prompt: str,
    discourse: Optional["DiscourseState"],
    follow_up: "FollowUpClassification",
) -> ResolvedUserQuery:
    original = (prompt or "").strip()
    if not original or discourse is None:
        return ResolvedUserQuery(original, original, (), 0.0, "none")

    referent = (rewrite_referent_target(discourse) or "").strip()
    if not referent:
        return ResolvedUserQuery(original, original, (), 0.0, "none")

    if not follow_up.active and not (
        is_deictic_prompt(original) or has_possessive_anaphor(original)
    ):
        return ResolvedUserQuery(original, original, (), 0.0, "none")

    if _user_already_names_entity(original, referent):
        return ResolvedUserQuery(original, original, (), 0.0, "none")

    working = _LEADING_AND.sub("", original).strip()
    subs: list[tuple[str, str]] = []
    reason: RewriteReason = "none"
    confidence = 0.0

    poss = _possessive_form(referent)

    if _POPULATION_SIZE.search(working) or _POPULATION_FRAME.search(working):
        resolved = f"What is the population of {referent}?"
        subs.append(("its", referent))
        result = ResolvedUserQuery(original, resolved, tuple(subs), 0.88, "frame_template")
        return _finalize_resolved_query(result, discourse)

    if _AREA_FRAME.search(working):
        resolved = f"What is the area of {referent}?"
        subs.append(("its", referent))
        result = ResolvedUserQuery(original, resolved, tuple(subs), 0.85, "frame_template")
        return _finalize_resolved_query(result, discourse)

    if has_possessive_anaphor(working):
        def _repl(m: re.Match[str]) -> str:
            pron = m.group(1)
            noun = m.group(2)
            subs.append((f"{pron} {noun}", f"{poss} {noun}"))
            return f"{poss} {noun}"

        new_working = _POSSESSIVE_HEAD.sub(_repl, working, count=1)
        if new_working != working:
            working = new_working
            reason = "possessive_substitution"
            confidence = 0.82

    rtype = getattr(discourse, "referent_type", "unknown")
    if _PRONOUN_SUBJ.search(working) and rtype in ("person", "entity"):
        def _subj(m: re.Match[str]) -> str:
            subs.append((m.group(0), referent))
            return referent

        new_working = _PRONOUN_SUBJ.sub(_subj, working, count=1)
        if new_working != working:
            working = new_working
            reason = "pronoun_substitution"
            confidence = max(confidence, 0.80)

    if _PRONOUN_OBJ.search(working) and rtype in ("person", "entity"):
        def _obj(m: re.Match[str]) -> str:
            subs.append((m.group(0), referent))
            return referent

        new_working = _PRONOUN_OBJ.sub(_obj, working, count=1)
        if new_working != working:
            working = new_working
            reason = "pronoun_substitution"
            confidence = max(confidence, 0.78)

    if reason == "none":
        return ResolvedUserQuery(original, original, (), 0.0, "none")

    return _finalize_resolved_query(
        ResolvedUserQuery(original, working, tuple(subs), confidence, reason),
        discourse,
    )


def _finalize_resolved_query(
    result: ResolvedUserQuery,
    discourse: Optional["DiscourseState"],
) -> ResolvedUserQuery:
    if not result.succeeded:
        return result
    ok, reject = validate_resolved_query(result.resolved, discourse)
    if ok:
        return result
    from core.discourse_telemetry import log_discourse_rewrite_validation_failed

    log_discourse_rewrite_validation_failed(
        original=result.original,
        resolved=result.resolved,
        reject_reason=reject,
        referent=(rewrite_referent_target(discourse) or ""),
    )
    return ResolvedUserQuery(result.original, result.original, (), 0.0, "none")
