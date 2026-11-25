"""Acronyms defined earlier in a conversation, for retrieval queries (plan S5).

A follow-up such as "how does ANT analyze AI?" retrieves nothing useful when
"ANT" was only defined in an earlier turn ("Actor-network theory (ANT) is …").
This module finds such definitions in recent history and expands the bare
acronym in the retrieval query to ``Long Form (LF)``: the long form for
semantic/keyword search, the acronym kept for lexical matching.

Deterministic and conservative by design:

* only explicit definitions count — ``Long Form (LF)`` or ``LF (Long Form)`` —
  and the acronym's letters must match the long form's word initials
  (hyphenated parts count as words; small function words may be skipped);
* an acronym defined two different ways in the window is dropped;
* expansion is case-sensitive and whole-word, so "ant" or "us" are never touched
  and an undefined "WHO" stays as written.

Not user-visible: only retrieval strings use it.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

DEFAULT_MAX_TURNS = 6

# Function words a long form may contain without contributing a letter:
# "Food and Drug Administration (FDA)", "Department of Defense (DoD)".
_SKIPPABLE = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "&"}
)

# 2–10 chars, starts with a letter, at least two capitals ("ANT", "DoD", "mRNA").
_ACRONYM = r"[A-Za-z][A-Za-z0-9&]{1,9}"
_LONG_THEN_ACRONYM = re.compile(r"\((" + _ACRONYM + r")\)")
_ACRONYM_THEN_LONG = re.compile(r"(?<![\w-])(" + _ACRONYM + r")\s*\(([^()]{3,100})\)")
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_MAX_LONG_FORM_WORDS = 12


def _is_acronym_token(token: str) -> bool:
    return sum(1 for c in token if c.isupper()) >= 2 and token[0].isalpha()


def _letters(acronym: str) -> str:
    """Letters the long form's initials must spell; a plural "s" ("GANs") is not one."""
    if len(acronym) > 2 and acronym.endswith("s") and acronym[:-1].isupper():
        acronym = acronym[:-1]
    return "".join(c for c in acronym.upper() if c.isalpha())


def _words(text: str) -> list[tuple[str, int, int]]:
    """Words (hyphen parts split) with their spans in ``text``."""
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]


def _initials_match(words: list[str], letters: str) -> bool:
    """Every content word contributes its initial, in order, and nothing else."""
    i = 0
    for word in words:
        low = word.lower()
        if i < len(letters) and low[0].upper() == letters[i]:
            i += 1
        elif low in _SKIPPABLE:
            continue
        else:
            return False
    return i == len(letters)


def _long_form_before(text: str, paren_start: int, letters: str) -> str | None:
    """Shortest run of words ending right before ``(`` whose initials spell ``letters``."""
    prefix = text[:paren_start]
    words = _words(prefix)
    if not words:
        return None
    # The definition must sit directly before the parenthesis ("theory (ANT)",
    # not "theory, see (ANT)").
    if prefix[words[-1][2]:].strip():
        return None
    window = words[-min(len(words), len(letters) + 4):]
    for start in range(len(window) - 1, -1, -1):
        candidate = window[start:]
        if candidate[0][0].lower() in _SKIPPABLE:
            continue
        if _initials_match([w for w, _, _ in candidate], letters):
            return prefix[candidate[0][1]:candidate[-1][2]]
    return None


def _normalise_long_form(long_form: str) -> str:
    return " ".join(long_form.split())


def _definitions_in(text: str) -> Iterable[tuple[str, str]]:
    for m in _LONG_THEN_ACRONYM.finditer(text):
        acronym = m.group(1)
        if not _is_acronym_token(acronym):
            continue
        long_form = _long_form_before(text, m.start(), _letters(acronym))
        if long_form:
            yield acronym, _normalise_long_form(long_form)
    for m in _ACRONYM_THEN_LONG.finditer(text):
        acronym, inner = m.group(1), m.group(2).strip()
        if not _is_acronym_token(acronym):
            continue
        words = [w for w, _, _ in _words(inner)]
        if not words or len(words) > _MAX_LONG_FORM_WORDS:
            continue
        if _initials_match(words, _letters(acronym)) and words[0].lower() not in _SKIPPABLE:
            yield acronym, _normalise_long_form(inner)


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    return ""


def extract_acronym_definitions(
    history: list[dict[str, Any]] | None,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, str]:
    """Acronym → long form for definitions in the last ``max_turns`` messages.

    Both user and assistant messages count. An acronym defined with two
    different long forms (case-insensitive) is left out.
    """
    if not history or max_turns <= 0:
        return {}
    found: dict[str, str] = {}
    conflicting: set[str] = set()
    for message in history[-max_turns:]:
        for acronym, long_form in _definitions_in(_message_text(message)):
            if acronym in conflicting:
                continue
            prior = found.get(acronym)
            if prior is None:
                found[acronym] = long_form
            elif prior.lower() != long_form.lower():
                conflicting.add(acronym)
                del found[acronym]
    return found


def expand_acronyms(
    text: str,
    definitions: dict[str, str],
) -> tuple[str, list[tuple[str, str]]]:
    """Expand the first whole-word use of each defined acronym in ``text``.

    ``"how ANT analyzes AI"`` → ``"how actor-network theory (ANT) analyzes AI"``.
    Skipped when the long form already appears in ``text``. Returns the new text
    and the ``(acronym, long_form)`` pairs applied, in text order.
    """
    if not text or not definitions:
        return text, []
    spans: list[tuple[int, int, str, str]] = []
    lowered = text.lower()
    for acronym, long_form in definitions.items():
        if long_form.lower() in lowered:
            continue
        m = re.search(r"(?<![\w-])" + re.escape(acronym) + r"(?![\w-])", text)
        if m is None:
            continue
        # Already written as "(ANT)": the long form is right before it.
        if text[max(0, m.start() - 1):m.start()] == "(":
            continue
        spans.append((m.start(), m.end(), acronym, long_form))
    if not spans:
        return text, []
    spans.sort()
    out: list[str] = []
    applied: list[tuple[str, str]] = []
    pos = 0
    for start, end, acronym, long_form in spans:
        out.append(text[pos:start])
        out.append(f"{long_form} ({acronym})")
        applied.append((acronym, long_form))
        pos = end
    out.append(text[pos:])
    return "".join(out), applied


__all__ = [
    "DEFAULT_MAX_TURNS",
    "expand_acronyms",
    "extract_acronym_definitions",
]
