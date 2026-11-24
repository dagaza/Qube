"""Correctly spelt query variant for scholarly search (plan S8, decision D6).

A misspelt content word ("Artificial Inteligence") can remove every relevant
result from a keyword search engine (S5 live finding). The user's wording is
never replaced: ``spelling_variant`` returns a *second* wording that the
scientific pipeline searches alongside it, and the relevance gate still scores
every row against the user's own question.

A word is corrected only when the evidence is strong, because the English word
list (``pyspellchecker``) does not know most technical vocabulary ("statins",
"covid", "mRNA", "apoptosis" are all unknown to it):

* the word is alphabetic, at least 5 letters, and unknown to the word list;
* it is not an acronym or internally capitalised ("mRNA", "OAuth", "CRISPR");
* it is not a drug / condition / trial entity, nor shaped like a drug name
  (WHO INN stems: -statin, -mab, -pril, -sartan, …);
* it does not appear in the assistant's earlier messages (names and terms the
  conversation introduced; the user's own repeated misspelling is not
  protected);
* among the known words one edit away that keep its first letter, the most
  frequent is common (≥ ``MIN_CANDIDATE_COUNT``) and at least
  ``MIN_DOMINANCE`` × as frequent as the runner-up. "statins" stays: its
  neighbours "stations", "stains" and "stating" are equally plausible.

If ``pyspellchecker`` is not installed the feature is off (``None``).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MIN_WORD_LENGTH = 5
MIN_CANDIDATE_COUNT = 200
MIN_DOMINANCE = 3.0

_WORD = re.compile(r"(?<![\w'’-])[A-Za-z]+(?![\w'’-])")
# WHO INN stems and common pharmacological / chemical suffixes.
_DRUG_SHAPE = re.compile(
    r"(?:statins?|mabs?|nibs?|prils?|sartans?|olols?|azoles?|mycins?|cillins?|virs?|"
    r"tides?|gliflozins?|gliptins?|oxacins?|dipines?|prazoles?|azepams?|tidines?|"
    r"parins?|platins?|taxels?|rubicins?|cyclines?|profens?|setrons?|triptans?|"
    r"lukasts?|zumabs?|ximabs?)$",
    re.IGNORECASE,
)

_checker: Any = None
_checker_failed = False
_lock = threading.Lock()


@dataclass(frozen=True)
class SpellingCorrection:
    original: str
    corrected: str
    candidate_count: int
    runner_up_count: int


@dataclass(frozen=True)
class SpellingVariant:
    text: str
    corrections: tuple[SpellingCorrection, ...]

    def diag(self) -> list[dict[str, Any]]:
        return [
            {
                "original": c.original,
                "corrected": c.corrected,
                "count": c.candidate_count,
                "runner_up": c.runner_up_count,
            }
            for c in self.corrections
        ]


def _get_checker() -> Any:
    global _checker, _checker_failed
    if _checker is not None or _checker_failed:
        return _checker
    with _lock:
        if _checker is None and not _checker_failed:
            try:
                from spellchecker import SpellChecker

                _checker = SpellChecker(language="en", distance=1)
            except Exception as exc:  # missing package or data: feature off
                _checker_failed = True
                logger.info("[Spelling] spell checker unavailable: %s", exc)
    return _checker


def spelling_available() -> bool:
    return _get_checker() is not None


def conversation_terms(history: Iterable[dict[str, Any]] | None, *, max_messages: int = 12) -> frozenset[str]:
    """Lower-cased words of the assistant's recent messages (protected from correction)."""
    words: set[str] = set()
    msgs = [m for m in (history or []) if isinstance(m, dict)]
    for msg in msgs[-max_messages:]:
        if str(msg.get("role") or "").lower() != "assistant":
            continue
        words.update(w.lower() for w in _WORD.findall(str(msg.get("content") or "")))
    return frozenset(words)


def _entity_words(text: str) -> set[str]:
    from core.knowledge.entities.conditions import extract_condition_entities
    from core.knowledge.entities.drug_classes import extract_drug_entities
    from core.knowledge.entities.trials import extract_trial_entities

    words: set[str] = set()
    for extract in (extract_drug_entities, extract_condition_entities, extract_trial_entities):
        try:
            for _eid, label in extract(text):
                words.update(w.lower() for w in _WORD.findall(str(label)))
        except Exception:
            continue
    return words


def _protected(token: str) -> bool:
    if len(token) < MIN_WORD_LENGTH:
        return True
    if token.isupper() or any(c.isupper() for c in token[1:]):
        return True  # acronym or internal capital: mRNA, OAuth, CRISPR
    return bool(_DRUG_SHAPE.search(token))


def correct_word(word: str, checker: Any | None = None) -> SpellingCorrection | None:
    """A confident correction of one unknown word, or ``None``."""
    sp = checker or _get_checker()
    if sp is None:
        return None
    lower = word.lower()
    freq = sp.word_frequency
    if freq[lower] > 0:
        return None
    counts = sorted(
        ((int(freq[c]), c) for c in sp.known(sp.edit_distance_1(lower)) if c != lower and c[:1] == lower[:1]),
        reverse=True,
    )
    if not counts:
        return None
    top, best = counts[0]
    runner_up = counts[1][0] if len(counts) > 1 else 0
    if top < MIN_CANDIDATE_COUNT or top < MIN_DOMINANCE * runner_up:
        return None
    corrected = best.capitalize() if word[:1].isupper() else best
    return SpellingCorrection(original=word, corrected=corrected, candidate_count=top, runner_up_count=runner_up)


def spelling_variant(text: str, *, protected_terms: Iterable[str] = ()) -> SpellingVariant | None:
    """``text`` with confidently misspelt words corrected, or ``None`` (see module doc)."""
    raw = str(text or "")
    if not raw.strip():
        return None
    sp = _get_checker()
    if sp is None:
        return None
    protected = {t.lower() for t in protected_terms} | _entity_words(raw)
    corrections: list[SpellingCorrection] = []
    pieces: list[str] = []
    last = 0
    for match in _WORD.finditer(raw):
        token = match.group(0)
        if _protected(token) or token.lower() in protected:
            continue
        fix = correct_word(token, sp)
        if fix is None:
            continue
        pieces.append(raw[last : match.start()])
        pieces.append(fix.corrected)
        last = match.end()
        corrections.append(fix)
    if not corrections:
        return None
    pieces.append(raw[last:])
    return SpellingVariant(text="".join(pieces), corrections=tuple(corrections))


__all__ = [
    "MIN_CANDIDATE_COUNT",
    "MIN_DOMINANCE",
    "SpellingCorrection",
    "SpellingVariant",
    "conversation_terms",
    "correct_word",
    "spelling_available",
    "spelling_variant",
]
