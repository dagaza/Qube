"""
Map conversational preference phrases to structured profile keys.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from core.user_profile import get_user_profile_store

PREFERENCE_KIND_PRESENTATION = "presentation"
PREFERENCE_KIND_FACTUAL = "factual"

_VALID_KINDS = frozenset({PREFERENCE_KIND_PRESENTATION, PREFERENCE_KIND_FACTUAL})

_PROFILE_KEYS = frozenset({
    "units",
    "temperature",
    "wind_speed",
    "locale",
    "language",
    "verbosity",
    "display_name",
})


@dataclass(frozen=True)
class InferredPreference:
    profile_key: str
    value: str
    preference_kind: str
    confidence: float


_METRIC_PATTERNS = (
    re.compile(r"\bmetric\b", re.I),
    re.compile(r"\bcelsius\b", re.I),
    re.compile(r"\bcentigrade\b", re.I),
    re.compile(r"\bkm/h\b", re.I),
    re.compile(r"\bkilomet(er|re)s?\b", re.I),
)

_IMPERIAL_PATTERNS = (
    re.compile(r"\bimperial\b", re.I),
    re.compile(r"\bfahrenheit\b", re.I),
    re.compile(r"\b(?:°?\s*)f\b", re.I),
    re.compile(r"\bmph\b", re.I),
    re.compile(r"\bmiles?\b", re.I),
)

_CALL_ME = re.compile(
    r"\b(?:call me|my name is|i'?m called)\s+([A-Za-z][A-Za-z\s'\-]{0,40})\b",
    re.I,
)

_CONCISE = re.compile(r"\b(?:be\s+)?(?:concise|brief|short answers?)\b", re.I)
_VERBOSE = re.compile(r"\b(?:be\s+)?(?:verbose|detailed|thorough)\b", re.I)


def classify_preference_from_fact(fact: dict) -> tuple[str, Optional[str]]:
    """
    Return (preference_kind, profile_key) for a validated extraction fact.
    """
    content = str(fact.get("content") or "")
    category = str(fact.get("category") or "").lower()
    subject = str(fact.get("subject") or "").lower()
    inferred = infer_from_text(content)
    if inferred and inferred.preference_kind == PREFERENCE_KIND_PRESENTATION:
        return PREFERENCE_KIND_PRESENTATION, inferred.profile_key
    if category == "preference" and subject == "user":
        if inferred:
            return inferred.preference_kind, inferred.profile_key
        return PREFERENCE_KIND_FACTUAL, None
    if category in ("identity",) and subject == "user":
        if inferred and inferred.profile_key == "display_name":
            return PREFERENCE_KIND_PRESENTATION, "display_name"
    return PREFERENCE_KIND_FACTUAL, None


def infer_from_text(text: str) -> Optional[InferredPreference]:
    t = (text or "").strip()
    if not t:
        return None
    for pat in _METRIC_PATTERNS:
        if pat.search(t):
            return InferredPreference(
                profile_key="units",
                value="metric",
                preference_kind=PREFERENCE_KIND_PRESENTATION,
                confidence=0.88,
            )
    for pat in _IMPERIAL_PATTERNS:
        if pat.search(t):
            return InferredPreference(
                profile_key="units",
                value="imperial",
                preference_kind=PREFERENCE_KIND_PRESENTATION,
                confidence=0.88,
            )
    m = _CALL_ME.search(t)
    if m:
        name = m.group(1).strip().rstrip(".")
        if name:
            return InferredPreference(
                profile_key="display_name",
                value=name,
                preference_kind=PREFERENCE_KIND_PRESENTATION,
                confidence=0.82,
            )
    if _CONCISE.search(t):
        return InferredPreference(
            profile_key="verbosity",
            value="concise",
            preference_kind=PREFERENCE_KIND_PRESENTATION,
            confidence=0.75,
        )
    if _VERBOSE.search(t):
        return InferredPreference(
            profile_key="verbosity",
            value="detailed",
            preference_kind=PREFERENCE_KIND_PRESENTATION,
            confidence=0.75,
        )
    return None


def apply_inferred_to_profile(
    fact: dict,
    *,
    confidence: float | None = None,
) -> Optional[InferredPreference]:
    """
    When a stored fact is presentation-class, upsert user_profile.json.
    Returns the inference applied, or None.
    """
    kind, profile_key = classify_preference_from_fact(fact)
    if kind != PREFERENCE_KIND_PRESENTATION or not profile_key:
        return None
    if profile_key not in _PROFILE_KEYS:
        return None
    inferred = infer_from_text(str(fact.get("content") or ""))
    if not inferred:
        return None
    conf = float(confidence if confidence is not None else inferred.confidence)
    get_user_profile_store().set_inferred(
        inferred.profile_key,
        inferred.value,
        confidence=conf,
        source="conversation",
    )
    return inferred


def normalize_fact_payload(fact: dict) -> dict:
    """Attach preference_kind and profile_key to a fact payload dict."""
    payload = dict(fact)
    kind, key = classify_preference_from_fact(payload)
    payload["preference_kind"] = kind
    if key:
        payload["profile_key"] = key
    return payload


__all__ = [
    "PREFERENCE_KIND_PRESENTATION",
    "PREFERENCE_KIND_FACTUAL",
    "InferredPreference",
    "classify_preference_from_fact",
    "infer_from_text",
    "apply_inferred_to_profile",
    "normalize_fact_payload",
]
