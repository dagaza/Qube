"""
Shared deictic / possessive patterns for discourse tracking.

Single source of truth for discourse_state and discourse_intent.
"""
from __future__ import annotations

import re

DEICTIC_PRONOUN_RE = re.compile(
    r"\b("
    r"this|that|it|its|they|them|their|those|these|"
    r"he|him|his|she|her|hers|"
    r"we|our|us|you|your|"
    r"the same|the above|the latter"
    r")\b",
    re.I,
)

_POSSESSIVE_ANAPHOR_RE = re.compile(
    r"\b(its|his|her|their|our|your)\s+\w+",
    re.I,
)


def is_deictic_prompt(text: str) -> bool:
    """True when the prompt contains deictic follow-up markers."""
    return bool(DEICTIC_PRONOUN_RE.search((text or "").strip()))


def is_deictic_topic_phrase(topic: str) -> bool:
    """True when an extracted topic phrase still contains unresolved deictic refs."""
    t = (topic or "").strip()
    if not t:
        return False
    return bool(DEICTIC_PRONOUN_RE.search(t)) or bool(_POSSESSIVE_ANAPHOR_RE.search(t))


def has_possessive_anaphor(text: str) -> bool:
    """True when text contains a possessive pronoun anchoring a noun phrase."""
    return bool(_POSSESSIVE_ANAPHOR_RE.search((text or "").strip()))


# Search venues the user names as *where* to look, not *what* to look up.
_REFERENCE_SOURCES = (
    r"(?:wikipedia|wiki|pubmed|arxiv|britannica|encyclopedia|google\s+scholar|"
    r"(?:the\s+)?(?:web|internet))"
)
# Company names double as real topics ("CEO at Google"), so only search-style prepositions.
_SEARCH_ENGINES = r"(?:google|bing|duckduckgo)"
_POLITE_TAIL = r"(?:\s+(?:please|for\s+me|too|as\s+well|now|instead|again))*"

_SOURCE_VENUE_RE = re.compile(
    r"\s*\b(?:"
    r"(?:on|in|from|via|using|through|against|with|by\s+(?:checking|searching|using))\s+"
    r"(?:the\s+)?" + _REFERENCE_SOURCES
    + r"|(?:on|via|using|through|by\s+(?:checking|searching|using))\s+" + _SEARCH_ENGINES
    + r")\b",
    re.I,
)
_TRAILING_ONLINE_RE = re.compile(r"\s+online(?=" + _POLITE_TAIL + r"\s*[?.!]*\s*$)", re.I)
_SOURCE_NAME_RE = re.compile(
    r"^(?:the\s+)?(?:" + _REFERENCE_SOURCES + r"|" + _SEARCH_ENGINES + r"|online)"
    + _POLITE_TAIL + r"[?.!]*$",
    re.I,
)
_COMPOSER_TOKEN_RE = re.compile(r"@\[[a-z_]+:[^\]]*\]|(?:(?<=\s)|^)@[a-z][\w-]*", re.I)


def is_source_name(topic: str) -> bool:
    """True when an extracted topic is just a search venue (``Wikipedia``, ``the web``)."""
    return bool(_SOURCE_NAME_RE.match((topic or "").strip()))


def mentions_source_venue(text: str) -> bool:
    """True when the text names a venue to search ("on Wikipedia", "online")."""
    t = (text or "").strip()
    return bool(_SOURCE_VENUE_RE.search(t) or _TRAILING_ONLINE_RE.search(t))


def strip_source_venue_phrases(text: str) -> str:
    """Drop composer tokens and venue phrases, leaving what the user asked about."""
    s = _COMPOSER_TOKEN_RE.sub(" ", text or "")
    s = _SOURCE_VENUE_RE.sub("", s)
    s = _TRAILING_ONLINE_RE.sub("", s)
    s = re.sub(r"\s+([?.!,])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()
