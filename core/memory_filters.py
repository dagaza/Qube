"""
Shared filters for the memory subsystem.

Used by both the extraction path (``workers/enrichment_worker.py``) and the
retrieval path (``workers/llm_worker.py`` / ``mcp/memory_tool.py``) so that
the rules about what counts as a durable user fact are defined in exactly
one place.

Pure functions; no Qt, no LanceDB, no I/O.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ============================================================
# Assistant failure / refusal patterns.
#
# These catch the common shapes of assistant messages that should never be
# mined as "facts about the user": apologies, refusals, capability claims,
# "as an AI" disclaimers, stale offline/limitation messages, etc.
#
# Every pattern is case-insensitive and matches on a substring of the
# message content.
# ============================================================
_ASSISTANT_FAILURE_PATTERN_SOURCES: tuple[str, ...] = (
    r"i\s+don'?t\s+have\s+(?:access\s+to|the\s+ability|internet)",
    r"i\s+can(?:'|no)?t\s+(?:access|browse|reach|look\s+up|search|connect)",
    r"i\s+cannot\s+(?:access|browse|reach|look\s+up|search|connect)",
    r"\bas\s+an\s+ai\b",
    r"i'?m\s+not\s+able\s+to",
    r"i\s+am\s+not\s+able\s+to",
    r"\bi\s+don'?t\s+know\b",
    r"\bi\s+do\s+not\s+know\b",
    r"\bi\s+cannot\b",
    r"\bi\s+can'?t\s+help\b",
    r"\bunable\s+to\b",
    r"\bmy\s+training\s+data\b",
    r"\bno\s+internet\b",
    r"\bcan'?t\s+browse\b",
    r"\boffline\b",
    r"sorry,\s+my\s+brain",
    r"connection\s+(?:error|timeout)",
    r"\bpipeline\s+error\b",
)

ASSISTANT_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in _ASSISTANT_FAILURE_PATTERN_SOURCES
)


def is_assistant_failure_message(text: str) -> bool:
    """True when ``text`` looks like an assistant refusal / apology / limitation claim.

    Used both to scrub messages before they reach the extraction LLM and to
    reject candidate facts whose content is essentially an assistant failure
    sentence.
    """
    if not text:
        return False
    s = text.strip()
    if not s:
        return False
    for p in ASSISTANT_FAILURE_PATTERNS:
        if p.search(s):
            return True
    return False


# ============================================================
# Thin-content guard.
#
# Rejects candidate facts that carry essentially no durable information:
# a single bare proper noun, a two-word stub, or a string that is entirely
# made of stopwords / punctuation.
# ============================================================
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "or", "the", "of", "in", "on", "at", "to", "for",
        "with", "by", "is", "was", "were", "be", "been", "being", "are",
        "this", "that", "these", "those", "it", "its", "as", "if", "but",
        "so", "then", "than", "he", "she", "they", "them", "his", "her",
        "their", "we", "our", "you", "your", "i", "me", "my", "mine",
        "not", "no", "yes", "do", "does", "did", "done", "has", "have",
        "had", "just", "only", "also", "too", "from", "about",
    }
)

_TOKEN_SPLIT_RE = re.compile(r"[^\w']+")


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    s = unicodedata.normalize("NFKC", text).strip()
    if not s:
        return []
    raw = _TOKEN_SPLIT_RE.split(s)
    return [t for t in raw if t]


def is_thin_content(content: str) -> bool:
    """True when ``content`` is too low-information to be a useful memory.

    Heuristics:
    - Fewer than 3 word tokens (a bare name is exactly this shape).
    - A single Title-case proper noun regardless of length.
    - Every token is a stopword after lowercasing.
    - Contains no alphabetic character.
    """
    if content is None:
        return True
    s = content.strip()
    if not s:
        return True
    if not re.search(r"[A-Za-z]", s):
        return True

    toks = _tokens(s)
    if not toks:
        return True

    if len(toks) == 1:
        tok = toks[0]
        if tok[:1].isupper():
            return True
        if tok.lower() in _STOPWORDS:
            return True
        return True

    if len(toks) < 3:
        return True

    lowered = [t.lower() for t in toks]
    if all(t in _STOPWORDS for t in lowered):
        return True

    return False


# ============================================================
# Explicit "remember that ..." detector.
#
# Used to allow third-party / knowledge memories only when the user has
# explicitly asked for them. Returns the fact body the user wants stored,
# or ``None`` if no explicit-remember phrase was detected.
# ============================================================
_EXPLICIT_REMEMBER_RE = re.compile(
    r"""
    (?:^|[.!?;,\s])
    (?:
        please\s+ )?
    (?:
        remember\s+(?:that|this)?\s+
      | note\s+(?:that|this)?\s+
      | don'?t\s+forget\s+(?:that|this)?\s+
      | make\s+(?:a\s+)?note\s*(?:that|:)?\s+
      | keep\s+in\s+mind\s+(?:that)?\s+
      | memorize\s+(?:that|this)?\s+
      | store\s+(?:this|that)\s+(?:in\s+memory)?\s*:?\s*
    )
    (?P<body>.+?)
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_explicit_remember(user_text: str) -> Optional[str]:
    """Return the fact body the user explicitly asked to remember, else ``None``.

    Matches patterns like:
        "remember that my sister is called Alice"
        "please note that the project deadline is March"
        "don't forget my wifi password is ..."
        "keep in mind that I prefer dark mode"
        "make a note: I work part-time on Fridays"
    """
    if not user_text:
        return None
    s = user_text.strip()
    if not s:
        return None

    m = _EXPLICIT_REMEMBER_RE.search(s)
    if not m:
        return None

    body = (m.group("body") or "").strip().strip("\"'`")
    if not body:
        return None

    if is_thin_content(body):
        return None

    return body


# ============================================================
# Recall-intent detector.
#
# Phase A uses substring patterns. Phase B layers a semantic centroid on top
# of this in ``mcp/cognitive_router.py`` / ``workers/intent_router.py`` and
# this function becomes the fallback when no embedding vector is available.
# ============================================================
_RECALL_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\btell\s+me\s+about\b",
        r"\bwho\s+is\b",
        r"\bwho\s+was\b",
        r"\bwhat\s+is\s+[a-z]",
        r"\bremind\s+me\s+(?:about|of)\b",
        r"\bwhat\s+do\s+(?:i|we)\s+know\s+about\b",
        r"\bwhat\s+did\s+(?:we|you)\s+say\s+about\b",
        r"\bsummari[sz]e\s+what\s+you\s+know\s+about\b",
        r"\brefresh\s+my\s+memory\b",
        r"\brecall\s+what\b",
        r"\bdo\s+you\s+remember\b",
        r"\banything\s+about\b",
    )
)


def detect_recall_intent(user_text: str) -> bool:
    """Substring-based detector for "tell me about X" / "who is X" style queries.

    Used by Phase A's fusion-for-recall override in ``llm_worker`` to upgrade
    NONE / MEMORY / RAG routes to HYBRID when the user is asking a recall
    question.
    """
    if not user_text:
        return False
    s = user_text.strip()
    if not s:
        return False
    for p in _RECALL_INTENT_PATTERNS:
        if p.search(s):
            return True
    return False


# ============================================================
# Narrative / recap intent detector (T3.2).
#
# Used by the T3.2 episodic-summaries pass to route narrative questions
# ("what have we been working on?", "recap my session", "where did we
# leave off?") to MEMORY with ``prefer_episode=True`` so the episode
# summary rows outrank the atomic-fact rows in retrieval.
# ============================================================
_NARRATIVE_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat\s+have\s+(?:we|i)\s+been\b",
        r"\bwhat\s+were\s+we\b",
        r"\bwhat\s+(?:did|were)\s+we\s+(?:talk(?:ing)?|work(?:ing)?|discuss(?:ing)?)\b",
        r"\brecap\b",
        r"\bcatch\s+me\s+up\b",
        r"\bwhere\s+did\s+we\s+leave\s+off\b",
        r"\bwhat'?s\s+the\s+status\b",
        r"\bsummari[sz]e\s+(?:our|my|this|the)\s+(?:chat|conversation|session|discussion|project)\b",
        r"\bwhat\s+have\s+we\s+decided\b",
    )
)


def detect_narrative_intent(user_text: str) -> bool:
    """True when the user is asking a narrative / recap / "what were we
    working on?" style question.

    Used by T3.2 to force the turn onto the MEMORY route and boost
    ``qube_memory::episode::*`` rows over the atomic-fact rows.
    """
    if not user_text:
        return False
    s = user_text.strip()
    if not s:
        return False
    for p in _NARRATIVE_INTENT_PATTERNS:
        if p.search(s):
            return True
    return False


# ============================================================
# Explicit file-search intent detector.
#
# Catches "look into my files", "check my documents", "is there a mention
# of X in my notes", etc. These are unambiguous RAG requests — the user
# literally points the assistant at the local document library. When one
# of these patterns fires we want to route straight to RAG and *skip*
# memory retrieval, otherwise unrelated memories (e.g. a "my mom's name
# is Cornelia" memory when the query is about a document's "Dr. Evelyn")
# pollute the LLM's context and cause the model to either cite the wrong
# source or collapse into a bare citation token.
# ============================================================
_FILE_SEARCH_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Explicit "look / peek / dig into my files" family.
        r"\b(?:look|peek|dig|go)\s+(?:in|into|inside|through|through\s+all)\s+(?:my|the)\s+(?:files?|documents?|docs?|notes?|library|folder|folders|directory|knowledge\s*base|kb)\b",
        # Explicit "check / search my files" family.
        r"\b(?:check|search|scan|browse|read|review|find\s+(?:in|something\s+in))\s+(?:in\s+|through\s+|inside\s+)?(?:my|the)\s+(?:files?|documents?|docs?|notes?|library|folder|folders|directory|knowledge\s*base|kb)\b",
        # "in my files / documents / notes" used as a locus.
        r"\bin\s+(?:my|the)\s+(?:files?|documents?|docs?|notes?|library|knowledge\s*base|kb)\b",
        # "is there a / any mention|reference|note of X in ..."
        r"\bis\s+there\s+(?:a|any)\s+(?:mention|reference|note|record|entry)\s+(?:of|about|for)\b",
        # "does (it|this|my file|my document|my notes) (say|mention|contain|talk about)"
        r"\bdoes\s+(?:it|this|that|my\s+(?:file|document|notes?|docs?|library))\s+(?:say|mention|contain|talk\s+about|reference)\b",
        # "according to my files / documents / notes"
        r"\baccording\s+to\s+(?:my|the)\s+(?:files?|documents?|docs?|notes?|library|knowledge\s*base|kb)\b",
        # "from my files / documents / notes / library"
        r"\bfrom\s+(?:my|the)\s+(?:files?|documents?|docs?|notes?|library|knowledge\s*base|kb)\b",
        # "pull up / open the document ..."
        r"\b(?:pull\s+up|open|show\s+me)\s+(?:my|the)\s+(?:files?|documents?|docs?|notes?|library|folder)\b",
        # "what do my (files|documents|notes) say about X"
        r"\bwhat\s+do\s+my\s+(?:files?|documents?|docs?|notes?|library)\s+(?:say|contain|mention)\b",
    )
)


def detect_file_search_intent(user_text: str) -> bool:
    """True when the user is explicitly asking Qube to search the local library.

    This detector is used by ``workers/llm_worker.py`` to force an RAG-only
    route that bypasses the cognitive router's semantic recall centroid —
    otherwise queries like "look into my files and tell me if there is a
    mention of <person>" are misclassified as recall/HYBRID and pull
    unrelated memories into the prompt.
    """
    if not user_text:
        return False
    s = user_text.strip()
    if not s:
        return False
    for p in _FILE_SEARCH_INTENT_PATTERNS:
        if p.search(s):
            return True
    return False


# ============================================================
# System-prompt suffix that disciplines the LLM's use of memory sources
# on MEMORY / HYBRID / RAG routes.
# ============================================================
RECALL_FUSION_SYSTEM_SUFFIX: str = (
    " Memory entries describe the user's own preferences, identity, projects, "
    "and things the user explicitly asked you to remember. "
    "If a memory entry consists only of a name or a very short phrase, do NOT "
    "answer the user's question from that entry alone — consult the numbered "
    "document sources for details and cite those instead. "
    "Ignore any memory entry that claims a system limitation "
    "(for example that you have no internet access, that you cannot browse, "
    "or that you are offline); such entries are stale and must not shape "
    "your behavior."
)


# ============================================================
# System-prompt suffix appended when ``detect_file_search_intent`` fires.
# Discourages the model from injecting memory-derived facts into an
# answer that the user explicitly scoped to document search.
# ============================================================
FILE_SEARCH_SYSTEM_SUFFIX: str = (
    " The user has explicitly asked you to search their local files / "
    "documents / notes. Answer ONLY from the numbered document sources "
    "below. If none of the document sources contain the requested "
    "information, say so plainly in one sentence. Do NOT bring in facts "
    "from long-term memory (user preferences, names, or unrelated stored "
    "memories) on this turn, even if memory entries happen to be "
    "available. Always answer with natural prose."
)


# ============================================================
# Citation-discipline suffix. Prevents the degenerate failure mode where
# a small LLM emits *only* a bare citation token (e.g. "[2]") as the
# entire reply when two conflicting sources are in context.
# ============================================================
CITATION_DISCIPLINE_SUFFIX: str = (
    " NEVER reply with only a bracket citation token like [1], [2], or [W]. "
    "A citation annotates a sentence — it is never the whole answer. "
    "Always write a full natural-language reply that answers the user's "
    "question in your own words; place citations at the end of the "
    "sentences they support. If none of the sources are relevant, "
    "say so plainly in a sentence without brackets."
)


# ============================================================
# Grounded-answer / anti-confabulation suffix.
# ------------------------------------------------------------
# Applied on every retrieval-bearing route (RAG / HYBRID / MEMORY / WEB).
# Forbids the small-LLM failure mode where the model, faced with a source
# that is *about the wrong entity*, confabulates a plausible answer by
# inventing a last name or qualifier ("Dr. Evelyn" -> "Dr. Evelyn Vogel")
# and cites the unrelated source anyway.
# ============================================================
GROUNDED_ANSWER_SYSTEM_SUFFIX: str = (
    " Strict grounding rules: "
    "(1) Do NOT invent or extend names — if a source mentions 'Evelyn' "
    "and the user asks about 'Dr. Evelyn', do not add a last name, "
    "affiliation, or qualifier that is not literally in the source. "
    "(2) Do NOT merge or connect entities from different sources into a "
    "single claim unless the sources themselves make that connection. "
    "(3) A source is only evidence for what it literally states — if a "
    "source talks about Entity A and the user asks about Entity B, that "
    "source is NOT evidence about Entity B, even if they share a role "
    "or a first name. Treat it as not-relevant. "
    "(4) If after this check no source actually answers the user's "
    "question, say plainly in one sentence that you don't have that "
    "information — never fabricate a plausible-sounding answer."
)


# ============================================================
# No-sources suffix.
# ------------------------------------------------------------
# Appended when the retrieval layer returned zero useful sources on a
# turn that the router thought should retrieve. This is the contract the
# user relied on previously: "if no matching file, just say so." With an
# active retrieval system prompt but no sources, a small LLM otherwise
# tends to fabricate.
# ============================================================
NO_SOURCES_SYSTEM_SUFFIX: str = (
    " IMPORTANT: retrieval ran for this turn but found no relevant "
    "sources in the user's local files or long-term memory. Answer "
    "plainly in one or two sentences that you couldn't find anything "
    "matching the user's question in their files or memory. Do NOT "
    "emit any bracket citation tokens such as [1], [2], or [W]. Do NOT "
    "invent names, dates, affiliations, or other details."
)


# ============================================================
# Memory tiering (T3.4).
# ------------------------------------------------------------
# A pure function mapping an extracted fact payload to its
# structural tier ("preference" / "knowledge" / "episode" /
# "context"). The tier is embedded in the LanceDB ``source``
# field as ``qube_memory::<tier>::<category>``. Retrieval uses
# the tier as a cheap WHERE filter so ordinary chat turns see
# only preferences (+ legacy context) while recall / hybrid /
# narrative turns also see knowledge and episodes.
#
# Classification order (first match wins):
#   1. category=="episode"                               -> episode
#   2. subject=="user" AND origin in user_stated /
#      user_confirmed                                    -> preference
#   3. fact["_explicit_remember"] is True                -> knowledge
#   4. subject=="third_party"                            -> knowledge
#   5. origin=="document_derived"                        -> knowledge
#   6. anything else                                     -> context
# ============================================================
MEMORY_TIERS: tuple[str, ...] = (
    "preference",
    "knowledge",
    "episode",
    "context",
)


def derive_memory_tier(fact: dict) -> str:
    """Deterministic mapping from a fact payload to its storage tier.

    Defensive against missing / malformed keys: an empty dict or a dict
    whose ``category`` / ``subject`` / ``origin`` are missing falls back
    to ``"context"``. Unknown category values also fall back to
    ``"context"``.
    """
    if not isinstance(fact, dict):
        return "context"

    category = str(fact.get("category") or "").strip().lower()
    subject = str(fact.get("subject") or "").strip().lower()
    origin = str(fact.get("origin") or "").strip().lower()
    explicit_remember = bool(fact.get("_explicit_remember", False))

    if category == "episode":
        return "episode"
    if subject == "user" and origin in {"user_stated", "user_confirmed"}:
        return "preference"
    if explicit_remember:
        return "knowledge"
    if subject == "third_party":
        return "knowledge"
    if origin == "document_derived":
        return "knowledge"
    return "context"


# ============================================================
# Narrative / recap suffix (T3.2).
# ------------------------------------------------------------
# Appended to the system prompt when ``detect_narrative_intent`` fires.
# Tells the LLM to prefer the EPISODE-labelled sources (session
# summaries) over atomic facts on a recap-style turn.
# ============================================================
NARRATIVE_RECALL_SYSTEM_SUFFIX: str = (
    " This is a narrative/recap question. Prefer the EPISODE-labelled "
    "memory sources (session summaries) over atomic facts; cite them "
    "inline. If no episode summary is available, answer concisely from "
    "atomic facts that are clearly relevant. Do not invent a session "
    "history that is not reflected in the sources."
)


__all__ = [
    "ASSISTANT_FAILURE_PATTERNS",
    "is_assistant_failure_message",
    "is_thin_content",
    "detect_explicit_remember",
    "detect_recall_intent",
    "detect_narrative_intent",
    "detect_file_search_intent",
    "derive_memory_tier",
    "MEMORY_TIERS",
    "RECALL_FUSION_SYSTEM_SUFFIX",
    "FILE_SEARCH_SYSTEM_SUFFIX",
    "CITATION_DISCIPLINE_SUFFIX",
    "GROUNDED_ANSWER_SYSTEM_SUFFIX",
    "NO_SOURCES_SYSTEM_SUFFIX",
    "NARRATIVE_RECALL_SYSTEM_SUFFIX",
]
