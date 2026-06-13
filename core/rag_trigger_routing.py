"""Custom NLP RAG trigger routing helpers (Settings -> LLMWorker)."""

from __future__ import annotations

import re

# Seeded on first DB init; missing phrases are backfilled on later launches.
DEFAULT_RAG_TRIGGERS: tuple[str, ...] = (
    "search my files",
    "in my documents",
    "according to my knowledge base",
    "based on my files",
    "check my library",
    "from my files",
    "from my library",
    "from my knowledge base",
    "from my pdf",
    "based on my library",
)

# Block retrieval triggers on app-management / how-to prompts (e.g. deleting
# library entries, uploading docs, or asking how Qube's library works).
_OPERATIONAL_RAG_TRIGGER_BLOCKERS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bhow\s+(?:can|do|to)\s+i\s+(?:remove|delete|add|move|organize|manage|clear|edit|rename|upload|download|import|export|install|uninstall|sync|backup|restore)\b",
        r"\bhow\s+to\s+(?:remove|delete|add|move|organize|manage|clear|edit|rename|upload|download|import|export|install|uninstall|sync|backup|restore)\b",
        r"\bhow\s+(?:can|do|to)\s+i\s+(?:use|access|open|navigate|set\s+up|configure)\s+(?:the\s+)?(?:library|knowledge\s*base|file\s+(?:manager|browser|upload))\b",
        r"\bhow\s+(?:does|do)\s+(?:qube|the\s+app|this\s+app|the\s+library|my\s+library)\s+work\b",
        r"\bhow\s+(?:can|do|to)\s+i\s+(?:add|upload|import|ingest|put)\s+.+\s+(?:to|into)\s+my\s+(?:library|files|documents|knowledge\s*base)\b",
        r"\b(?:remove|delete|add|move|clear|edit|rename|upload|download|import|export)\s+(?:entries?|files?|documents?|items?|stuff|things?|a\s+file|an\s+entry)\s+from\s+my\s+(?:library|files|documents|knowledge\s*base)\b",
    )
)


def is_operational_library_prompt(clean_prompt: str) -> bool:
    """True when the user is asking how to manage the app/library, not retrieve."""
    if not clean_prompt:
        return False
    return any(p.search(clean_prompt) for p in _OPERATIONAL_RAG_TRIGGER_BLOCKERS)


def matches_custom_rag_trigger(
    clean_prompt: str,
    triggers: tuple[str, ...] | list[str],
) -> bool:
    if not clean_prompt or not triggers:
        return False
    if is_operational_library_prompt(clean_prompt):
        return False
    normalized = clean_prompt.lower().strip()
    return any(t in normalized for t in triggers)


def apply_custom_rag_trigger_route(
    execution_route: str,
    *,
    matched: bool,
) -> tuple[str, bool]:
    """
    Apply a custom NLP RAG trigger match to the execution route.

    Returns ``(execution_route, force_rag_via_trigger)``.

    * ``NONE`` → ``RAG`` with a one-turn library bypass when the master
      Knowledge Base toggle is off.
    * ``MEMORY`` → ``HYBRID`` so memory retrieval is preserved while the
      library leg is added.
    * ``HYBRID`` is left unchanged; ``force_rag_via_trigger`` is still set
      so the RAG leg runs under master-off.
    * ``WEB`` / ``INTERNET`` are unchanged here (the WEB block may override
      later in ``LLMWorker``).
    """
    route = str(execution_route or "NONE").upper()
    if not matched:
        return route, False

    if route == "HYBRID":
        return route, True
    if route in ("WEB", "INTERNET"):
        return route, False
    if route == "NONE":
        return "RAG", True
    if route == "MEMORY":
        return "HYBRID", True
    return route, True
