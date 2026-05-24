"""Custom NLP RAG trigger routing helpers (Settings -> LLMWorker)."""

from __future__ import annotations


def matches_custom_rag_trigger(
    clean_prompt: str,
    triggers: tuple[str, ...] | list[str],
) -> bool:
    if not clean_prompt or not triggers:
        return False
    return any(t in clean_prompt for t in triggers)


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
