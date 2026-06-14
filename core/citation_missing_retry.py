"""One-shot retry when a WEB turn omits required bracket citations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.citation_integrity import analyze_citations

_FIXUP_USER = (
    "Your previous answer omitted required citations. Rewrite the answer. "
    "Every factual claim derived from the provided sources MUST end with a citation token. "
    "Valid citation ids for this turn are: [1], [2], [3] etc. Use only ids like these. "
    "Do not change the meaning of the answer. Do not add explanations about citations. "
    "Return only the corrected answer."
)


@dataclass(frozen=True)
class MissingCitationRetryOutcome:
    text: str
    retry_attempted: bool = False
    retry_used: bool = False
    retry_reason: str = ""


def maybe_retry_missing_web_citations(
    engine: Any,
    messages: list[dict],
    text: str,
    sources: list[dict],
    *,
    max_tokens: int = 512,
) -> MissingCitationRetryOutcome:
    """Regenerate once with a citation fixup nudge when web sources were not cited."""
    original = (text or "").strip()
    report = analyze_citations(original, sources)
    if not report.missing_citation:
        return MissingCitationRetryOutcome(original, retry_reason="not_missing")

    contract = getattr(engine, "_last_prompt_contract", None)
    exec_fn = getattr(engine, "execute_from_contract", None)
    if contract is None or not callable(exec_fn):
        return MissingCitationRetryOutcome(original, retry_reason="no_engine_contract")

    retry_messages = list(messages or []) + [
        {"role": "assistant", "content": original},
        {"role": "user", "content": _FIXUP_USER},
    ]
    try:
        retried = str(exec_fn(contract, retry_messages) or "").strip()
    except Exception:
        return MissingCitationRetryOutcome(
            original,
            retry_attempted=True,
            retry_reason="engine_error",
        )

    if not retried:
        return MissingCitationRetryOutcome(
            original,
            retry_attempted=True,
            retry_reason="empty_retry",
        )

    post = analyze_citations(retried, sources)
    if not post.cited_ids:
        return MissingCitationRetryOutcome(
            original,
            retry_attempted=True,
            retry_reason="retry_still_missing_citations",
        )
    if post.has_violation:
        return MissingCitationRetryOutcome(
            original,
            retry_attempted=True,
            retry_reason="retry_introduced_orphan_citations",
        )

    _ = max_tokens  # reserved for future budget wiring through execute_from_contract
    return MissingCitationRetryOutcome(
        retried,
        retry_attempted=True,
        retry_used=True,
        retry_reason="missing_citations_added",
    )
