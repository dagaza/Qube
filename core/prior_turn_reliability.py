"""Prompt hints when a prior assistant turn was suppressed from session history."""
from __future__ import annotations

from typing import Any

from core.history_degeneration import HISTORY_SUPPRESSION_PLACEHOLDER

PRIOR_TURN_UNRELIABLE_SUFFIX = (
    " The immediately previous assistant reply was unreliable and has been removed "
    "from conversation history. Do not infer facts from that missing turn; answer "
    "from earlier reliable context and general knowledge only. If uncertain, say so."
)


def history_contains_suppressed_assistant(
    history: list[dict[str, Any]] | None,
) -> bool:
    """True when the latest prior assistant message is a degeneration placeholder."""
    if not history:
        return False
    for msg in reversed(history):
        role = str(msg.get("role", "")).lower()
        if role != "assistant":
            continue
        content = str(msg.get("content") or "").strip()
        return content == HISTORY_SUPPRESSION_PLACEHOLDER
    return False


def build_prior_turn_unreliable_suffix() -> str:
    return PRIOR_TURN_UNRELIABLE_SUFFIX
