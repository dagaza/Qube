"""User-assisted recovery after a DuckDuckGo bot challenge.

When DDG challenges a search, the user can open the same search in their own
browser, complete the human check there, and resume the blocked turn. Resuming
lifts the DDG pause, starts a fresh browser-like session and re-asks the
question with live web search pinned.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote_plus

from core.knowledge.discovery.ddg_http import DDG_HTML_ENDPOINT
from core.knowledge.discovery.policy import PRIMARY_DISCOVERY_PROVIDER_ID

INTERNET_TOOL_TOKEN = "@[tool:internet]"

PENDING_BLOCKED_SEARCH_TTL_SEC = 3600.0

_CHALLENGE_SENTINELS = frozenset({"ddg_bot_challenge", "ddg_backoff"})


@dataclass(frozen=True)
class BlockedWebSearch:
    session_id: str
    persist_content: str
    web_query: str
    created_at: float

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > PENDING_BLOCKED_SEARCH_TTL_SEC


_lock = threading.Lock()
_pending: BlockedWebSearch | None = None


def _is_ddg_challenge_dict(raw: Mapping[str, Any] | None) -> bool:
    if not isinstance(raw, Mapping):
        return False
    if str(raw.get("provider") or "").strip().lower() != PRIMARY_DISCOVERY_PROVIDER_ID:
        return False
    if str(raw.get("kind") or "").strip().lower() != "bot_challenge":
        return False
    sentinel = str(raw.get("failure_sentinel_reason") or "").strip().lower()
    if sentinel in _CHALLENGE_SENTINELS:
        return True
    signals = {str(s).strip().lower() for s in (raw.get("bot_challenge_signals") or ())}
    return bool(signals) and not signals & {
        "pacing_timeout",
        "burst_budget_exhausted",
        "session_budget_exhausted",
    }


def outcome_lost_ddg_to_challenge(outcome: Any) -> bool:
    """True when DDG was challenged (or paused after one) for this search.

    Budget exhaustion and pacing timeouts are excluded: verifying in a browser
    does not help with those.
    """
    if outcome is None:
        return False
    payload = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    if not isinstance(payload, Mapping):
        return False
    if _is_ddg_challenge_dict(payload):
        return True
    return _is_ddg_challenge_dict(payload.get("primary_outcome"))


def record_blocked_search(
    *,
    session_id: str,
    persist_content: str,
    web_query: str,
) -> None:
    global _pending
    content = (persist_content or "").strip()
    if not session_id or not content:
        return
    with _lock:
        _pending = BlockedWebSearch(
            session_id=str(session_id),
            persist_content=content,
            web_query=(web_query or "").strip() or content,
            created_at=time.time(),
        )


def get_pending_blocked_search() -> BlockedWebSearch | None:
    global _pending
    with _lock:
        if _pending is not None and _pending.expired:
            _pending = None
        return _pending


def take_pending_blocked_search() -> BlockedWebSearch | None:
    global _pending
    with _lock:
        pending = _pending
        _pending = None
    if pending is None or pending.expired:
        return None
    return pending


def clear_pending_blocked_search() -> None:
    global _pending
    with _lock:
        _pending = None


def ddg_verification_url(query: str | None = None) -> str:
    """Same endpoint Qube searches, so the browser shows the same human check."""
    q = (query or "").strip()
    if not q:
        return DDG_HTML_ENDPOINT
    return f"{DDG_HTML_ENDPOINT}?q={quote_plus(q)}"


def resume_prompt_text(persist_content: str) -> str:
    """Re-ask the blocked message with live web search pinned."""
    content = (persist_content or "").strip()
    if "@[tool:" in content:
        return content
    return f"{INTERNET_TOOL_TOKEN} {content}"


def lift_ddg_pause_after_user_verification() -> None:
    """Clear the DDG pause and start a fresh browsing session."""
    from core.knowledge.discovery.backoff import clear_provider_backoff
    from core.knowledge.discovery.ddg_http import reset_ddg_browser_session

    clear_provider_backoff(PRIMARY_DISCOVERY_PROVIDER_ID)
    reset_ddg_browser_session()
