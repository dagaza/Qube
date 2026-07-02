"""Provider quota/limit events and debounced notification dispatch (Slice 11)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal

from core.knowledge.credentials import CredentialMode, resolve_credential
from core.knowledge.provider_status import provider_id_for_metrics_host

logger = logging.getLogger("Qube.Knowledge.ProviderLimits")

ProviderLimitKind = Literal["daily_quota", "rate_limit"]

_DEBOUNCE_SEC = 86400.0

_limit_handler: Callable[["ProviderLimitEvent"], None] | None = None
_last_notified_at: dict[str, float] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class ProviderLimitEvent:
    provider_id: str
    kind: ProviderLimitKind
    metrics_host: str
    resets_at: float | None = None


def register_provider_limit_handler(
    handler: Callable[[ProviderLimitEvent], None] | None,
) -> None:
    global _limit_handler
    _limit_handler = handler


def utc_midnight_after(now: datetime | None = None) -> float:
    """Return epoch seconds for the next UTC midnight (OpenAlex daily reset)."""
    current = now or datetime.now(timezone.utc)
    next_day = (current + timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return next_day.timestamp()


def notify_budget_exhausted(
    *,
    metrics_host: str,
    kind: ProviderLimitKind = "daily_quota",
    resets_at: float | None = None,
) -> None:
    """Emit a debounced provider limit event for anonymous quota exhaustion."""
    provider_id = provider_id_for_metrics_host(metrics_host)
    if not provider_id:
        return

    cred = resolve_credential(provider_id)
    if cred.mode != CredentialMode.ANONYMOUS:
        return

    now = datetime.now(timezone.utc).timestamp()
    with _lock:
        last = _last_notified_at.get(provider_id, 0.0)
        if now - last < _DEBOUNCE_SEC:
            logger.debug(
                "Skipping debounced provider limit notification for %s",
                provider_id,
            )
            return
        _last_notified_at[provider_id] = now

    event = ProviderLimitEvent(
        provider_id=provider_id,
        kind=kind,
        metrics_host=metrics_host,
        resets_at=resets_at if resets_at is not None else utc_midnight_after(),
    )
    handler = _limit_handler
    if handler is None:
        logger.debug("No provider limit handler registered for %s", provider_id)
        return
    try:
        handler(event)
    except Exception:
        logger.exception("Provider limit handler failed for %s", provider_id)


def reset_provider_limit_notify_state_for_tests() -> None:
    """Clear debounce state (unit tests only)."""
    with _lock:
        _last_notified_at.clear()
