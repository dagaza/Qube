"""Typed discovery/search outcomes for traces, inspector, and ops logging."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from core.knowledge.adapters.duckduckgo import failure_sentinel_reason, is_failure_sentinel


class SearchOutcomeKind(StrEnum):
    SERP_SUCCESS = "serp_success"
    BOT_CHALLENGE = "bot_challenge"
    EMPTY_PARSE = "empty_parse"
    NETWORK_ERROR = "network_error"
    NO_RESULTS = "no_results"
    NO_CANDIDATES = "no_candidates"
    RELEVANCE_FILTERED = "relevance_filtered"


_SEARCH_OUTCOME_LABELS: dict[SearchOutcomeKind, str] = {
    SearchOutcomeKind.SERP_SUCCESS: "SERP success",
    SearchOutcomeKind.BOT_CHALLENGE: "Search engine bot challenge",
    SearchOutcomeKind.EMPTY_PARSE: "Empty SERP parse",
    SearchOutcomeKind.NETWORK_ERROR: "Network error",
    SearchOutcomeKind.NO_RESULTS: "No search results",
    SearchOutcomeKind.NO_CANDIDATES: "No URL candidates",
    SearchOutcomeKind.RELEVANCE_FILTERED: "Relevance gate filtered all",
}


@dataclass(frozen=True)
class SearchOutcome:
    kind: SearchOutcomeKind
    provider: str = "duckduckgo"
    http_status: int | None = None
    parsed_rows: int = 0
    candidate_count: int = 0
    bot_challenge_signals: tuple[str, ...] = ()
    failure_sentinel_reason: str | None = None
    recovery_hint: str | None = None
    fallback_from: str | None = None
    fallback_reason: str | None = None
    primary_outcome: dict[str, Any] | None = None
    # S12: seconds from a provider's Retry-After (HTTP 429), unbounded as sent.
    retry_after_s: float | None = None

    @property
    def block_cause(self) -> str | None:
        """Why discovery was blocked, or ``None`` (see ``discovery_block_cause``)."""
        return discovery_block_cause(self.failure_sentinel_reason, self.bot_challenge_signals)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "provider": self.provider,
            "http_status": self.http_status,
            "parsed_rows": self.parsed_rows,
            "candidate_count": self.candidate_count,
            "bot_challenge_signals": list(self.bot_challenge_signals),
            "failure_sentinel_reason": self.failure_sentinel_reason,
            "recovery_hint": self.recovery_hint,
        }
        cause = self.block_cause
        if cause:
            payload["block_cause"] = cause
        if self.retry_after_s is not None:
            payload["retry_after_s"] = self.retry_after_s
        if self.fallback_from:
            payload["fallback_from"] = self.fallback_from
        if self.fallback_reason:
            payload["fallback_reason"] = self.fallback_reason
        if self.primary_outcome:
            payload["primary_outcome"] = dict(self.primary_outcome)
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> SearchOutcome | None:
        if not isinstance(raw, Mapping):
            return None
        kind_raw = str(raw.get("kind") or "").strip().lower()
        try:
            kind = SearchOutcomeKind(kind_raw)
        except ValueError:
            return None
        signals = raw.get("bot_challenge_signals")
        if not isinstance(signals, list):
            signals = ()
        return cls(
            kind=kind,
            provider=str(raw.get("provider") or "duckduckgo"),
            http_status=raw.get("http_status"),
            parsed_rows=int(raw.get("parsed_rows") or 0),
            candidate_count=int(raw.get("candidate_count") or 0),
            bot_challenge_signals=tuple(str(s) for s in signals),
            failure_sentinel_reason=(
                str(raw.get("failure_sentinel_reason"))
                if raw.get("failure_sentinel_reason")
                else None
            ),
            recovery_hint=(
                str(raw.get("recovery_hint")) if raw.get("recovery_hint") else None
            ),
            fallback_from=(
                str(raw.get("fallback_from")) if raw.get("fallback_from") else None
            ),
            retry_after_s=(
                float(raw["retry_after_s"]) if raw.get("retry_after_s") is not None else None
            ),
            fallback_reason=(
                str(raw.get("fallback_reason")) if raw.get("fallback_reason") else None
            ),
            primary_outcome=(
                dict(raw.get("primary_outcome"))
                if isinstance(raw.get("primary_outcome"), Mapping)
                else None
            ),
        )

    @property
    def label(self) -> str:
        return _SEARCH_OUTCOME_LABELS.get(self.kind, self.kind.value)

    @property
    def is_failure(self) -> bool:
        return self.kind != SearchOutcomeKind.SERP_SUCCESS


# S12 (V15): Qube's own limits and provider blocks both surface as
# ``bot_challenge`` outcomes (so the fallback chain runs), but they need
# different handling and reporting.
BLOCK_CAUSE_QUBE_LIMIT = "qube_limit"  # burst / session budget, pacing timeout
BLOCK_CAUSE_PROVIDER_CHALLENGE = "provider_challenge"  # anomaly page, HTTP 403, or a pause after one
BLOCK_CAUSE_PROVIDER_RATE_LIMIT = "provider_rate_limit"  # HTTP 429

_QUBE_LIMIT_REASONS = frozenset(
    {"burst_budget_exhausted", "session_budget_exhausted", "ddg_pacing_timeout"}
)
_QUBE_LIMIT_SIGNALS = frozenset({"burst_budget_exhausted", "session_budget_exhausted", "pacing_timeout"})
_PROVIDER_CHALLENGE_REASONS = frozenset({"ddg_bot_challenge", "ddg_backoff", "ddg_forbidden"})

BLOCK_CAUSE_LABELS = {
    BLOCK_CAUSE_QUBE_LIMIT: "Qube's own search limit",
    BLOCK_CAUSE_PROVIDER_CHALLENGE: "provider bot challenge",
    BLOCK_CAUSE_PROVIDER_RATE_LIMIT: "provider rate limit",
}


def discovery_block_cause(
    failure_sentinel_reason: str | None,
    signals: tuple[str, ...] | list[str] = (),
) -> str | None:
    reason = str(failure_sentinel_reason or "").strip().lower()
    if reason in _QUBE_LIMIT_REASONS:
        return BLOCK_CAUSE_QUBE_LIMIT
    if reason == "ddg_rate_limited":
        return BLOCK_CAUSE_PROVIDER_RATE_LIMIT
    if reason in _PROVIDER_CHALLENGE_REASONS:
        return BLOCK_CAUSE_PROVIDER_CHALLENGE
    sigs = {str(s).strip().lower() for s in signals or ()}
    if sigs & _QUBE_LIMIT_SIGNALS:
        return BLOCK_CAUSE_QUBE_LIMIT
    return None


def build_search_outcome_from_ddg(
    rows: list[dict[str, Any]] | None,
    inspection: Mapping[str, Any] | None,
    *,
    candidate_count: int = 0,
    provider: str = "duckduckgo",
) -> SearchOutcome:
    """Map DDG rows + optional inspection metadata to a typed outcome."""
    safe_rows = [dict(r) for r in (rows or []) if isinstance(r, dict)]
    inspect = dict(inspection or {})
    response_kind = str(inspect.get("response_kind") or "").strip().lower()
    http_status = inspect.get("http_status")
    parsed_rows = int(inspect.get("parsed_rows") or 0)
    bot_signals = tuple(
        str(s) for s in (inspect.get("bot_challenge_signals") or ()) if s
    )
    sentinel_reason = failure_sentinel_reason(safe_rows)
    retry_after_s: float | None = None

    if response_kind == "rate_limited":
        from core.knowledge.http_client import retry_after_seconds

        kind = SearchOutcomeKind.BOT_CHALLENGE
        sentinel_reason = "ddg_rate_limited"
        retry_after_s = retry_after_seconds({"Retry-After": inspect.get("retry_after")}) if inspect.get("retry_after") else None
        recovery = "DuckDuckGo rate limit (HTTP 429); fallbacks may still apply."
    elif response_kind == "forbidden":
        kind = SearchOutcomeKind.BOT_CHALLENGE
        sentinel_reason = "ddg_forbidden"
        recovery = "DuckDuckGo refused the request (HTTP 403); retry later or change network."
    elif sentinel_reason == "ddg_bot_challenge" or response_kind == "bot_challenge":
        kind = SearchOutcomeKind.BOT_CHALLENGE
        recovery = "Retry later or change network."
    elif sentinel_reason == "ddg_pacing_timeout" or response_kind == "pacing_timeout":
        kind = SearchOutcomeKind.BOT_CHALLENGE
        recovery = "Discovery pacing timeout; fallbacks may still apply."
    elif sentinel_reason == "network_error":
        kind = SearchOutcomeKind.NETWORK_ERROR
        recovery = "Check network connectivity and retry."
    elif sentinel_reason == "ddg_empty_parse" or response_kind == "empty_parse":
        kind = SearchOutcomeKind.EMPTY_PARSE
        recovery = "Search HTML may have changed; retry later."
    elif candidate_count > 0 and not is_failure_sentinel(safe_rows):
        kind = SearchOutcomeKind.SERP_SUCCESS
        recovery = None
    elif safe_rows and not is_failure_sentinel(safe_rows):
        kind = SearchOutcomeKind.NO_CANDIDATES
        recovery = "SERP rows lacked parseable URLs."
    else:
        kind = SearchOutcomeKind.NO_RESULTS
        recovery = None

    return SearchOutcome(
        kind=kind,
        provider=provider,
        http_status=int(http_status) if http_status is not None else None,
        parsed_rows=parsed_rows,
        candidate_count=int(candidate_count),
        bot_challenge_signals=bot_signals,
        failure_sentinel_reason=sentinel_reason,
        recovery_hint=recovery,
        retry_after_s=retry_after_s,
    )


def build_search_outcome_from_brave(
    rows: list[dict[str, Any]] | None,
    inspection: Mapping[str, Any] | None,
    *,
    candidate_count: int = 0,
    provider: str = "brave_search",
) -> SearchOutcome:
    """Map Brave API rows + inspection metadata to a typed outcome."""
    safe_rows = [dict(r) for r in (rows or []) if isinstance(r, dict)]
    inspect = dict(inspection or {})
    response_kind = str(inspect.get("response_kind") or "").strip().lower()
    http_status = inspect.get("http_status")
    parsed_rows = int(inspect.get("parsed_rows") or len(safe_rows))

    if response_kind == "no_credentials":
        kind = SearchOutcomeKind.NO_RESULTS
        recovery = "Add a Brave Search API key in Settings → Knowledge."
    elif response_kind in {"auth_error"}:
        kind = SearchOutcomeKind.NETWORK_ERROR
        recovery = "Check Brave Search API key in Settings → Knowledge."
    elif response_kind == "network_error":
        kind = SearchOutcomeKind.NETWORK_ERROR
        recovery = "Check network connectivity and retry."
    elif candidate_count > 0:
        kind = SearchOutcomeKind.SERP_SUCCESS
        recovery = None
    elif safe_rows:
        kind = SearchOutcomeKind.NO_CANDIDATES
        recovery = "Brave rows lacked parseable URLs."
    else:
        kind = SearchOutcomeKind.NO_RESULTS
        recovery = None

    return SearchOutcome(
        kind=kind,
        provider=provider,
        http_status=int(http_status) if http_status is not None else None,
        parsed_rows=parsed_rows,
        candidate_count=int(candidate_count),
        failure_sentinel_reason=response_kind or None,
        recovery_hint=recovery,
    )


def build_search_outcome_from_provider_rows(
    rows: list[dict[str, Any]] | None,
    *,
    candidate_count: int = 0,
    provider: str,
) -> SearchOutcome:
    """Map generic provider rows (title/snippet/url) to a typed outcome."""
    safe_rows = [dict(r) for r in (rows or []) if isinstance(r, dict)]
    parsed_rows = len(safe_rows)
    if candidate_count > 0:
        kind = SearchOutcomeKind.SERP_SUCCESS
        recovery = None
    elif safe_rows:
        kind = SearchOutcomeKind.NO_CANDIDATES
        recovery = "Provider rows lacked parseable URLs."
    else:
        kind = SearchOutcomeKind.NO_RESULTS
        recovery = None
    return SearchOutcome(
        kind=kind,
        provider=provider,
        parsed_rows=parsed_rows,
        candidate_count=int(candidate_count),
        recovery_hint=recovery,
    )


def with_discovery_fallback(
    outcome: SearchOutcome,
    *,
    fallback_from: str,
    fallback_reason: str,
    primary_outcome: SearchOutcome | None,
) -> SearchOutcome:
    """Attach fallback metadata when a secondary provider succeeded."""
    return SearchOutcome(
        kind=outcome.kind,
        provider=outcome.provider,
        http_status=outcome.http_status,
        parsed_rows=outcome.parsed_rows,
        candidate_count=outcome.candidate_count,
        bot_challenge_signals=outcome.bot_challenge_signals,
        failure_sentinel_reason=outcome.failure_sentinel_reason,
        recovery_hint=outcome.recovery_hint,
        fallback_from=fallback_from,
        fallback_reason=fallback_reason,
        primary_outcome=primary_outcome.to_dict() if primary_outcome else None,
        retry_after_s=outcome.retry_after_s,
    )


def search_outcome_from_relevance_diag(
    relevance_diag: Mapping[str, Any] | None,
) -> SearchOutcome | None:
    if not relevance_diag:
        return None
    raw = relevance_diag.get("search_outcome")
    if isinstance(raw, Mapping):
        return SearchOutcome.from_dict(raw)
    return None


def attach_search_outcome(
    relevance_diag: dict[str, Any] | None,
    outcome: SearchOutcome | None,
) -> dict[str, Any]:
    diag = dict(relevance_diag or {})
    if outcome is not None:
        diag["search_outcome"] = outcome.to_dict()
    return diag


def format_search_outcome_summary_line(outcome: SearchOutcome | None) -> str | None:
    if outcome is None:
        return None
    from core.knowledge.discovery.policy import discovery_provider_label

    provider_label = discovery_provider_label(outcome.provider)
    parts = [f"Search: {outcome.label} ({provider_label})"]
    if outcome.http_status is not None:
        parts.append(f"http={outcome.http_status}")
    if outcome.parsed_rows:
        parts.append(f"parsed_rows={outcome.parsed_rows}")
    if outcome.candidate_count:
        parts.append(f"candidates={outcome.candidate_count}")
    if outcome.fallback_from:
        parts.append(
            "fallback_from=" + discovery_provider_label(outcome.fallback_from)
        )
    cause = outcome.block_cause or (outcome.primary_outcome or {}).get("block_cause")
    if cause:
        parts.append("cause=" + BLOCK_CAUSE_LABELS.get(str(cause), str(cause)))
    if outcome.bot_challenge_signals:
        parts.append(
            "signals="
            + ",".join(outcome.bot_challenge_signals[:4])
            + ("…" if len(outcome.bot_challenge_signals) > 4 else "")
        )
    return " | ".join(parts)


def degraded_discovery_note(outcome: SearchOutcome | None) -> str:
    """One sentence when web discovery did not run on its primary provider (S12, V15).

    "" when discovery ran normally. Used in the bundle's coverage rationale so
    the inspector and trace say that a web search became, say, a Wikipedia search.
    """
    if outcome is None:
        return ""
    from core.knowledge.discovery.policy import discovery_provider_label

    if outcome.fallback_from:
        cause = (outcome.primary_outcome or {}).get("block_cause")
        why = BLOCK_CAUSE_LABELS.get(str(cause or ""), "it was unavailable")
        return (
            f"Search ran on {discovery_provider_label(outcome.provider)} instead of "
            f"{discovery_provider_label(outcome.fallback_from)} ({why})."
        )
    if outcome.kind == SearchOutcomeKind.BOT_CHALLENGE:
        why = BLOCK_CAUSE_LABELS.get(str(outcome.block_cause or ""), "blocked")
        return f"Search on {discovery_provider_label(outcome.provider)} did not run ({why})."
    return ""


def format_search_outcome_explain_text(outcome: SearchOutcome | None) -> str:
    if outcome is None:
        return ""
    from core.knowledge.discovery.policy import discovery_provider_label

    lines = [
        "Search outcome:",
        f"  kind: {outcome.kind.value}",
        f"  label: {outcome.label}",
        f"  provider: {discovery_provider_label(outcome.provider)}",
    ]
    if outcome.http_status is not None:
        lines.append(f"  http_status: {outcome.http_status}")
    lines.append(f"  parsed_rows: {outcome.parsed_rows}")
    lines.append(f"  candidate_count: {outcome.candidate_count}")
    if outcome.fallback_from:
        lines.append(
            "  fallback_from: " + discovery_provider_label(outcome.fallback_from)
        )
    if outcome.fallback_reason:
        lines.append(f"  fallback_reason: {outcome.fallback_reason}")
    if outcome.primary_outcome:
        primary_kind = outcome.primary_outcome.get("kind")
        primary_provider = outcome.primary_outcome.get("provider")
        if primary_kind or primary_provider:
            lines.append(
                "  primary_outcome: "
                + ", ".join(
                    part
                    for part in (
                        f"kind={primary_kind}" if primary_kind else "",
                        f"provider={primary_provider}" if primary_provider else "",
                    )
                    if part
                )
            )
    if outcome.failure_sentinel_reason:
        lines.append(f"  sentinel_reason: {outcome.failure_sentinel_reason}")
    if outcome.bot_challenge_signals:
        lines.append(
            "  bot_signals: " + ", ".join(outcome.bot_challenge_signals)
        )
    if outcome.recovery_hint:
        lines.append(f"  recovery: {outcome.recovery_hint}")
    return "\n".join(lines)
