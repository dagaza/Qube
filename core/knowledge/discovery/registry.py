"""Discovery provider registry."""

from __future__ import annotations

import logging
from dataclasses import replace

from core.knowledge.discovery.policy import PRIMARY_DISCOVERY_PROVIDER_ID
from core.knowledge.discovery.session_budget import (
    discovery_budget_log_fields,
    format_burst_budget_summary,
    format_session_budget_summary,
    get_ddg_budget_block_reason,
    is_ddg_session_budget_exhausted,
)
from core.knowledge.discovery.backoff import (
    clear_provider_backoff,
    format_backoff_summary,
    get_provider_backoff,
    is_provider_in_backoff,
    mark_provider_backoff,
)
from core.knowledge.discovery.cache import (
    get_cached_discovery,
    store_cached_discovery,
)
from core.knowledge.discovery.types import DiscoveryProvider, DiscoveryResult
from core.knowledge.search_outcome import (
    SearchOutcome,
    SearchOutcomeKind,
    with_discovery_fallback,
)

logger = logging.getLogger("Qube.Knowledge.Discovery")

_PROVIDERS: dict[str, DiscoveryProvider] = {}


def register_discovery_provider(provider: DiscoveryProvider) -> None:
    provider_id = (provider.id or "").strip().lower()
    if not provider_id:
        raise ValueError("DiscoveryProvider.id is required")
    _PROVIDERS[provider_id] = provider


def get_discovery_provider(provider_id: str) -> DiscoveryProvider | None:
    return _PROVIDERS.get((provider_id or "").strip().lower())


def list_discovery_providers() -> list[str]:
    return sorted(_PROVIDERS.keys())


def default_discovery_provider() -> DiscoveryProvider:
    provider = _PROVIDERS.get(PRIMARY_DISCOVERY_PROVIDER_ID)
    if provider is None:
        raise RuntimeError(
            f"Default discovery provider {PRIMARY_DISCOVERY_PROVIDER_ID!r} is not registered"
        )
    return provider


def fallback_discovery_provider() -> DiscoveryProvider | None:
    chain = bot_challenge_fallback_chain()
    if not chain:
        return None
    return get_discovery_provider(chain[-1])


def _provider_discover_full(
    provider: DiscoveryProvider,
    query: str,
    *,
    max_results: int,
    site_bias: tuple[str, ...] | None,
    use_cache: bool = True,
    retrieval_profile: str | None = None,
) -> DiscoveryResult:
    provider_id = getattr(provider, "id", PRIMARY_DISCOVERY_PROVIDER_ID)
    if use_cache:
        cached = get_cached_discovery(
            provider_id,
            query,
            max_results=max_results,
            site_bias=site_bias,
            retrieval_profile=retrieval_profile,
        )
        if cached is not None:
            logger.info(
                "[Discovery] cache_hit provider=%s query=%r",
                provider_id,
                query[:120],
            )
            return cached

    if hasattr(provider, "discover_full"):
        result = provider.discover_full(
            query,
            max_results=max_results,
            site_bias=site_bias,
        )
    else:
        candidates = tuple(
            provider.discover(query, max_results=max_results, site_bias=site_bias)
        )
        result = DiscoveryResult(
            candidates=candidates,
            raw_rows=(),
            search_outcome=None,
            provider_id=provider_id,
        )

    if use_cache:
        store_cached_discovery(
            provider_id,
            query,
            max_results=max_results,
            site_bias=site_bias,
            result=result,
            retrieval_profile=retrieval_profile,
        )
    return result


def _synthetic_backoff_result(provider_id: str) -> DiscoveryResult:
    entry = get_provider_backoff(provider_id)
    remaining = entry.remaining_seconds if entry is not None else 0
    outcome = SearchOutcome(
        kind=SearchOutcomeKind.BOT_CHALLENGE,
        provider=provider_id,
        parsed_rows=0,
        candidate_count=0,
        bot_challenge_signals=("backoff_active",),
        failure_sentinel_reason="ddg_backoff",
        recovery_hint=(
            f"Provider paused after bot challenge; retry in ~{max(1, (remaining + 59) // 60)} min."
        ),
    )
    return DiscoveryResult(
        candidates=(),
        raw_rows=(),
        search_outcome=outcome,
        provider_id=provider_id,
    )


def _synthetic_session_budget_result(provider_id: str) -> DiscoveryResult:
    reason = get_ddg_budget_block_reason()
    if reason == "burst":
        summary = format_burst_budget_summary() or "DDG burst limit reached."
        signals = ("burst_budget_exhausted",)
        sentinel = "burst_budget_exhausted"
    else:
        summary = format_session_budget_summary() or "DDG session limit reached."
        signals = ("session_budget_exhausted",)
        sentinel = "session_budget_exhausted"
    outcome = SearchOutcome(
        kind=SearchOutcomeKind.BOT_CHALLENGE,
        provider=provider_id,
        parsed_rows=0,
        candidate_count=0,
        bot_challenge_signals=signals,
        failure_sentinel_reason=sentinel,
        recovery_hint=summary,
    )
    return DiscoveryResult(
        candidates=(),
        raw_rows=(),
        search_outcome=outcome,
        provider_id=provider_id,
    )


def _skip_primary_ddg_result(provider_id: str) -> DiscoveryResult | None:
    if provider_id != PRIMARY_DISCOVERY_PROVIDER_ID:
        return None
    if is_provider_in_backoff(provider_id):
        entry = get_provider_backoff(provider_id)
        summary = format_backoff_summary(entry)
        logger.warning(
            "[Discovery] primary=%s skipped (%s)",
            provider_id,
            summary or "backoff active",
        )
        return _synthetic_backoff_result(provider_id)
    if get_ddg_budget_block_reason() is not None:
        fields = discovery_budget_log_fields()
        summary = format_burst_budget_summary() or format_session_budget_summary()
        logger.warning(
            "[Discovery] primary=%s skipped (%s) burst=%d/%d session=%d/%d",
            provider_id,
            summary or "budget exhausted",
            fields["burst_used"],
            fields["burst_limit"],
            fields["session_used"],
            fields["session_limit"],
        )
        return _synthetic_session_budget_result(provider_id)
    return None


def _record_primary_outcome(provider_id: str, result: DiscoveryResult) -> None:
    from core.knowledge.discovery.health import (
        record_ddg_bot_challenge,
        record_ddg_serp_success,
    )

    outcome = result.search_outcome
    if outcome is None:
        return
    pid = (provider_id or "").strip().lower()
    if pid != PRIMARY_DISCOVERY_PROVIDER_ID:
        return
    if outcome.kind == SearchOutcomeKind.SERP_SUCCESS and result.candidates:
        record_ddg_serp_success()
        clear_provider_backoff(provider_id)
        return
    if outcome.kind == SearchOutcomeKind.BOT_CHALLENGE:
        mark_provider_backoff(provider_id, reason="bot_challenge")
        record_ddg_bot_challenge()


def _should_discovery_fallback(result: DiscoveryResult) -> bool:
    outcome = result.search_outcome
    return (
        outcome is not None
        and outcome.kind == SearchOutcomeKind.BOT_CHALLENGE
    )


def _should_try_fallbacks(primary_id: str, result: DiscoveryResult) -> bool:
    if _should_discovery_fallback(result):
        return True
    if primary_id == PRIMARY_DISCOVERY_PROVIDER_ID:
        return False
    return not result.candidates


def _tag_discovery_result(result: DiscoveryResult, *, privacy_tier: str) -> DiscoveryResult:
    return replace(result, privacy_tier=privacy_tier)


def _merge_fallback_result(
    primary: DiscoveryResult,
    fallback: DiscoveryResult,
) -> DiscoveryResult:
    outcome = fallback.search_outcome
    if outcome is not None:
        outcome = with_discovery_fallback(
            outcome,
            fallback_from=primary.provider_id,
            fallback_reason="bot_challenge",
            primary_outcome=primary.search_outcome,
        )
    return DiscoveryResult(
        candidates=fallback.candidates,
        raw_rows=fallback.raw_rows,
        search_outcome=outcome,
        provider_id=fallback.provider_id,
    )


def _fallback_query_for_provider(
    provider_id: str,
    query: str,
    site_bias: tuple[str, ...] | None,
) -> tuple[str, tuple[str, ...] | None]:
    from core.knowledge.discovery.duckduckgo import format_site_bias_query

    scoped_query, _target = format_site_bias_query(query, site_bias)
    if provider_id == "wikipedia":
        base_query, _ = format_site_bias_query(query, site_bias)
        return base_query, None
    return scoped_query, site_bias


def discover_full_with_fallback(
    query: str,
    *,
    max_results: int = 5,
    site_bias: tuple[str, ...] | None = None,
    primary_provider_id: str | None = None,
    fallback_provider_id: str | None = None,
    retrieval_profile: str | None = None,
) -> DiscoveryResult:
    """Try primary discovery; on failure, walk the tier-aware fallback chain."""
    from core.knowledge.discovery.privacy_policy import resolve_discovery_route

    route = resolve_discovery_route(site_bias=site_bias)
    primary_id = (primary_provider_id or route.primary_id).strip().lower()
    primary = get_discovery_provider(primary_id)
    if primary is None:
        raise ValueError(f"Unknown discovery provider: {primary_id}")

    if primary_id == PRIMARY_DISCOVERY_PROVIDER_ID:
        skipped = _skip_primary_ddg_result(primary_id)
        if skipped is not None:
            primary_result = skipped
        else:
            primary_result = _provider_discover_full(
                primary,
                query,
                max_results=max_results,
                site_bias=site_bias,
                retrieval_profile=retrieval_profile,
            )
            _record_primary_outcome(primary_id, primary_result)
    else:
        primary_result = _provider_discover_full(
            primary,
            query,
            max_results=max_results,
            site_bias=site_bias,
            retrieval_profile=retrieval_profile,
        )
        _record_primary_outcome(primary_id, primary_result)

    if not _should_try_fallbacks(primary_id, primary_result):
        return _tag_discovery_result(primary_result, privacy_tier=route.privacy_tier)

    if fallback_provider_id:
        fallback_ids = (fallback_provider_id.strip().lower(),)
    else:
        fallback_ids = route.fallback_ids

    for fallback_id in fallback_ids:
        if fallback_id == primary_id:
            continue
        fallback = get_discovery_provider(fallback_id)
        if fallback is None:
            continue

        if fallback_id == PRIMARY_DISCOVERY_PROVIDER_ID:
            skipped = _skip_primary_ddg_result(fallback_id)
            if skipped is not None:
                fallback_result = skipped
            else:
                fallback_query, fallback_site_bias = _fallback_query_for_provider(
                    fallback_id,
                    query,
                    site_bias,
                )
                fallback_result = _provider_discover_full(
                    fallback,
                    fallback_query,
                    max_results=max_results,
                    site_bias=fallback_site_bias,
                    retrieval_profile=retrieval_profile,
                )
                _record_primary_outcome(fallback_id, fallback_result)
        else:
            fallback_query, fallback_site_bias = _fallback_query_for_provider(
                fallback_id,
                query,
                site_bias,
            )
            logger.warning(
                "[Discovery] primary=%s outcome=%s → fallback=%s query=%r",
                primary_id,
                (
                    primary_result.search_outcome.kind.value
                    if primary_result.search_outcome
                    else "unknown"
                ),
                fallback_id,
                fallback_query[:120],
            )
            fallback_result = _provider_discover_full(
                fallback,
                fallback_query,
                max_results=max_results,
                site_bias=fallback_site_bias,
                use_cache=fallback_id != PRIMARY_DISCOVERY_PROVIDER_ID,
                retrieval_profile=retrieval_profile,
            )

        if fallback_result.candidates:
            logger.info(
                "[Discovery] fallback=%s recovered candidates=%d tier=%s",
                fallback_id,
                len(fallback_result.candidates),
                route.privacy_tier,
            )
            merged = _merge_fallback_result(primary_result, fallback_result)
            return _tag_discovery_result(merged, privacy_tier=route.privacy_tier)

        logger.warning(
            "[Discovery] fallback=%s returned no candidates",
            fallback_id,
        )

    logger.warning(
        "[Discovery] all fallbacks exhausted; keeping primary failure (tier=%s)",
        route.privacy_tier,
    )
    return _tag_discovery_result(primary_result, privacy_tier=route.privacy_tier)


def discover(
    query: str,
    *,
    max_results: int = 5,
    site_bias: tuple[str, ...] | None = None,
    provider_id: str | None = None,
):
    """Discover candidate URLs via the default or named provider."""
    return list(
        discover_full(
            query,
            max_results=max_results,
            site_bias=site_bias,
            provider_id=provider_id,
        ).candidates
    )


def discover_full(
    query: str,
    *,
    max_results: int = 5,
    site_bias: tuple[str, ...] | None = None,
    provider_id: str | None = None,
    fallback: bool = True,
    retrieval_profile: str | None = None,
):
    """Discover candidates plus raw SERP rows and typed search outcome."""
    if provider_id:
        provider = get_discovery_provider(provider_id)
        if provider is None:
            raise ValueError(f"Unknown discovery provider: {provider_id}")
        pid = (provider_id or "").strip().lower()
        skipped = _skip_primary_ddg_result(pid)
        if skipped is not None:
            return skipped
        return _provider_discover_full(
            provider,
            query,
            max_results=max_results,
            site_bias=site_bias,
            retrieval_profile=retrieval_profile,
        )
    if fallback:
        return discover_full_with_fallback(
            query,
            max_results=max_results,
            site_bias=site_bias,
            retrieval_profile=retrieval_profile,
        )
    skipped = _skip_primary_ddg_result(PRIMARY_DISCOVERY_PROVIDER_ID)
    if skipped is not None:
        return skipped
    return _provider_discover_full(
        default_discovery_provider(),
        query,
        max_results=max_results,
        site_bias=site_bias,
        retrieval_profile=retrieval_profile,
    )


def _register_builtin_providers() -> None:
    from core.knowledge.discovery.brave_search import BraveSearchDiscovery
    from core.knowledge.discovery.duckduckgo import DuckDuckGoDiscovery
    from core.knowledge.discovery.searxng import SearXNGDiscovery
    from core.knowledge.discovery.wikipedia import WikipediaDiscovery

    register_discovery_provider(DuckDuckGoDiscovery())
    register_discovery_provider(BraveSearchDiscovery())
    register_discovery_provider(SearXNGDiscovery())
    register_discovery_provider(WikipediaDiscovery())


_register_builtin_providers()
