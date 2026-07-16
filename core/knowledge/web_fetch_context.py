"""Resolve profile- and composer-gated web fetch options."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.retrieval_profiles import get_profile_spec, normalize_profile_id
from core.knowledge.site_bias import RECIPE_DEFAULT_SITE_BIAS
from core.knowledge.types import RetrievalContext
from core.retrieval_relevance import DEFAULT_WEB_MIN_TOKEN_OVERLAP


@dataclass(frozen=True)
class WebFetchOptions:
    fetch_url_count: int
    site_bias: tuple[str, ...] | None = None
    composer_tool: str | None = None


@dataclass(frozen=True)
class WebRelevanceOptions:
    """Context-aware SERP relevance policy before optional page fetch."""

    apply_gate: bool = True
    min_token_ratio: float = DEFAULT_WEB_MIN_TOKEN_OVERLAP
    use_embedding_gate: bool = True
    mode: str = "strict"


def resolve_web_fetch_options(ctx: RetrievalContext) -> WebFetchOptions:
    """Merge retrieval profile defaults with composer pin overrides."""
    profile = get_profile_spec(normalize_profile_id(ctx.retrieval_profile))
    fetch_url_count = (
        ctx.fetch_url_count
        if ctx.fetch_url_count is not None
        else profile.fetch_url_count
    )
    site_bias = ctx.site_bias
    tool = (ctx.composer_tool or "").strip().lower() or None

    if tool in {"fetch", "recipe"}:
        fetch_url_count = max(int(fetch_url_count or 0), 1)
    if tool == "recipe" and not site_bias:
        site_bias = RECIPE_DEFAULT_SITE_BIAS

    return WebFetchOptions(
        fetch_url_count=max(0, int(fetch_url_count or 0)),
        site_bias=site_bias,
        composer_tool=tool,
    )


def resolve_web_relevance_options(
    ctx: RetrievalContext,
    fetch_options: WebFetchOptions,
) -> WebRelevanceOptions:
    """Choose SERP gate strictness from composer tool and fetch intent."""
    _ = ctx
    tool = (fetch_options.composer_tool or "").strip().lower()
    if tool == "recipe" and fetch_options.fetch_url_count >= 1:
        # Recipe pages are validated by RecipeExtractor after fetch, not SERP titles.
        return WebRelevanceOptions(apply_gate=False, mode="recipe_fetch_skip")
    if tool == "fetch" and fetch_options.fetch_url_count >= 1:
        return WebRelevanceOptions(
            apply_gate=True,
            min_token_ratio=0.08,
            use_embedding_gate=False,
            mode="fetch_permissive",
        )
    return WebRelevanceOptions()
