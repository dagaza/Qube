"""Offline knowledge-quality evaluation over replay fixtures.

Phase 0 started with a single hard-coded ANT check (``evaluate_ant_fixture``).
Stage S1 (see ``docs/knowledge_retrieval_quality_assessment.md`` §12) adds a
generic, fixture-driven evaluator with graded ranking and abstention metrics so
every later stage can be measured per category, not only on the ANT case.

Fixture schema (v1 is still read; v2 adds optional fields)::

    {
      "schema_version": 2,
      "id": "...",
      "path": "scientific_evidence",        # retrieval path / knowledge service
      "category": "sts_humanities",          # reporting bucket (§12.5 rule 7)
      "expect": "sufficient|partial|insufficient",
      "query_raw": "...", "query_clean": "...",
      "standalone_question": "...",          # gold standalone question
      "gold_subject": "...", "gold_facet": "...",
      "budget": 3,                           # sources the pipeline may keep
      "adapter_queries": {"openalex": "..."},
      "history": [...],
      "candidates": [
        {"title": ..., "label": "relevant|partial|irrelevant",
         "label_source": "human|model", "synthetic": false, ...}
      ]
    }

Metric conventions (pinned by ``tests/test_knowledge_quality_eval.py``):

* Graded gains: relevant = 2, partial = 1, irrelevant = 0.
* ``synthetic: true`` candidates are removed from the pool before ranking unless
  ``include_synthetic=True`` — they are not real retrieval output.
* Unlabelled candidates stay in the pool (they are real rows) but contribute no
  gain and are excluded from label-based rates.
* A metric that is undefined for a fixture is ``None`` and is skipped when
  aggregating (never coerced to 0 or 1).
"""

from __future__ import annotations

import contextvars
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from core.knowledge.bundle_builder import build_scientific_evidence_bundle
from core.knowledge.ranking.relevance import score_rows
from core.knowledge.ui_adapter import evidence_to_ui_source
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt
from core.prompt_history import strip_composer_tokens_from_history
from core.prompt_renderers import render_system_ok_messages

ROOT = Path(__file__).resolve().parents[2]

LABEL_GAINS: dict[str, int] = {"relevant": 2, "partial": 1, "irrelevant": 0}
EXPECT_STATES: tuple[str, ...] = ("sufficient", "partial", "insufficient")
LABEL_SOURCES: tuple[str, ...] = ("human", "model")
DEFAULT_FIXTURE_BUDGET = 3  # balanced profile max_results
# production — the query today's code builds from the turn and history;
# standalone — the gold standalone question (what S5/S6 aim for);
# captured — the query string recorded at capture time (S2–S4 baselines).
QUERY_MODES: tuple[str, ...] = ("production", "standalone", "captured")

# Metrics where larger is better / smaller is better (used by baseline compare).
HIGHER_IS_BETTER: tuple[str, ...] = (
    "precision_at_k",
    "ndcg_at_5",
    "relevant_recall",
    "abstain_correct",
    "subject_in_query",
)
LOWER_IS_BETTER: tuple[str, ...] = ("irrelevant_kept_rate",)


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def title_matches_reject(title: str, patterns: list[str] | tuple[str, ...]) -> bool:
    t = (title or "").lower()
    for pat in patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def title_has_expected_token(title: str, tokens: list[str] | tuple[str, ...]) -> bool:
    t = (title or "").lower()
    return any(tok.lower() in t for tok in tokens if tok)


# ---------------------------------------------------------------------------
# Labels, pool, schema
# ---------------------------------------------------------------------------


def candidate_gain(row: dict[str, Any]) -> int | None:
    """Graded gain for a labelled candidate, ``None`` when unlabelled/unknown."""
    label = str(row.get("label") or "").strip().lower()
    return LABEL_GAINS.get(label)


POOL_MODES: tuple[str, ...] = ("production", "widened", "full")


def widened_depth(fixture: dict[str, Any]) -> dict[str, int]:
    """Per-adapter depth the S8 pipeline requests when the relevance gate runs.

    General web: ``GATED_WEB_POOL`` rows of the same results page (capped by
    what the fixture captured, 8 in v1). Every other path keeps its
    ``production_depth``: S8 measured and rejected a deeper scholarly pool,
    and did not change trusted or finance depth. Crossref and the user's raw
    wording were not captured, so they are not replayed.
    """
    from core.knowledge.pipeline_general_web import GATED_WEB_POOL

    depth = {str(k): int(v) for k, v in (fixture.get("production_depth") or {}).items()}
    if _normalize_path(fixture.get("path")) == PATH_GENERAL_WEB:
        return {adapter: max(n, GATED_WEB_POOL) for adapter, n in depth.items()}
    return depth


def _fixture_variants(fixture: dict[str, Any]) -> set[str]:
    variants: set[str] = set((fixture.get("adapter_queries") or {}).keys())
    for row in fixture.get("candidates") or []:
        for hit in (row.get("retrieved") or []) if isinstance(row, dict) else []:
            if isinstance(hit, dict) and hit.get("query"):
                variants.add(str(hit["query"]))
    return variants


def pool_variant(fixture: dict[str, Any], query_mode: str = "production") -> str:
    """Capture variant whose rows form the pool for ``query_mode``.

    Falls back to ``production`` when the fixture has no standalone capture
    (mirrors ``fixture_ranking_query``'s fallback).
    """
    if query_mode == "standalone" and "standalone" in _fixture_variants(fixture):
        return "standalone"
    return "production"


def fixture_route(
    fixture: dict[str, Any], variant: str, route_arm: str | None
) -> tuple[str, ...] | None:
    """Adapter order of ``route_arm`` for ``variant``, ``None`` when not captured."""
    if not route_arm:
        return None
    routes = (fixture.get("routes") or {}).get(variant) or {}
    route = routes.get(route_arm)
    return tuple(str(a) for a in route) if route is not None else None


def fixture_has_routes(fixture: dict[str, Any], query_mode: str = "production") -> bool:
    """True when the fixture captured every S13 route arm for ``query_mode``."""
    from core.knowledge.scientific_adapters import ROUTE_ARMS

    routes = (fixture.get("routes") or {}).get(pool_variant(fixture, query_mode)) or {}
    return all(arm in routes for arm in ROUTE_ARMS)


def fixture_pool(
    fixture: dict[str, Any],
    *,
    include_synthetic: bool = False,
    query_mode: str = "production",
    pool_mode: str = "production",
    route_arm: str | None = None,
) -> list[dict[str, Any]]:
    """Candidate rows to rank (copies), in the order the adapters returned them.

    Rows captured by ``tools/capture_knowledge_fixture.py`` carry ``retrieved``
    entries (query variant, adapter, rank). For ``pool_mode="production"`` only
    rows within the fixture's ``production_depth`` for that adapter are kept —
    the pool today's pipeline would actually see; ``"full"`` keeps the whole
    capture (recall-widening experiments). Rows without ``retrieved`` (v1 or
    hand-built fixtures) are always in the pool. Synthetic rows are dropped
    unless ``include_synthetic``.

    ``route_arm`` (S13) selects the adapters one A/B arm would query, from the
    fixture's ``routes``. Without it, hits captured only for an arm
    (``route_added``) are left out, so the pool is the one first captured.
    """
    if pool_mode not in POOL_MODES:
        raise ValueError(f"pool_mode must be one of {POOL_MODES}")
    variant = pool_variant(fixture, query_mode)
    depth = (
        widened_depth(fixture)
        if pool_mode == "widened"
        else dict(fixture.get("production_depth") or {})
    )
    route = fixture_route(fixture, variant, route_arm)
    adapter_order = {name: i for i, name in enumerate(route if route is not None else depth)}
    picked: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for idx, row in enumerate(fixture.get("candidates") or []):
        if not isinstance(row, dict):
            continue
        if row.get("synthetic") and not include_synthetic:
            continue
        hits = row.get("retrieved")
        if not hits:
            # Hand-built / v1 rows (no capture metadata) come first, in file order.
            picked.append(((-1, 0, idx), dict(row)))
            continue
        best: tuple[int, int, int] | None = None
        for hit in hits:
            if not isinstance(hit, dict) or hit.get("query") != variant:
                continue
            adapter = str(hit.get("adapter") or "")
            if route is None:
                if hit.get("route_added"):
                    continue
            elif adapter not in route:
                continue
            rank = int(hit.get("rank") or 0)
            if pool_mode != "full" and adapter in depth and rank >= int(depth[adapter]):
                continue
            key = (adapter_order.get(adapter, len(adapter_order)), rank, idx)
            if best is None or key < best:
                best = key
        if best is not None:
            picked.append((best, dict(row)))
    picked.sort(key=lambda item: item[0])
    return [row for _, row in picked]


def derive_expect(pool: Sequence[dict[str, Any]]) -> str:
    """Expected sufficiency of a labelled pool.

    ``sufficient``: ≥ 2 relevant rows; ``partial``: ≥ 1 relevant or partial
    row; ``insufficient``: nothing useful (the pipeline should abstain).
    """
    relevant = sum(1 for r in pool if candidate_gain(r) == 2)
    useful = sum(1 for r in pool if (candidate_gain(r) or 0) >= 1)
    if relevant >= 2:
        return "sufficient"
    if useful >= 1:
        return "partial"
    return "insufficient"


def validate_fixture(fixture: dict[str, Any]) -> list[str]:
    """Return schema problems (empty list when the fixture is valid)."""
    errors: list[str] = []
    if not isinstance(fixture, dict):
        return ["fixture is not an object"]
    if not str(fixture.get("id") or "").strip():
        errors.append("missing id")
    version = fixture.get("schema_version", 1)
    if version not in (1, 2):
        errors.append(f"unsupported schema_version {version!r}")
    if not (fixture.get("query_clean") or fixture.get("query_raw")):
        errors.append("missing query_clean/query_raw")
    expect = fixture.get("expect")
    if expect is not None and expect not in EXPECT_STATES:
        errors.append(f"invalid expect {expect!r}")
    budget = fixture.get("budget")
    if budget is not None and (not isinstance(budget, int) or budget < 1):
        errors.append(f"invalid budget {budget!r}")
    candidates = fixture.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates must be a list")
        return errors
    degraded = fixture.get("capture_degraded")
    if degraded:
        errors.append(
            f"capture degraded for {degraded} (discovery fell back from the primary "
            "provider); re-capture before using as a baseline"
        )
    deferred = fixture.get("deferred_adapters")
    if deferred:
        errors.append(
            f"capture incomplete: adapters deferred {deferred}; fill them with "
            "tools/capture_knowledge_fixture.py --add-routes before using as a baseline"
        )
    seen: set[str] = set()
    for idx, row in enumerate(candidates):
        where = f"candidates[{idx}]"
        if not isinstance(row, dict):
            errors.append(f"{where} is not an object")
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            errors.append(f"{where} missing title")
        key = str(row.get("doi") or row.get("url") or title).strip().lower()
        if key and key in seen:
            errors.append(f"{where} duplicate candidate {key[:60]!r}")
        seen.add(key)
        label = row.get("label")
        if label is not None and str(label).lower() not in LABEL_GAINS:
            errors.append(f"{where} invalid label {label!r}")
        source = row.get("label_source")
        if source is not None and source not in LABEL_SOURCES:
            errors.append(f"{where} invalid label_source {source!r}")
    return errors


# ---------------------------------------------------------------------------
# Metrics (pure functions over ranked/kept rows)
# ---------------------------------------------------------------------------


def precision_at_k(ranked: Sequence[dict[str, Any]], k: int) -> float | None:
    """Share of the first ``min(k, len(ranked))`` labelled rows that are useful.

    Useful = relevant or partial. Divides by the number of rows actually
    returned (not ``k``) so a gate that keeps fewer, better rows is not
    penalised. ``None`` when nothing labelled was returned.
    """
    if k <= 0:
        return None
    top = [r for r in list(ranked)[:k] if candidate_gain(r) is not None]
    if not top:
        return None
    useful = sum(1 for r in top if (candidate_gain(r) or 0) >= 1)
    return useful / len(top)


def _dcg(gains: Iterable[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(
    ranked: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    k: int,
) -> float | None:
    """Graded nDCG@k of ``ranked`` against the ideal ordering of ``pool``.

    ``None`` when the pool has no relevant/partial rows (ideal DCG is 0):
    the case is an abstain case and is measured by ``abstain_correct``.
    An empty ``ranked`` list with a non-empty ideal scores 0.0.
    """
    if k <= 0:
        return None
    ideal_gains = sorted(
        (candidate_gain(r) or 0 for r in pool), reverse=True
    )[:k]
    idcg = _dcg(ideal_gains)
    if idcg <= 0:
        return None
    gains = [candidate_gain(r) or 0 for r in list(ranked)[:k]]
    return _dcg(gains) / idcg


def irrelevant_kept_rate(kept: Sequence[dict[str, Any]]) -> float | None:
    """Share of labelled kept rows labelled irrelevant; ``None`` if none kept."""
    labelled = [r for r in kept if candidate_gain(r) is not None]
    if not labelled:
        return None
    return sum(1 for r in labelled if candidate_gain(r) == 0) / len(labelled)


def relevant_recall(
    kept: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    *,
    budget: int,
) -> float | None:
    """Relevant rows kept / relevant rows the budget could have held.

    Denominator is ``min(relevant_in_pool, budget)`` so a pipeline limited to
    ``budget`` sources can reach 1.0. ``None`` when the pool has no relevant row.
    """
    relevant_in_pool = sum(1 for r in pool if candidate_gain(r) == 2)
    if relevant_in_pool == 0 or budget <= 0:
        return None
    kept_relevant = sum(1 for r in kept if candidate_gain(r) == 2)
    return min(1.0, kept_relevant / min(relevant_in_pool, budget))


def abstained(kept: Sequence[dict[str, Any]], state: str | None = None) -> bool:
    """True when the pipeline would present no evidence for the turn."""
    return not kept or (state or "").lower() in ("insufficient", "none")


def abstain_correct(expect: str | None, did_abstain: bool) -> bool | None:
    """Whether abstention matched the fixture's ``expect`` (``None`` if unset)."""
    if expect not in EXPECT_STATES:
        return None
    if expect == "insufficient":
        return did_abstain
    return not did_abstain


# ---------------------------------------------------------------------------
# Rankers — thin wrappers over each path's production ranking code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankerOutput:
    kept: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    state: str = "unscored"  # sufficiency state once S4 lands
    warnings: tuple[str, ...] = ()  # S7: recency warnings ("stale_sources", …)


# The "today" freshness is judged against (S7): the fixture's capture date, or
# that date shifted forward to replay the same pool as if it had aged.
_EVAL_TODAY: contextvars.ContextVar[date | None] = contextvars.ContextVar("kq_eval_today", default=None)

# Plan S13: ranking-side heuristics the scientific replay can switch off, one
# A/B arm each. Routing heuristics are pool-side (``fixture_pool(route_arm=…)``).
ABLATE_TRIAL_GROUNDING = "trial_grounding"
ABLATE_DISCIPLINE_SLOTS = "discipline_slots"
RANKING_ABLATIONS: tuple[str, ...] = (ABLATE_TRIAL_GROUNDING, ABLATE_DISCIPLINE_SLOTS)
_EVAL_ABLATE: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "kq_eval_ablate", default=frozenset()
)


def _scientific_heuristics(query: str) -> tuple[frozenset[str], str | None]:
    """Trial signals and slot discipline ``pipeline_scientific.run`` derives from the turn.

    Production detects both on the user's turn (``ctx.query``); the replay uses
    the ranking query, which is the same text for single-turn cases.
    """
    from core.knowledge.ranking.trial_grounding import extract_trial_signal
    from core.knowledge.scientific_discipline import detect_scientific_discipline

    ablate = _EVAL_ABLATE.get()
    trial = frozenset() if ABLATE_TRIAL_GROUNDING in ablate else extract_trial_signal(query)
    discipline = (
        None if ABLATE_DISCIPLINE_SLOTS in ablate else detect_scientific_discipline(query).discipline
    )
    return trial, discipline


def fixture_today(fixture: dict[str, Any], *, shift_days: int = 0) -> date | None:
    """Capture date of ``fixture`` (plus ``shift_days``), or ``None`` if unknown."""
    raw = str(fixture.get("captured_at") or "").strip()
    if not raw:
        return None
    try:
        day = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            day = date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return day + timedelta(days=int(shift_days))


# (query, pool rows, budget, retrieval path) -> RankerOutput
RankerFn = Callable[[str, list[dict[str, Any]], int, str], RankerOutput]

PATH_SCIENTIFIC = "scientific_evidence"
PATH_GENERAL_WEB = "general_web"
PATH_TRUSTED = "trusted_knowledge"
PATH_FINANCE = "finance_knowledge"
RANKED_PATHS: tuple[str, ...] = (PATH_SCIENTIFIC, PATH_GENERAL_WEB, PATH_TRUSTED, PATH_FINANCE)


class RankerUnavailable(RuntimeError):
    """Raised by ``get_fixture_ranker`` when a ranker cannot run here."""


def _normalize_path(path: str | None) -> str:
    key = (path or "").strip().lower()
    # v1 fixtures carry no path; they were all scientific (@evidence) replays.
    return key if key and key != "unspecified" else PATH_SCIENTIFIC


def _rest(rows: list[dict[str, Any]], kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept_ids = {id(r) for r in kept}
    return [r for r in rows if id(r) not in kept_ids]


def _rank_scientific(query, rows, budget, *, query_vector, embed_fn) -> RankerOutput:
    # pipeline_scientific (UNSCORED path): score_rows with the trial boost → MMR
    # → reliability, then the discipline slots over the whole pool.
    from core.knowledge.pipeline_scientific import _rank_candidates, apply_discipline_slots
    from core.knowledge.types import RetrievalBudget, RetrievalContext

    ctx = RetrievalContext(
        query=query,
        semantic_query=query,
        knowledge_service=PATH_SCIENTIFIC,
        query_vector=query_vector,
        embed_fn=embed_fn,
        budget=RetrievalBudget(max_results=budget),
    )
    trial, discipline = _scientific_heuristics(query)
    kept, rejected = _rank_candidates(rows, ctx=ctx, max_results=budget, trial_signals=trial)
    if discipline is not None:
        kept = apply_discipline_slots(
            kept, list(rows), discipline=discipline, query=query, max_results=budget
        )
    return RankerOutput(kept=list(kept), rejected=list(rejected))


def _rank_general_web(query, rows, budget, *, query_vector, embed_fn) -> RankerOutput:
    # pipeline_general_web (default "strict" gate): filter_web_results keeps SERP
    # order; every survivor is a candidate source (the balanced profile then
    # fetches the top survivor — fetch is not replayed offline).
    from core.knowledge.ranking.web_gate import drop_search_ads
    from core.retrieval_relevance import filter_web_results

    # S7: search ads are dropped before any gate (both paths).
    candidates, _ads = drop_search_ads(rows)
    kept, _ = filter_web_results(
        query,
        candidates,
        query_vector=query_vector,
        embed_text_fn=embed_fn,
        use_embedding_gate=True,
    )
    kept_urls = {str(r.get("url") or "") for r in kept}
    rejected = [r for r in rows if str(r.get("url") or "") not in kept_urls]
    return RankerOutput(kept=list(kept), rejected=rejected)


def _rank_trusted(query, rows, budget, *, query_vector, embed_fn) -> RankerOutput:
    # pipeline_trusted: Wikipedia rows (up to min(2, budget)) are kept with no
    # relevance check; allowlisted SERP rows are gated and appended to budget.
    from core.knowledge.adapters.wikipedia_api import ADAPTER_ID as WIKI_ADAPTER
    from core.knowledge.ranking.authority import is_allowlisted_url
    from core.retrieval_relevance import filter_web_results

    wiki = [r for r in rows if str(r.get("_adapter") or "") == WIKI_ADAPTER][: min(2, budget)]
    kept: list[dict[str, Any]] = list(wiki)
    if len(kept) < budget:
        seen = {str(r.get("url") or "").strip().lower() for r in kept if r.get("url")}
        allowlisted = [
            r
            for r in rows
            if str(r.get("_adapter") or "") != WIKI_ADAPTER
            and is_allowlisted_url(str(r.get("url") or ""), widened=False)
            and str(r.get("url") or "").strip().lower() not in seen
        ]
        if allowlisted:
            filtered, _ = filter_web_results(
                query,
                allowlisted,
                query_vector=query_vector,
                embed_text_fn=embed_fn,
                use_embedding_gate=True,
            )
            by_url = {str(r.get("url") or ""): r for r in allowlisted}
            for row in filtered:
                if len(kept) >= budget:
                    break
                kept.append(by_url.get(str(row.get("url") or ""), row))
    return RankerOutput(kept=kept, rejected=_rest(rows, kept))


def _rank_finance(query, rows, budget, *, query_vector, embed_fn) -> RankerOutput:
    # pipeline_finance: gate (no embeddings) only when rows exceed budget.
    from core.knowledge.finance_query_planner import plan_finance_query
    from core.retrieval_relevance import filter_web_results

    kept = list(rows)
    if len(kept) > budget:
        plan = plan_finance_query(query, semantic_query=query)
        filtered, _ = filter_web_results(
            plan.semantic_query,
            kept,
            query_vector=query_vector,
            embed_text_fn=embed_fn,
            use_embedding_gate=False,
        )
        by_url = {str(r.get("url") or ""): r for r in kept}
        kept = [by_url.get(str(r.get("url") or ""), r) for r in filtered[:budget]]
    return RankerOutput(kept=kept, rejected=_rest(rows, kept))


_PATH_RANKERS = {
    PATH_SCIENTIFIC: _rank_scientific,
    PATH_GENERAL_WEB: _rank_general_web,
    PATH_TRUSTED: _rank_trusted,
    PATH_FINANCE: _rank_finance,
}


def production_rank(
    query: str,
    rows: list[dict[str, Any]],
    budget: int,
    path: str | None,
    *,
    query_vector: Any = None,
    embed_fn: Callable[[str], Any] | None = None,
) -> RankerOutput:
    """Rank ``rows`` the way the production pipeline for ``path`` does today."""
    key = _normalize_path(path)
    fn = _PATH_RANKERS.get(key)
    if fn is None:
        raise RankerUnavailable(f"no production ranker replay for path {key!r}")
    return fn(query, rows, budget, query_vector=query_vector, embed_fn=embed_fn)


def _token_ranker(
    query: str, rows: list[dict[str, Any]], budget: int, path: str = PATH_SCIENTIFIC
) -> RankerOutput:
    return production_rank(query, rows, budget, path)


def _build_embed_ranker() -> RankerFn:
    """Production ranking with the locally installed Balanced embedding preset."""
    from core.bootstrap_search_models import embedding_preset_cached_on_disk
    from core.embedding_modes import get_mode_spec

    spec = get_mode_spec("balanced")
    if not embedding_preset_cached_on_disk(spec.mode_id):
        raise RankerUnavailable(
            f"embedding preset {spec.mode_id!r} not on disk"
        )
    try:
        from rag.backends.fastembed_backend import FastembedBackend

        backend = FastembedBackend(spec)
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RankerUnavailable(f"embedding preset failed to load: {exc}") from exc

    cache: dict[str, Any] = {}

    def embed(text: str) -> Any:
        # Production embeds both queries and docs with embed_query
        # (workers/intent_router.EmbeddingCache.get_embedding).
        if text not in cache:
            cache[text] = backend.embed_query(text)
        return cache[text]

    def rank(
        query: str, rows: list[dict[str, Any]], budget: int, path: str = PATH_SCIENTIFIC
    ) -> RankerOutput:
        return production_rank(
            query, rows, budget, path, query_vector=embed(query), embed_fn=embed
        )

    return rank


def _ce_rank_non_scientific(
    query: str, rows: list[dict[str, Any]], budget: int, path: str
) -> RankerOutput | None:
    """Replay the S7 gates of ``pipeline_general_web`` / ``pipeline_trusted`` /
    ``pipeline_finance`` (``None`` when the reranker cannot run → legacy)."""
    from core.knowledge.ranking.web_gate import (
        FINANCE_THRESHOLDS,
        SERP_RANK_PRIOR,
        TRUSTED_THRESHOLDS,
        WEB_THRESHOLDS,
        drop_search_ads,
        gate_rows,
    )

    today = _EVAL_TODAY.get()
    if path == PATH_GENERAL_WEB:
        # As in production (S8): up to ``budget`` rows are presented and judged;
        # the other survivors are only fetch candidates.
        pool, _ads = drop_search_ads(rows)
        gate = gate_rows(
            query, pool, budget=budget, thresholds=WEB_THRESHOLDS, today=today, rank_prior=SERP_RANK_PRIOR
        )
    elif path == PATH_TRUSTED:
        from core.knowledge.adapters.wikipedia_api import ADAPTER_ID as WIKI_ADAPTER
        from core.knowledge.ranking.authority import is_allowlisted_url

        wiki = [r for r in rows if str(r.get("_adapter") or "") == WIKI_ADAPTER][: min(2, budget)]
        for r in wiki:
            r.setdefault("_wiki_source", True)
        gate = gate_rows(query, wiki, budget=budget, thresholds=TRUSTED_THRESHOLDS, today=today)
        if gate is not None and len(gate.kept) < budget:
            wiki_urls = {str(r.get("url") or "").strip().lower() for r in wiki if r.get("url")}
            allowlisted, _ads = drop_search_ads(
                [
                    r
                    for r in rows
                    if str(r.get("_adapter") or "") != WIKI_ADAPTER
                    and is_allowlisted_url(str(r.get("url") or ""))
                    and str(r.get("url") or "").strip().lower() not in wiki_urls
                ]
            )
            if allowlisted:
                gate = gate_rows(
                    query, [*wiki, *allowlisted], budget=budget, thresholds=TRUSTED_THRESHOLDS, today=today
                )
    elif path == PATH_FINANCE:
        from core.knowledge.finance_query_planner import plan_finance_query

        plan = plan_finance_query(query, semantic_query=query)
        gate = gate_rows(
            plan.semantic_query or query, rows, budget=budget, thresholds=FINANCE_THRESHOLDS, today=today
        )
    else:
        return None
    if gate is None:
        return None
    return RankerOutput(
        kept=list(gate.kept),
        rejected=_rest(rows, gate.kept),
        state=gate.sufficiency.state.value,
        warnings=tuple(gate.warnings),
    )


def _build_ce_ranker() -> RankerFn:
    """Cross-encoder relevance gate (``ce``), S4.

    * ``scientific_evidence``: the shipped S4 path,
      ``pipeline_scientific._rank_candidates_gated`` (CE order, ``keep`` gate,
      MMR among survivors, reliability), state from its ``SufficiencyResult``.
    * ``general_web`` / ``trusted_knowledge`` / ``finance_knowledge``: the
      shipped S7 gates (``_ce_rank_non_scientific``), with freshness judged
      against the fixture's capture date.

    Content-free / non-English questions fall back to the legacy ranking, as in
    production. Raises ``RankerUnavailable`` when the model is not on disk.
    ``token`` / ``embed`` keep measuring the legacy, reranker-free path.
    """
    from core.knowledge.ranking import rerank

    reason = rerank.unavailable_reason()
    if reason is not None:
        raise RankerUnavailable(f"cross-encoder reranker unavailable ({reason})")

    def rank(query: str, rows: list[dict[str, Any]], budget: int, path: str = PATH_SCIENTIFIC) -> RankerOutput:
        key = _normalize_path(path)
        if key == PATH_SCIENTIFIC:
            from core.knowledge.pipeline_scientific import (
                SCIENTIFIC_THRESHOLDS,
                _final_sufficiency,
                _rank_candidates_gated,
                apply_discipline_slots,
            )
            from core.knowledge.types import RetrievalBudget, RetrievalContext

            ctx = RetrievalContext(
                query=query,
                semantic_query=query,
                knowledge_service=PATH_SCIENTIFIC,
                budget=RetrievalBudget(max_results=budget),
            )
            # Mirrors ``pipeline_scientific.run``: trial boost in the pre-order,
            # discipline slots over the gate's survivors, state recomputed.
            trial, discipline = _scientific_heuristics(query)
            ranking = _rank_candidates_gated(
                rows, ctx=ctx, max_results=budget, trial_signals=trial, question=query
            )
            kept = list(ranking.kept)
            if discipline is not None:
                kept = apply_discipline_slots(
                    kept, ranking.slot_pool, discipline=discipline, query=query, max_results=budget
                )
            sufficiency = ranking.sufficiency
            if ranking.scored:
                sufficiency = _final_sufficiency(kept, ranking.sufficiency, SCIENTIFIC_THRESHOLDS)
            return RankerOutput(
                kept=kept, rejected=list(ranking.rejected), state=sufficiency.state.value
            )
        output = _ce_rank_non_scientific(query, rows, budget, key)
        return output if output is not None else _token_ranker(query, rows, budget, path)

    return rank


RANKER_NAMES: tuple[str, ...] = ("token", "embed", "ce")


def get_fixture_ranker(name: str) -> RankerFn:
    """Return a ranker by name or raise ``RankerUnavailable`` with the reason."""
    key = (name or "").strip().lower()
    if key == "token":
        return _token_ranker
    if key == "embed":
        return _build_embed_ranker()
    if key == "ce":
        return _build_ce_ranker()
    raise RankerUnavailable(f"unknown ranker {name!r}; choose from {RANKER_NAMES}")


# ---------------------------------------------------------------------------
# Generic fixture evaluation
# ---------------------------------------------------------------------------


def _captured_query(fixture: dict[str, Any]) -> str:
    return str(
        fixture.get("retrieval_query")
        or fixture.get("query_clean")
        or fixture.get("query_raw")
        or ""
    ).strip()


def resolved_production_query(fixture: dict[str, Any]) -> str:
    """The ranking query the current code would build for this turn.

    Mirrors the worker: discourse resolution of the turn against its history
    (``build_discourse_resolution``; acronym expansion since S5) gives the web
    text, then the path's own cleanup — the scientific planner's semantic
    query, or request-frame stripping for general web / trusted.
    """
    turn = str(fixture.get("query_clean") or "").strip()
    if not turn:
        return _captured_query(fixture)
    from core.query_resolution_evaluation import build_discourse_resolution

    _fu, _state, resolved = build_discourse_resolution(turn, list(fixture.get("history") or []))
    query = (resolved.web_text or turn).strip()
    path = _normalize_path(fixture.get("path"))
    if path == "scientific_evidence":
        from core.knowledge.scientific_query_planner import plan_scientific_query

        return plan_scientific_query(query, semantic_query=query).semantic_query or query
    if path in ("general_web", "trusted_knowledge"):
        from core.retrieval_query_cleanup import strip_request_frames

        return strip_request_frames(query)
    return query


def fixture_ranking_query(fixture: dict[str, Any], *, query_mode: str = "production") -> str:
    """Query the ranker scores against (see ``QUERY_MODES``).

    ``standalone`` falls back to the production query when no standalone
    question exists.
    """
    if query_mode == "captured":
        return _captured_query(fixture)
    production = resolved_production_query(fixture)
    if query_mode == "standalone":
        return str(fixture.get("standalone_question") or production).strip()
    return production


def subject_in_query(fixture: dict[str, Any], query: str) -> bool | None:
    """Multi-turn fixtures only: does the query name the gold subject?"""
    subject = str(fixture.get("gold_subject") or "").strip().lower()
    if not subject or not fixture.get("history"):
        return None
    return subject in (query or "").lower()


def evaluate_fixture(
    fixture: dict[str, Any],
    *,
    ranker: str | RankerFn = "token",
    query_mode: str = "production",
    include_synthetic: bool = False,
    pool_mode: str = "production",
    k: int = 3,
    recency_shift_days: int = 0,
    ablate: Iterable[str] = (),
    route_arm: str | None = None,
) -> dict[str, Any]:
    """Rank a fixture's pool and report graded ranking + abstention metrics.

    ``ablate`` switches ranking heuristics off (``RANKING_ABLATIONS``);
    ``route_arm`` picks the pool of one S13 routing arm (``fixture_pool``).

    ``recency_shift_days`` replays the pool as if it were that many days older
    (S7: a fresh capture becomes a stale one without a new capture).
    """
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of {QUERY_MODES}")
    rank_fn = get_fixture_ranker(ranker) if isinstance(ranker, str) else ranker
    ranker_name = ranker if isinstance(ranker, str) else getattr(ranker, "__name__", "custom")

    budget = int(fixture.get("budget") or DEFAULT_FIXTURE_BUDGET)
    pool = fixture_pool(
        fixture,
        include_synthetic=include_synthetic,
        query_mode=query_mode,
        pool_mode=pool_mode,
        route_arm=route_arm,
    )
    query = fixture_ranking_query(fixture, query_mode=query_mode)
    token = _EVAL_TODAY.set(fixture_today(fixture, shift_days=recency_shift_days))
    ablate_token = _EVAL_ABLATE.set(frozenset(ablate))
    try:
        output = rank_fn(query, [dict(r) for r in pool], budget, _normalize_path(fixture.get("path")))
    finally:
        _EVAL_ABLATE.reset(ablate_token)
        _EVAL_TODAY.reset(token)
    kept = output.kept
    recency = _recency_report(query, kept, output, fixture, recency_shift_days)
    did_abstain = abstained(kept, output.state)
    # An explicit fixture ``expect`` is for single-pool fixtures; captured
    # fixtures derive it from the labels of the pool being evaluated.
    # Empty pool -> nothing to cite (should abstain); an unlabelled pool gives
    # no verdict rather than a fake "insufficient".
    if fixture.get("expect"):
        expect, expect_source = fixture.get("expect"), "fixture"
    elif not pool:
        expect, expect_source = "insufficient", "derived"
    elif any(candidate_gain(r) is not None for r in pool):
        expect, expect_source = derive_expect(pool), "derived"
    else:
        expect, expect_source = None, "unlabelled"

    return {
        "fixture_id": fixture.get("id"),
        "path": fixture.get("path") or "unspecified",
        "category": fixture.get("category") or "uncategorized",
        "expect": expect,
        "expect_source": expect_source,
        "pool_mode": pool_mode,
        "ranker": ranker_name,
        "query_mode": query_mode,
        "query": query,
        "budget": budget,
        "pool_size": len(pool),
        "synthetic_excluded": sum(
            1 for r in fixture.get("candidates") or [] if isinstance(r, dict) and r.get("synthetic")
        )
        if not include_synthetic
        else 0,
        "unlabelled_in_pool": sum(1 for r in pool if candidate_gain(r) is None),
        "labelled_in_pool": sum(1 for r in pool if candidate_gain(r) is not None),
        "relevant_in_pool": sum(1 for r in pool if candidate_gain(r) == 2),
        "kept_count": len(kept),
        "kept_titles": [str(r.get("title") or "")[:120] for r in kept],
        "state": output.state,
        "abstained": did_abstain,
        "precision_at_k": precision_at_k(kept, k),
        "k": k,
        "ndcg_at_5": ndcg_at_k(kept, pool, 5),
        "irrelevant_kept_rate": irrelevant_kept_rate(kept),
        "relevant_recall": relevant_recall(kept, pool, budget=budget),
        "abstain_correct": abstain_correct(expect, did_abstain),
        "subject_in_query": subject_in_query(fixture, query),
        "warnings": list(output.warnings),
        **recency,
    }


def _recency_report(
    query: str,
    kept: Sequence[dict[str, Any]],
    output: RankerOutput,
    fixture: dict[str, Any],
    shift_days: int,
) -> dict[str, Any]:
    """S7 acceptance fields for turns that ask for current information.

    ``stale_only``: something is kept, nothing kept is fresh, and some kept row
    is dated outside the window. ``stale_marked``: such a turn is ``partial``
    (or abstains) and carries ``stale_sources``. ``stale_presented_sufficient``:
    such a turn is reported ``sufficient`` (must never happen).
    """
    from core.retrieval_query_cleanup import recency_need

    need = recency_need(query, today=fixture_today(fixture, shift_days=shift_days))
    if need is None:
        return {"recency_need": None}
    statuses = [r.get("_freshness") for r in kept]
    stale_only = bool(kept) and "fresh" not in statuses and "stale" in statuses
    scored = output.state != "unscored"
    return {
        "recency_need": need.window_days,
        "recency_shift_days": shift_days,
        "kept_freshness": statuses,
        "stale_only": stale_only if scored else None,
        "stale_marked": (
            (output.state in ("partial", "insufficient", "none") and "stale_sources" in output.warnings)
            if stale_only and scored
            else None
        ),
        "stale_presented_sufficient": (stale_only and output.state == "sufficient") if scored else None,
    }


def _mean(values: Iterable[float | bool | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def aggregate_reports(
    reports: Sequence[dict[str, Any]],
    *,
    group_by: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Mean of each metric over fixtures, overall or per ``group_by`` key.

    Also reports abstention precision/recall treating ``expect=insufficient``
    as the positive class.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rep in reports:
        key = "all" if group_by is None else str(rep.get(group_by) or "unspecified")
        groups.setdefault(key, []).append(rep)

    out: dict[str, dict[str, Any]] = {}
    for key, reps in sorted(groups.items()):
        tp = sum(1 for r in reps if r.get("expect") == "insufficient" and r.get("abstained"))
        fp = sum(1 for r in reps if r.get("expect") in ("sufficient", "partial") and r.get("abstained"))
        fn = sum(1 for r in reps if r.get("expect") == "insufficient" and not r.get("abstained"))
        out[key] = {
            "fixtures": len(reps),
            "precision_at_k": _mean(r.get("precision_at_k") for r in reps),
            "ndcg_at_5": _mean(r.get("ndcg_at_5") for r in reps),
            "irrelevant_kept_rate": _mean(r.get("irrelevant_kept_rate") for r in reps),
            "relevant_recall": _mean(r.get("relevant_recall") for r in reps),
            "abstain_correct": _mean(r.get("abstain_correct") for r in reps),
            "subject_in_query": _mean(r.get("subject_in_query") for r in reps),
            "abstain_precision": (tp / (tp + fp)) if (tp + fp) else None,
            "abstain_recall": (tp / (tp + fn)) if (tp + fn) else None,
            # S7 (R9): turns that ask for current information.
            "recency_turns": sum(1 for r in reps if r.get("recency_need")),
            "stale_only_turns": sum(1 for r in reps if r.get("stale_only")),
            "stale_marked": _mean(r.get("stale_marked") for r in reps),
            "stale_presented_sufficient": sum(1 for r in reps if r.get("stale_presented_sufficient")),
        }
    return out


def compare_to_baseline(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    min_drop: float = 0.05,
) -> list[dict[str, Any]]:
    """Per-group regressions vs a baseline aggregate (§12.5 rule 7).

    A regression is a move in the bad direction larger than
    ``max(min_drop, 1 / fixtures_in_group)`` — i.e. more than 1 fixture or
    5 percentage points, whichever is larger. Groups or metrics missing on
    either side are skipped (not counted as regressions).
    """
    regressions: list[dict[str, Any]] = []
    for group, cur in current.items():
        base = baseline.get(group)
        if not base:
            continue
        n = max(1, int(cur.get("fixtures") or 1))
        tolerance = max(min_drop, 1.0 / n)
        for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
            c, b = cur.get(metric), base.get(metric)
            if c is None or b is None:
                continue
            delta = float(c) - float(b)
            worse = -delta if metric in HIGHER_IS_BETTER else delta
            if worse > tolerance:
                regressions.append(
                    {
                        "group": group,
                        "metric": metric,
                        "baseline": b,
                        "current": c,
                        "tolerance": round(tolerance, 4),
                    }
                )
    return regressions


# ---------------------------------------------------------------------------
# Phase 0/1 prompt-contract checks (scientific / @evidence fixtures)
# ---------------------------------------------------------------------------


def score_fixture_candidates(
    fixture: dict[str, Any],
    *,
    query: str | None = None,
    include_synthetic: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the scientific relevance scorer on fixture candidates (token path)."""
    q = (query or fixture.get("query_clean") or fixture.get("query_raw") or "").strip()
    rows = fixture_pool(fixture, include_synthetic=include_synthetic)
    return score_rows(rows, query=q, min_score=0.12)


def build_fixture_bundle(
    fixture: dict[str, Any],
    *,
    kept: list[dict[str, Any]] | None = None,
) -> Any:
    query = (fixture.get("query_clean") or fixture.get("query_raw") or "").strip()
    rows = kept
    if rows is None:
        kept_rows, _ = score_fixture_candidates(fixture, query=query)
        rows = kept_rows
    return build_scientific_evidence_bundle(
        query_raw=str(fixture.get("query_raw") or query),
        query_resolved=query,
        kept_rows=rows,
        rejected_count=0,
        latency_ms=1.0,
        adapter_calls=tuple(
            sorted({str(r.get("_adapter") or "") for r in rows if r.get("_adapter")})
        ),
        medical_query=False,
    )


def prompt_allows_abstention(system_text: str, user_turn_text: str = "") -> bool:
    blob = f"{system_text}\n{user_turn_text}".lower()
    markers = (
        "if none of the sources are relevant",
        "if the retrieved sources do not address",
        "if none of the sources address",
        "say so plainly",
        "say so clearly",
    )
    return any(m in blob for m in markers)


def evaluate_prompt_contract(
    fixture: dict[str, Any],
    *,
    kept: list[dict[str, Any]] | None = None,
    include_synthetic: bool = False,
) -> dict[str, Any]:
    """Phase 1 prompt/evidence-pack contract flags for a scientific fixture."""
    history = list(fixture.get("history") or [])
    history.append(
        {
            "role": "user",
            "content": str(fixture.get("query_raw") or fixture.get("query_clean") or ""),
        }
    )
    cleaned = strip_composer_tokens_from_history(history)
    last_user = next(
        (
            str(m.get("content") or "")
            for m in reversed(cleaned)
            if str(m.get("role") or "").lower() == "user"
        ),
        "",
    )
    token_stripped = "@[tool:" not in last_user and not last_user.lstrip().startswith(
        "@evidence"
    )

    if kept is None:
        kept, _ = score_fixture_candidates(fixture, include_synthetic=include_synthetic)
    bundle = build_fixture_bundle(fixture, kept=kept)
    ui_sources = [
        evidence_to_ui_source(obj, ui_id=i)
        for i, obj in enumerate(bundle.sources, start=1)
    ]
    has_metadata = any(
        src.get("venue") or src.get("publication_date") or src.get("preprint") is not None
        for src in ui_sources
    )
    has_fuller_body = any(
        len(str(src.get("content") or "")) > 200 for src in ui_sources
    )

    blocks = build_prompt_blocks(
        execution_route="WEB",
        explicit_remember_active=False,
        has_retrieval_sources=True,
        retrieval_source_count=len(ui_sources),
        web_hit_count=len(ui_sources),
        knowledge_service="scientific_evidence",
        conversation_history=cleaned,
        retrieval_context="--- [1]: Example ---\nBody.",
        current_date="2026-11-23",
        engine_mode="external",
    )
    system = compose_system_prompt(blocks)
    messages = render_system_ok_messages(blocks)
    user_turn = messages[-1]["content"] if messages else ""

    return {
        "composer_token_stripped": token_stripped,
        "prompt_allows_abstention": prompt_allows_abstention(system, user_turn),
        "scientific_persona": "Scientific literature results" in system,
        "date_injected": "Today's date is 2026-11-23" in system,
        "evidence_pack_has_metadata": has_metadata,
        "evidence_pack_fuller_body": has_fuller_body,
        "coverage": bundle.coverage,
        "confidence": round(float(bundle.confidence), 4),
        "freshness_populated": any(
            getattr(s, "freshness_score", None) is not None for s in bundle.sources
        ),
    }


CONTRACT_FLAGS: tuple[str, ...] = (
    "composer_token_stripped",
    "prompt_allows_abstention",
    "scientific_persona",
    "date_injected",
    "evidence_pack_has_metadata",
    "freshness_populated",
)


def evaluate_ant_fixture(
    fixture: dict[str, Any] | None = None,
    *,
    include_synthetic: bool = False,
) -> dict[str, Any]:
    """Baseline checks for the recorded ANT/@evidence failure mode.

    Kept for the Phase 0/1 regression suite. Since S1 the synthetic positive
    candidate is excluded by default (it was never real retrieval output).
    """
    if fixture is None:
        fixture = load_json(
            "eval/fixtures/knowledge_quality/ant_ai_candidates.json"
        )

    kept, rejected = score_fixture_candidates(
        fixture, include_synthetic=include_synthetic
    )
    pool = fixture_pool(fixture, include_synthetic=include_synthetic)
    reject_patterns = list(fixture.get("reject_title_patterns") or [])
    kept_rejects = [
        r for r in kept if title_matches_reject(str(r.get("title") or ""), reject_patterns)
    ]
    relevant_kept = [r for r in kept if str(r.get("label") or "") == "relevant"]

    contract = evaluate_prompt_contract(fixture, kept=kept)
    return {
        "fixture_id": fixture.get("id"),
        **contract,
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "kept_reject_title_count": len(kept_rejects),
        "relevant_in_pool": any(str(r.get("label") or "") == "relevant" for r in pool),
        "relevant_kept_count": len(relevant_kept),
        # Token scorer still keeps irrelevant ANT/AI puns (gap until S3/S4).
        "phase0_known_gap_token_ranker_keeps_irrelevant": len(kept_rejects) > 0,
    }
