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

import json
import math
import re
from dataclasses import dataclass
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
QUERY_MODES: tuple[str, ...] = ("production", "standalone")

# Metrics where larger is better / smaller is better (used by baseline compare).
HIGHER_IS_BETTER: tuple[str, ...] = (
    "precision_at_k",
    "ndcg_at_5",
    "relevant_recall",
    "abstain_correct",
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


POOL_MODES: tuple[str, ...] = ("production", "full")


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


def fixture_pool(
    fixture: dict[str, Any],
    *,
    include_synthetic: bool = False,
    query_mode: str = "production",
    pool_mode: str = "production",
) -> list[dict[str, Any]]:
    """Candidate rows to rank (copies), in the order the adapters returned them.

    Rows captured by ``tools/capture_knowledge_fixture.py`` carry ``retrieved``
    entries (query variant, adapter, rank). For ``pool_mode="production"`` only
    rows within the fixture's ``production_depth`` for that adapter are kept —
    the pool today's pipeline would actually see; ``"full"`` keeps the whole
    capture (recall-widening experiments). Rows without ``retrieved`` (v1 or
    hand-built fixtures) are always in the pool. Synthetic rows are dropped
    unless ``include_synthetic``.
    """
    if pool_mode not in POOL_MODES:
        raise ValueError(f"pool_mode must be one of {POOL_MODES}")
    variant = pool_variant(fixture, query_mode)
    depth = dict(fixture.get("production_depth") or {})
    adapter_order = {name: i for i, name in enumerate(depth)}
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
            rank = int(hit.get("rank") or 0)
            if pool_mode == "production" and adapter in depth and rank >= int(depth[adapter]):
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
    # pipeline_scientific: score_rows → MMR → reliability.
    from core.knowledge.pipeline_scientific import _rank_candidates
    from core.knowledge.types import RetrievalBudget, RetrievalContext

    ctx = RetrievalContext(
        query=query,
        semantic_query=query,
        knowledge_service=PATH_SCIENTIFIC,
        query_vector=query_vector,
        embed_fn=embed_fn,
        budget=RetrievalBudget(max_results=budget),
    )
    kept, rejected = _rank_candidates(rows, ctx=ctx, max_results=budget)
    return RankerOutput(kept=list(kept), rejected=list(rejected))


def _rank_general_web(query, rows, budget, *, query_vector, embed_fn) -> RankerOutput:
    # pipeline_general_web (default "strict" gate): filter_web_results keeps SERP
    # order; every survivor is a candidate source (the balanced profile then
    # fetches the top survivor — fetch is not replayed offline).
    from core.retrieval_relevance import filter_web_results

    kept, _ = filter_web_results(
        query,
        rows,
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
            and is_allowlisted_url(str(r.get("url") or ""))
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


def _build_ce_ranker() -> RankerFn:
    """Cross-encoder relevance gate (``ce``), S4.

    * ``scientific_evidence``: the shipped S4 path,
      ``pipeline_scientific._rank_candidates_gated`` (CE order, ``keep`` gate,
      MMR among survivors, reliability), state from its ``SufficiencyResult``.
    * other paths: a preview of the S7/S8 gates with the same thresholds — score
      the pool, keep rows ``>= keep`` best-first up to the budget, state from
      ``assess_sufficiency``. Not production behaviour yet.

    Content-free / non-English questions fall back to the legacy ranking, as in
    production. Raises ``RankerUnavailable`` when the model is not on disk.
    ``token`` / ``embed`` keep measuring the legacy, reranker-free path.
    """
    from core.knowledge.ranking import rerank
    from core.knowledge.ranking.sufficiency import SCIENTIFIC_THRESHOLDS, assess_sufficiency

    reason = rerank.unavailable_reason()
    if reason is not None:
        raise RankerUnavailable(f"cross-encoder reranker unavailable ({reason})")

    def rank(query: str, rows: list[dict[str, Any]], budget: int, path: str = PATH_SCIENTIFIC) -> RankerOutput:
        key = _normalize_path(path)
        if key == PATH_SCIENTIFIC:
            from core.knowledge.pipeline_scientific import _rank_candidates_gated
            from core.knowledge.types import RetrievalBudget, RetrievalContext

            ctx = RetrievalContext(
                query=query,
                semantic_query=query,
                knowledge_service=PATH_SCIENTIFIC,
                budget=RetrievalBudget(max_results=budget),
            )
            ranking = _rank_candidates_gated(rows, ctx=ctx, max_results=budget, question=query)
            state = ranking.sufficiency.state.value
            return RankerOutput(kept=list(ranking.kept), rejected=list(ranking.rejected), state=state)
        pool = list(rows)
        result = rerank.rerank_rows(query, pool)
        if result is None:
            return _token_ranker(query, rows, budget, path)
        for row, score in zip(pool, result.scores):
            row["_rerank_score"] = float(score)
        suff = assess_sufficiency(result.scores, SCIENTIFIC_THRESHOLDS)
        kept = [pool[i] for i in suff.kept_indices][: max(0, budget)]
        kept_suff = assess_sufficiency([r["_rerank_score"] for r in kept], SCIENTIFIC_THRESHOLDS) if kept else suff
        return RankerOutput(kept=kept, rejected=_rest(pool, kept), state=kept_suff.state.value)

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


def fixture_ranking_query(fixture: dict[str, Any], *, query_mode: str = "production") -> str:
    """Query the ranker scores against.

    ``production`` — what today's pipeline would rank with (the cleaned turn
    text; composer tokens already stripped). ``standalone`` — the gold
    standalone question, i.e. what S5/S6 query resolution aims to produce.
    Falls back to the production query when no standalone question exists.
    """
    production = str(
        fixture.get("retrieval_query")
        or fixture.get("query_clean")
        or fixture.get("query_raw")
        or ""
    ).strip()
    if query_mode == "standalone":
        return str(fixture.get("standalone_question") or production).strip()
    return production


def evaluate_fixture(
    fixture: dict[str, Any],
    *,
    ranker: str | RankerFn = "token",
    query_mode: str = "production",
    include_synthetic: bool = False,
    pool_mode: str = "production",
    k: int = 3,
) -> dict[str, Any]:
    """Rank a fixture's pool and report graded ranking + abstention metrics."""
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
    )
    query = fixture_ranking_query(fixture, query_mode=query_mode)
    output = rank_fn(query, [dict(r) for r in pool], budget, _normalize_path(fixture.get("path")))
    kept = output.kept
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
            "abstain_precision": (tp / (tp + fp)) if (tp + fp) else None,
            "abstain_recall": (tp / (tp + fn)) if (tp + fn) else None,
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
