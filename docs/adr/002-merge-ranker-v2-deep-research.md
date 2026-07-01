# ADR 002: Merge Ranker v2 — weighted post-merge ranking for deep research

**Status:** Accepted  
**Date:** 2026-06-25  
**Deciders:** Qube maintainers (documented after Phase 6 manual QA + external architecture review)

## Context

Deep research merges sub-query bundles, dedupes by DOI/title/URL, then filters tangential hits before synthesis. Phase 5 evolved this path in slices:

| Phase 5 slice | Merge behavior |
|---------------|----------------|
| Slice 2 | Domain anchor tokens + optional embedding gate |
| Slice 4 | **Title-first anchor gate** + query-linked reject patterns + ACE drug-name sub-query angle |

Slice 4 improved topical precision on ACE/HF queries (chemo cardiotoxicity rejected; angiotensin trials surfaced) and passed live eval (`evaluate_deep_research.py --live --require-relevance --min-relevance-ok 3` → **3/3**).

During **Phase 6 manual QA** (case QA-6B-G, `@research ACE inhibitors heart failure mortality evidence`), deep research completed with `status=ok` but the merged bundle collapsed to **1 source** and `coverage: poor`. Trace diagnostics showed:

```
merged_sources_pre_filter:  8
merged_title_anchor_dropped: 7
merged_sources_post_filter: 1
```

Only one survivor had the domain anchor (`angiotensin`) in the **title**. On-topic PubMed/HF papers were dropped because the title-first gate required anchor tokens in the title, while many valid hits mention ACE inhibitors only in the abstract or use generic HF titles (e.g. GDMT gap reviews, finerenone narratives).

The scientific retrieval pipeline already uses **score-based ranking** (`core/knowledge/ranking/relevance.py`, MMR, authority weighting). The merge path used a **binary gate chain** (reject → title anchor → overlap/semantic), which:

1. Removed candidates before scoring could compare them.
2. Could not apply `min_keep` when the gate reduced the pool below the keep floor.
3. Diverged from the scoring model used elsewhere in the knowledge platform.

External review aligned on replacing tactical patches (e.g. `min_keep` alone, excerpt-aware anchor gates) with a **unified weighted ranker** that reuses scientific ranking patterns and keeps reject patterns as the only hard filter.

### Authoritative merge path (after this ADR)

```
Sub-query bundles
  → merge_evidence_bundles() — dedupe by DOI / title / URL
  → rank_merged_sources_for_query() — reject patterns (hard drop)
  → score each survivor (weighted features)
  → sort descending → top-K (default 8) with min_keep=2
  → apply_merged_relevance_gate() — rebuild bundle + diagnostics
  → synthesis / bibliography
```

Eval harness decompose defaults to **heuristic**; the app worker uses **LLM decompose** when a synthesis LLM is loaded (`decompose_mode=None` → `"llm"`).

## Decision

**Replace** the Phase 5 Slice 4 title-first anchor gate chain with **Merge Ranker v2**: score all non-rejected merged sources, rank, and keep top-K.

**Accept** the following design:

| Concern | v1 (Slice 4) | v2 (this ADR) |
|---------|--------------|---------------|
| Hard drops | Reject patterns + title anchor gate | **Reject title patterns only** |
| Domain anchors | Binary pass/fail on title | **Features**: `anchor_title`, `anchor_excerpt` |
| Entity overlap | Not used at merge | **Feature**, gated by title anchor when query anchors exist |
| Ordering | Gate survivors, implicit order | **Explicit weighted score → sort** |
| Keep policy | Implicit from gate survivors | **top-K=8**, **min_keep=2**, **min_score=0.14** |
| Telemetry | `merged_title_first_gate: true` | `merged_ranker_version: "2.0"`, `merged_title_first_gate: false` |

### Scoring module

`core/knowledge/ranking/merged_source.py` — `score_merged_source()` composes features in `[0, 1]`:

| Feature | Weight (default) | Notes |
|---------|------------------|-------|
| `lexical` | 0.22 | Token overlap vs query + combined title/excerpt |
| `semantic` | 0.18 | Embedding similarity when worker provides embedder; row-score fallback |
| `entity` | 0.18 | Query/source entity overlap; **× anchor_title** when query anchors exist |
| `anchor_title` | 0.14 | Domain token in title |
| `anchor_excerpt` | 0.08 | Domain token in excerpt (zeroed when title already matches) |
| `prior_relevance` | 0.10 | Adapter retrieval score |
| `authority` | 0.10 | Source authority score |

Entity gating prevents excerpt-only anchor matches from receiving full entity corroboration when the title does not signal the domain drug/class (regression found on statin primary-prevention queries).

### Merge orchestration

`core/knowledge/deep_research_merge.py` — `rank_merged_sources_for_query()` implements reject → score → sort → top-K. `filter_merged_sources_for_query()` delegates here for backward compatibility.

`core/knowledge/deep_research.py` — `apply_merged_relevance_gate()` **always rebuilds the bundle when rank order changes**, even when `dropped == 0` (bug fix: pre-v2 path returned the pre-rank bundle unchanged).

### Decompose modes (eval + worker)

`core/knowledge/deep_research_decompose.py` — `decompose_query(..., mode=heuristic|llm|hybrid)`.

| Surface | Default mode | Purpose |
|---------|--------------|---------|
| `DeepResearchWorker` | `llm` when synthesis LLM loaded | Production sub-query planning |
| `tools/evaluate_deep_research.py` | `heuristic` (`--decompose`) | Deterministic eval baseline; optional A/B vs LLM |

Diagnostics include `decompose_mode` alongside legacy `decompose_method`.

## Alternatives considered

### A. Raise `min_keep` without changing gates

**Rejected.** When the title anchor gate leaves one survivor, `min_keep` cannot recover dropped sources. Treats a symptom of ordering/gating, not the cause.

### B. Title + excerpt anchor gate (soften Slice 4)

**Rejected.** Still binary per source; excerpt matches inflate scores for generic cardiovascular papers that mention the drug class only in the abstract. Eval relevance checks **title tokens**; excerpt-only passes do not improve measured `relevance_ok`.

### C. Keep gates and add a secondary score sort

**Rejected.** Sources removed by gates never reach the ranker; inconsistent with the scientific pipeline’s score-first model and harder to tune than a single ranker.

### D. Merge Ranker v2 (weighted features + reject-only hard drops)

**Accepted.** Aligns deep-research merge with `core/knowledge/ranking/relevance.py` patterns; anchors and entity overlap become features; reject patterns stay as safety rails for known off-topic classes (takotsubo, chagas, chemo cardiotoxicity on ACE queries).

## Consequences

### Positive

- **QA-6B-G regression fixed:** ACE/HF merge retains **8 sources**, `coverage: excellent`, `relevance_ok: true` (was 1 source / poor).
- Live eval restored: `python3 tools/evaluate_deep_research.py --live --require-relevance --min-relevance-ok 3` → **3/3** `relevance_ok`.
- Rank order is deterministic and inspectable via `merged_top_feature_scores` in `retrieval_trace.relevance_diag`.
- Reject patterns unchanged — known off-topic titles still hard-dropped.
- Eval can compare decompose strategies via `--decompose heuristic|llm|hybrid` without changing worker defaults.

### Negative / constraints

- **Phase 5 Slice 4 title-first gate is superseded** for merge; traces will show `merged_title_anchor_dropped: 0` and `merged_title_first_gate: false`. Historical traces before this ADR used different semantics for those fields.
- Top-K may retain lower-scoring tangential sources (e.g. finerenone narrative on ACE queries) if the pool is small; synthesis and reject patterns mitigate but do not eliminate all noise.
- Entity feature requires title anchor when query anchors exist — sources with correct entity links but generic titles rely on lexical/semantic/anchor_excerpt instead of entity boost.
- Weight tuning is a ongoing concern; defaults were validated on the three-query deep-research corpus, not exhaustive clinical coverage.

### Implementation guardrails

- Do **not** reintroduce a binary title anchor gate ahead of scoring without a new ADR and eval re-baseline.
- `merged_ranker_version` must appear in merge diagnostics for trace compatibility checks.
- `apply_merged_relevance_gate()` must apply ranker sort order even when no sources are dropped.
- Eval harness stays **heuristic decompose by default** so CI/live baselines remain reproducible; LLM decompose is opt-in via `--decompose llm`.

## Validation

| Check | Result |
|-------|--------|
| `tests/test_merge_ranker_v2.py` | QA-6B-G-like fixture keeps ≥2 sources; reject pattern drops takotsubo; entity gating |
| `tests/test_deep_research.py` | Reorder when `dropped == 0`; takotsubo rejected |
| `tests/test_deep_research_relevance.py` | Updated for ranker diagnostics |
| Live eval 3/3 | ace_inhibitors_hf, sglt2_hf, statin_primary_prevention — all `relevance_ok` |
| Manual QA QA-6B-G | Merged source count ≥2, adequate+ coverage (re-run recommended after deploy) |

## Failure modes if decision is violated

| Failure | Severity | Example |
|---------|----------|---------|
| Title gate reinstated before rank | Critical | HF/GDMT papers dropped; merge collapses to 1 source |
| Rank order skipped when `dropped == 0` | High | Statin simvastatin paper ranked below excerpt-only hits; `relevance_ok` fails |
| Entity overlap without title gate | Medium | Generic CV cohort studies outrank on-topic drug trials |
| Removing reject patterns | Medium | Takotsubo/chagas reviews return on ACE/SGLT2 queries |
| Eval uses LLM decompose by default | Low | Non-reproducible CI/live baselines; false regressions |

## References

- [`core/knowledge/ranking/merged_source.py`](../../core/knowledge/ranking/merged_source.py) — feature weights, `score_merged_source()`
- [`core/knowledge/deep_research_merge.py`](../../core/knowledge/deep_research_merge.py) — `rank_merged_sources_for_query()`
- [`core/knowledge/deep_research.py`](../../core/knowledge/deep_research.py) — `apply_merged_relevance_gate()`, `run_deep_research()`
- [`core/knowledge/ranking/relevance.py`](../../core/knowledge/ranking/relevance.py) — scientific retrieval scoring (pattern source)
- [`tools/evaluate_deep_research.py`](../../tools/evaluate_deep_research.py) — live eval + `--decompose`
- [`eval/retrieval_corpus/v1_deep_research.json`](../../eval/retrieval_corpus/v1_deep_research.json) — relevance corpus
- [`docs/manual_qa_phase6_slice6_discipline_routing.md`](../manual_qa_phase6_slice6_discipline_routing.md) — QA-6B-G case
- [`docs/external_knowledge_platform_plan.md`](../external_knowledge_platform_plan.md) — Phase 5 merge history; Phase 6 prerequisites
