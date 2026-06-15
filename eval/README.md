# Router evaluation framework

Offline harness for regression-testing `CognitiveRouterV4` routing decisions against a labeled corpus.

## Corpus format

File: JSON with schema `qube.router_corpus.v1` (see `router_corpus.schema.json`).

```json
{
  "schema": "qube.router_corpus.v1",
  "version": 1,
  "description": "Baseline router evaluation corpus",
  "cases": [
    {
      "id": "gk_001",
      "prompt": "Why is the sky blue?",
      "expected_route": "none",
      "category": "general_knowledge",
      "notes": "optional reviewer notes",
      "history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "flags": {
        "internet_enabled": true
      }
    }
  ]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Stable case identifier |
| `prompt` | yes | User utterance |
| `expected_route` | yes | `none`, `memory`, `rag`, `web`, `hybrid` (`chat` → `none`) |
| `category` | yes | Reporting bucket |
| `notes` | no | Human-readable intent |
| `history` | no | Prior turns for discourse / follow-up simulation |
| `flags` | no | Per-case overrides (`internet_enabled`, etc.) |

Baseline corpus: `router_corpus/v1_baseline.json` (100 prompts).

## Fixture library (automated RAG + memory)

Seeded content lives under `eval/fixtures/`:

| Path | Purpose |
|------|---------|
| `fixtures/library/*.md` | Synthetic documents aligned with `rag_*` corpus prompts |
| `fixtures/memories.json` | Synthetic memory rows aligned with `mem_*` corpus prompts |

Seed into an isolated LanceDB (default `eval/.lancedb`, gitignored):

```bash
# Index fixtures only
venv/bin/python tools/seed_router_eval_library.py

# One-shot: seed + evaluate with retrieval against fixtures
venv/bin/python tools/evaluate_router.py --eval-fixtures --run-id fixtures_smoke

# Custom LanceDB path
venv/bin/python tools/evaluate_router.py \
  --lancedb-dir eval/.lancedb \
  --seed-eval-library \
  --with-retrieval
```

`--force-seed` re-indexes fixtures (only allowed under `eval/` scope).

## Runner

```bash
# Substring-only router (no embedder GGUF required)
python3 tools/evaluate_router.py --no-embeddings

# Full Tier-2 centroids (requires venv + ~/.qube/models/embedding/nomic-embed-text-v1.5.Q4_K_M.gguf)
venv/bin/python tools/evaluate_router.py

# With LanceDB retrieval hit counts (user DB: ~/.qube/data/lancedb)
venv/bin/python tools/evaluate_router.py --with-retrieval

# Automated fixture-backed retrieval eval (no manual library setup)
venv/bin/python tools/evaluate_router.py --eval-fixtures

# Regression compare vs saved run
python3 tools/evaluate_router.py --no-embeddings \
  --baseline eval/runs/baseline/run.json \
  --fail-on-regression
```

Artifacts per run (under `eval/runs/<run_id>/`):

- `results.csv` — per-case metrics
- `run.json` — full run + summary (schema `qube.router_eval_run.v1`)

### Eval artifacts policy

**What belongs in git**

| Path | Why |
|------|-----|
| `router_corpus/` | Labeled test inputs — source of truth |
| `fixtures/` | Seeded RAG/memory content for `--eval-fixtures` |
| `eval/runs/shadow_policy_baseline_v1/` | **Canonical regression baseline** (full-stack shadow eval) |

**What stays local** (gitignored under `eval/runs/*`)

- Smoke runs, experiments, and re-runs you create locally
- Perturbation/hysteresis/canonicalization JSON exports from ad-hoc `--run-id` values

Forks and contributors should **regenerate** local runs rather than clone every
experiment artifact. Compare against the committed baseline or your own saved
`run.json` via `--baseline eval/runs/shadow_policy_baseline_v1/run.json`.

**Reproduce the canonical baseline**

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --with-retrieval \
  --route-perturbation-analysis \
  --shadow-retrieval-policy-analysis \
  --retrieval-propensity-analysis \
  --simulate-hysteresis \
  --canonicalization-analysis \
  --continuous-pilot-routing \
  --continuous-arch-validation \
  --report \
  --run-id shadow_policy_baseline_v1
```

Optional offline 2D frontier (after perturbation artifacts exist):

```bash
venv/bin/python -c "
from pathlib import Path
from eval.shadow_retrieval_frontier import run_frontier_from_run_dir, export_frontier_json
d = Path('eval/runs/shadow_policy_baseline_v1')
export_frontier_json(d / 'frontier_2d.json', run_frontier_from_run_dir(d))
"
```

**When to update the committed baseline:** after intentional router or shadow-policy
changes that should shift regression expectations — not after every local smoke test.

## Query resolution evaluation (PR5)

Offline harness for discourse query strings and fixture-based web retrieval quality.

### Corpus

File: `eval/router_corpus/query_resolution_v1.json` (schema: `qube.query_resolution_corpus.v1`).

Each case may include an `expect` block:

| Field | Meaning |
|-------|---------|
| `inference_contains` / `inference_not_contains` | Substrings required/forbidden on inference text |
| `web_contains` / `web_not_contains` | Substrings required/forbidden on web search text |
| `routing_contains` / `retrieval_contains` | Routing and memory/RAG query strings |
| `web_fixture_id` | Replay offline DuckDuckGo HTML from `eval/fixtures/web/<id>.html` |
| `min_web_hits` | Minimum snippets passing the production relevance gate |

### Web fixtures

Recorded HTML under `eval/fixtures/web/` is parsed with `mcp.internet_tool.parse_ddg_html_results`
and filtered with `core.retrieval_relevance.filter_web_results` (same gate as production).

### Runner

```bash
# Lexical web gate only (no embedder GGUF required)
python3 tools/evaluate_query_resolution.py --no-embeddings

# Full embedding gate (requires venv + nomic embedder)
venv/bin/python tools/evaluate_query_resolution.py

# Fail CI when any case regresses
python3 tools/evaluate_query_resolution.py --no-embeddings --fail-on-regression
```

Artifacts: `eval/runs/query_resolution_<run-id>/run.json`

### Router eval integration

Router cases may optionally embed query-resolution expectations:

```json
"flags": {
  "query_resolution": {
    "web_contains": ["Kathmandu"],
    "web_fixture_id": "kathmandu_population",
    "min_web_hits": 1
  }
}
```

Enable fixture replay on router runs with `--with-web-fixtures` (see `tools/evaluate_router.py`).

Discourse simulation in router eval now applies `resolve_ambiguous_user_query` via
`core.query_resolution_evaluation.build_discourse_resolution` (parity with `LLMWorker`).

---

## Metrics captured

| Column | Meaning |
|--------|---------|
| `router_route` | Raw `CognitiveRouterV4.route()` output |
| `execution_route_pre_retrieval` | After LLMWorker-style overrides |
| `execution_route_final` | After simulated empty-retrieval downgrade |
| `strict_success` | Final route equals expected route |
| `family_success` | Route family match (CHAT / RETRIEVAL / WEB) |
| `failure_reason` | Primary failure classifier (see below) |
| `top_intent`, `top_score`, `chat_score`, `confidence_margin` | Router decision signals |
| `memory_hits`, `rag_hits`, `web_hits` | Retrieval counts (`--with-retrieval`) |
| `memory_candidates`, `rag_candidates` | Raw vector candidates before relevance gate |
| `downgrade_fired` | Post-retrieval downgrade to `none` |
| `rewrite_attempted`, `rewrite_applied`, `query_expansion_confidence` | Sidecar rewrite telemetry |
| `hybrid_extra_memory`, `hybrid_extra_rag` | Additional hits when rewrite applied |

### Route families

| Route | Family |
|-------|--------|
| `none` | CHAT |
| `memory`, `rag`, `hybrid` | RETRIEVAL |
| `web` | WEB |

### Failure reasons

`router_miss`, `override_changed_route`, `recall_fusion_upgrade`, `web_veto`,
`empty_retrieval`, `relevance_gate_removed_results`, `downgrade_to_none`,
`query_rewrite_rejected`, `route_label_mismatch`, `no_failure`

### Retrieval calibration metrics

| Metric | Meaning |
|--------|---------|
| `over_retrieval_rate` | CHAT-labeled cases that ended RETRIEVAL/WEB **with hits > 0** |
| `under_retrieval_rate` | Retrieval-expected cases that ended CHAT with **zero hits** |
| `recall_fusion_over_retrieval_share` | Fraction of over-retrieval caused by recall fusion |
| `retrieval_suppression_candidates` | High-`chat_score` CHAT cases that still retrieved |
| `potential_chat_guard_threshold_candidate` | Median chat_score of correct CHAT cases − ε (not enforced) |

Requires `--with-retrieval` (or `--eval-fixtures`) for hit-based over-retrieval counts.

### Report mode

```bash
venv/bin/python tools/evaluate_router.py --eval-fixtures --report --run-id analysis
```

Writes `report.md` with **Retrieval Calibration Summary**, strict/family accuracy,
failure cause distribution, memory analysis, and rewrite impact summary.

### Routing stability analysis (shadow mode)

Post-hoc clustering of similar prompts to detect route oscillation. **Does not
change routing outcomes.**

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --routing-stability-analysis \
  --report \
  --run-id stability_v1
```

Outputs `routing_stability_clusters.json` and `routing_stability` block in
`run.json` with per-cluster instability/entropy and oscillation flags.

### Route perturbation invariance (shadow A/B stress test)

Controlled paraphrase variants per case to measure routing consistency under
semantic pressure. **Does not change baseline eval results.**

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --report \
  --run-id perturb_v1
```

Outputs `route_perturbation_cases.json`, `route_perturbation` in `run.json`, and
perturbation cache under `eval/cache/` for reproducible re-runs.

### Routing hysteresis simulation (shadow)

Post-hoc enter/exit threshold buffers on perturbation variants. Requires
`--route-perturbation-analysis`.

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --simulate-hysteresis \
  --report \
  --run-id hysteresis_v1
```

Outputs `hysteresis_comparison.json` and `routing_hysteresis` in `run.json`.

### Routing canonicalization learner (shadow)

Learns canonical routes per perturbation cluster and sweeps decision-boundary
thresholds to estimate repairability of instability.

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --canonicalization-analysis \
  --report \
  --run-id canon_v1
```

Optionally combine with `--routing-stability-analysis` for stability-cluster
enrichment. Outputs `routing_canonicalization.json`.

### Continuous retrieval propensity (shadow)

Replaces binary `recall_fusion_triggered` with a smooth propensity model during
offline simulation. Requires `--route-perturbation-analysis`.

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --retrieval-propensity-analysis \
  --simulate-hysteresis \
  --report \
  --run-id propensity_v1
```

Outputs `retrieval_propensity.json`. Combine with `--simulate-hysteresis` for
comparison in the report.

### Continuous recall-fusion pilot (shadow → routing candidate)

Elevates propensity model to a gated routing pilot with comparisons vs
hysteresis and canonicalization.

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --continuous-pilot-routing \
  --simulate-hysteresis \
  --canonicalization-analysis \
  --report \
  --run-id continuous_pilot_v1
```

Outputs `continuous_pilot_routing.json`.

### Continuous architectural validation

Full end-to-end validation with category/flip breakdowns, threshold sweep,
and top unstable clusters.

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --continuous-arch-validation \
  --canonicalization-analysis \
  --simulate-hysteresis \
  --report \
  --run-id continuous_arch_validation_v1
```

Outputs `continuous_arch_validation.json` and `continuous_arch_validation` in
`run.json`.

### Shadow LLMWorker retrieval policy (execution observational layer)

Replays the continuous propensity model from `core/shadow_retrieval_policy.py`
against perturbation variants — mirrors the parallel path inside `LLMWorker`
without changing routing or retrieval execution.

```bash
venv/bin/python tools/evaluate_router.py \
  --eval-fixtures \
  --route-perturbation-analysis \
  --shadow-retrieval-policy-analysis \
  --report \
  --run-id shadow_policy_v1
```

Outputs `shadow_retrieval_policy.json` and `shadow_retrieval_policy` in
`run.json`. Report section: **SHADOW LLMWORKER RETRIEVAL POLICY ANALYSIS**.

Live turns log `shadow_retrieval_policy` via `RouterTelemetryBrain` when
`QUBE_SHADOW_RETRIEVAL_POLICY` is not disabled (default: on).

## Regression workflow

1. Compare against canonical baseline: `--baseline eval/runs/shadow_policy_baseline_v1/run.json`
2. Change router or shadow-policy code
3. Re-run locally (any `--run-id`; outputs stay gitignored unless promoted)
4. Use `--fail-on-regression` to gate CI or pre-merge checks
5. Promote a new baseline only when metrics should change — replace `shadow_policy_baseline_v1/` in a dedicated commit

## Related

- `core/router_evaluation.py` — harness logic + summary stats
- `tools/analyze_routing_outcomes.py` — live-session `retrieval_outcome` log analysis
- `docs/cognitive_router.md` — routing architecture
