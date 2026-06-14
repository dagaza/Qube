# Cognitive Router & Knowledge Pathways

This document explains how Qube chooses a **knowledge pathway** before the primary LLM answers a turn. The goal is to give the main model the right context: answer from its own weights (**self-knowledge / CHAT**), pull from **long-term memory**, search the **document library (RAG)**, or fetch **live web results**.

**Primary orchestrator:** `workers/llm_worker.py` — one turn runs through pre-router overrides, the cognitive router, post-router overrides, tool execution, relevance gates, and prompt assembly.

**Router implementation:** `mcp/cognitive_router.py` → `CognitiveRouterV4`.

**Assistive layers (never authoritative for routing):** discourse grounding, sidecar query rewrite, source digest, reasoning skills. The sidecar LLM and skills layer do **not** pick routes; see [`docs/sidecar_tasks.md`](sidecar_tasks.md) and [ADR 001](adr/001-skills-orthogonal-to-routing.md).

---

## Route vocabulary

| Route | Meaning | Typical tools / data |
|-------|---------|----------------------|
| `none` | **Self-knowledge (CHAT)** — no retrieval tools; the model answers from its parameters (optionally with light preference injection). | None required |
| `memory` | **Long-term memory** — LanceDB rows under `qube_memory::*`. | `mcp/memory_tool.py` |
| `rag` | **Document library** — uploaded PDFs, notes, ingested files. | `mcp/rag_tool.py` |
| `web` | **Live internet** — DuckDuckGo search via the internet MCP tool. | `search_internet` |
| `hybrid` | **Memory + RAG** — both retrievers run; the LLM synthesizes across sources. | Memory + RAG |

`INTERNET` is treated as an alias of `WEB` in execution paths.

---

## End-to-end turn flow

```mermaid
flowchart TD
    A[User message] --> B[Discourse grounding]
    B --> C{Pre-router overrides?}
    C -->|explicit remember| D[NONE — write turn]
    C -->|file search / @file| E[RAG only]
    C -->|narrative recap| F[MEMORY + episodes]
    C -->|@internet| G[WEB forced]
    C -->|attachments| H[Attachment routing patch]
    C -->|default| I[CognitiveRouterV4.route]
    I --> J[Post-router overrides]
    J --> K[Tool execution]
    K --> L[Relevance gates]
    L --> M{Any sources?}
    M -->|no on retrieval route| N[Downgrade to NONE]
    M -->|yes| O[Prompt build + LLM]
    N --> O
```

### 1. Query preparation (before routing)

When discourse grounding is enabled (`core/discourse_*`):

- **Follow-up classification** detects pronouns / continuation turns.
- **`resolve_routing_query`** and **`resolve_retrieval_query`** (`core/discourse_query.py`) may expand the query with the active topic or referent — e.g. `"Regarding Paris: what about hotels?"` — for embedding and search. This text is **not** shown to the user.
- **Sidecar query rewrite** (`core/sidecar_query_rewrite.py`) may propose an additional expanded query for hybrid retrieval when confidence is high enough. The original user text is always preserved for display and telemetry.

### 2. Pre-router overrides (beat the cognitive router)

These run **before** `CognitiveRouterV4.route()` and take absolute priority:

| Detector | Module | Route | Why |
|----------|--------|-------|-----|
| Explicit remember | `core/memory_filters.detect_explicit_remember` | `none` | Write turn — store a fact, no retrieval |
| File-search intent | `detect_file_search_intent` | `rag` | User scoped to documents only |
| Narrative recap | `detect_narrative_intent` | `memory` (+ episodes) | Session recap, not doc lookup |
| Composer `@file` / `@conversation` / `@internet` | attachment routing | varies | User-picked scope |

### 3. Cognitive router decision

If no pre-router override applies and `USE_COGNITIVE_ROUTER` is true (default in `LLMWorker`):

```python
intent_vector = embedding_cache.get_embedding(routing_query)
decision = cognitive_router.route(
    routing_query,
    intent_vector=intent_vector,
    weights=router_tuner.get_weights() if USE_ADAPTIVE_ROUTER else None,
)
```

The router returns a rich decision dict (route, per-lane scores, thresholds, tier telemetry, and a `trace` explaining *why*).

### 4. Post-router overrides (LLMWorker)

After `decision["route"]` is mapped to `execution_route`:

| Override | Condition | Effect |
|----------|-----------|--------|
| **Recall fusion** | `detect_recall_intent(prompt)` and route is `NONE`, `MEMORY`, or `RAG` | Force `HYBRID` so "tell me about X" searches memory **and** documents |
| **Discourse downgrade** | High-confidence follow-up with active topic | `MEMORY` / `RAG` / `HYBRID` → `NONE` (answer from thread context) |
| **Custom RAG triggers** | User-defined patterns in settings | Upgrade to RAG (or keep HYBRID) |
| **Manual / auto web** | Explicit web phrases, `force_web`, or `USE_COGNITIVE_ROUTER_INTERNET` + `internet_enabled` | Force `WEB` |
| **Proactive web veto** | Router picked WEB but internet tool is off and no explicit web trigger | Revert to `NONE` |
| **Discourse web veto** | Ungrounded deictic follow-up ("search for this") with no topic | `WEB` → `NONE` |

### 5. Tool execution

| Route | What runs |
|-------|-----------|
| `MEMORY` / `HYBRID` | `memory_search` (hybrid vector + FTS fusion) |
| `RAG` / `HYBRID` | `rag_search` (hybrid vector + FTS fusion) |
| `WEB` / `HYBRID` | `search_internet` with discourse-resolved query |
| `NONE` | Optional **core memory** pass — preferences-only, top 3 hits, MemGPT-style background injection |

Sidecar-assisted **dual-query retrieval**: when query expansion is accepted, `_memory_search_hybrid` / `_rag_search_hybrid` search both the primary and expanded queries, then merge via `core/dual_query_retrieval.py`.

### 6. Relevance gates & post-retrieval downgrade

Retrieval quality is enforced **after** routing:

| Channel | Gate | Module |
|---------|------|--------|
| RAG | `MIN_RAG_SEMANTIC_SCORE` (0.30) on vector hits | `mcp/rag_tool.py` |
| Memory | Semantic score + proper-noun / core-memory gates | `mcp/memory_tool.py` |
| Web | Token overlap + optional embedding similarity | `core/retrieval_relevance.py` |

If a retrieval route (`MEMORY`, `RAG`, `HYBRID`, `WEB`) finishes with **zero** `all_ui_sources`, `LLMWorker` **downgrades `execution_route` to `NONE`** before prompt build. This prevents citation-disciplined or "you have live web results" prompts when there is nothing to cite — a common cause of hallucinated `[W]` tokens on small models.

Telemetry still records the **original** routed lane for tuning.

### 7. Prompt assembly

`core/prompt_blocks.build_prompt_blocks` selects persona and suffixes by final `execution_route`:

- **Self-knowledge (`NONE`)** — base persona; optional `CHAT_PERSONALITY_SUFFIX`; preferences via `PREFERENCE_APPLICATION_SUFFIX` when applicable.
- **Retrieval routes** — citation discipline, recall-fusion hints, grounded-answer suffixes when sources exist.
- **WEB** — web-specific persona requiring `[W]` / bracket citations from real hits.
- **`web_capability_blocked`** — user asked for live data but internet is disabled → `WEB_CAPABILITY_DISABLED_SUFFIX` (honest limitation, not silent chat).
- **`rag_capability_blocked`** — user plausibly expected document/library retrieval but Local Knowledge Base is disabled → `RAG_CAPABILITY_DISABLED_SUFFIX` (honest limitation; memory sources may still be present).
- **`strict_isolation_enabled`** — toolbar Strict Isolation Mode → `STRICT_ISOLATION_SYSTEM_SUFFIX` on retrieval routes with sources (prompt-only v1).
- **`explicit_web_empty_results`** — explicit web request, search ran, zero hits → `EXPLICIT_WEB_EMPTY_SUFFIX`.

#### RAG capability veto (pre-retrieval)

Mirrors the WEB proactive veto in `LLMWorker` after NLP trigger routing and before tool execution:

| Condition | Normalization |
|-----------|----------------|
| `RAG` route, master KB off, no bypass | `execution_route → NONE`, `rag_vetoed_tool_disabled` |
| `HYBRID` route, master KB off, no bypass | `execution_route → MEMORY`, `rag_library_leg_skipped` |

**Bypasses** (library search still runs when master is off): NLP trigger phrases, explicit file-search intent (`detect_file_search_intent`), composer `@file` attachments.

When `rag_capability_blocked` is set (library intent + blocked + route `NONE`/`MEMORY`), prompt build uses `RAG_CAPABILITY_DISABLED_SUFFIX` instead of silent plain chat or misleading retrieval framing.

Post-retrieval empty-source downgrade (§6) remains unchanged and handles turns where search ran but returned zero hits.
 (`resolve_retrieval_wrapper_mode`): on plain `NONE` turns with preference-only hits, context is framed as **background** (not grounded citation mode).

---

## Cognitive Router V4 — internal design

`CognitiveRouterV4` is a **six-tier, additive** architecture. Tiers 1–3 affect the `route` field; tiers 4–6 are **observability-only** in v1 (they never change the outgoing route).

### Priority tree (Tier 1 core)

After per-lane scores are computed and gated:

```
complexity > 0.75        → hybrid
internet_enabled         → web
recall_active            → hybrid
rag_enabled ∧ memory_enabled → hybrid
rag_enabled              → rag
memory_enabled           → memory
else                     → none   (self-knowledge)
```

**Intent drift:** if the current query embedding is dissimilar from the previous turn (`similarity < 0.35`), all retrieval lanes are suppressed for that turn — a conversation topic shift guard.

**Load control:** recent RAG usage and rolling latency adjust thresholds via `_compute_dynamic_thresholds` and `_rag_load_penalty`.

### Per-lane scoring

Each lane gets a **substring score** (normalized hit count over trigger lists) and, when centroids are installed, an **embedding score** (cosine vs lane centroid). Final lane score = `max(substring, embedding)`.

| Lane | Substring triggers (examples) | Embedding centroid |
|------|------------------------------|-------------------|
| Memory | `remember`, `my preference`, `about me` | `_MEMORY_INTENT_EXAMPLES` |
| RAG | `pdf`, `document`, `according to`, `library` | `_RAG_INTENT_EXAMPLES` |
| Web | `weather`, `latest`, `search the web`, `today` | `_WEB_INTENT_EXAMPLES` |
| Recall | substring fallback via `detect_recall_intent` | recall centroid ("tell me about X", "who is Y") |
| Chat (negative class) | — | chat centroid (general knowledge questions) |

**Recall gate (T4.2):** `recall_active` requires `recall_score ≥ 0.62` **and** margin over chat class ≥ 0.05.

**Web gate:** embedding-only WEB picks must also beat the chat class by `web_margin_over_chat` (0.05). High-precision substring web triggers bypass this.

Centroids are built once by `LLMWorker` via `workers/intent_router.build_centroid` and installed with `set_*_centroid`.

### Tier 2 — confidence layer

Active only when at least one lane centroid exists (`tier2_active`):

- **Confidence floor downgrade:** weak embedding-only top scores (`< 0.30` and below lane threshold + 0.05) → downgrade to `none`.
- **Ambiguity upgrade:** if RAG and MEMORY are within `AMBIGUITY_MARGIN` (0.10) and both clear the floor → upgrade single-lane pick to `hybrid`.

Pure substring matches **bypass** the confidence floor.

### Tier 3 — feedback calibration

`LaneStatsRegistry` (`mcp/router_lane_stats.py`) tracks recent per-lane success. After each turn, `LLMWorker` emits a `RouteFeedbackEvent` via `cognitive_router.observe_feedback()`.

Success is **deterministic**: e.g. ≥1 surviving source on the routed lane after relevance gates. When `top_score` sits in the decision band `[0.30, 0.75]`, a damped lane bias (max ±0.03) nudges dynamic thresholds. Below 10 observations per lane, Tier 3 is dormant.

### Tiers 4–6 — observability

| Tier | Module | Role |
|------|--------|------|
| 4 | `mcp/routing_stability_tracker.py` | Clusters similar queries; flags oscillation |
| 5 | `mcp/routing_policy_engine.py` | Policy recommendations (`accept`, `stabilize`, `suppress_flip`, …) |
| 6 | `mcp/routing_arbitration_layer.py` | Cross-tier conflict flags + interpretation |

These populate `tier4_*`, `tier5_*`, `tier6_*`, and `decision["trace"]` for debugging. See [`docs/logging_and_diagnostics.md`](logging_and_diagnostics.md).

### Adaptive Router Self-Tuner

`mcp/router_self_tuner.py` → `AdaptiveRouterSelfTunerV2` adjusts sensitivity weights from turn telemetry:

- Hybrid with zero hits → lower `hybrid_sensitivity`
- Memory route with zero hits → lower `memory_sensitivity`
- RAG latency spike or over-fetch → lower `rag_sensitivity`

Weights feed back into `_compute_dynamic_thresholds` as `rag_sensitivity`, `memory_sensitivity`, `internet_sensitivity` (contract pinned by `tests/test_router_tuner_router_contract.py`).

---

## Self-knowledge (CHAT / `none`)

When no retrieval lane clears its threshold — or post-retrieval downgrade fires — the turn is **self-knowledge**:

1. The model answers from its trained weights.
2. **Core memory** may still inject stable **preferences** (metric units, name, etc.) on `NONE` turns without recall intent — a MemGPT-style background pass, gated by discourse follow-up confidence.
3. No citation-disciplined system prompt unless sources actually exist.
4. Presentation preferences are resolved via `core/preference_policy.py` (session > explicit settings > inferred profile).

This is the safe default when routing confidence is low: the router rules in `.cursor/rules/rag-engine.mdc` require fallback to CHAT under uncertainty.

---

## Assistive mechanisms (improve answers without choosing routes)

These tools help the main model produce better output **after** or **alongside** routing, but do not replace `CognitiveRouterV4`:

| Mechanism | What it does | Authoritative? |
|-----------|--------------|----------------|
| **Discourse grounding** | Topic tracking, query expansion, follow-up suppression, web query rewrite | No — overrides are rule-based in `LLMWorker` |
| **Sidecar query rewrite** | Extra retrieval query on follow-ups | No — merged with primary search; user text unchanged |
| **Source digest** | Compresses large memory/RAG context before prompt injection | No — falls back to raw context on timeout |
| **Retrieval fusion** | `fuse_weighted_scores` / `fuse_ranked_results` in vector+FTS merge | N/A — ranking only |
| **Preference policy** | Units/format hints for web snippets and queries | N/A — presentation layer |
| **Prompt blocks / renderers** | Route-specific system suffixes, citation rules, layout modes | N/A — prompt shaping |
| **IntentRouter / centroids** | Fast embedding infrastructure for router scoring | Feeds router only |

**Not in scope for knowledge routing:** `core/model_router.py` selects which **LLM model** to load, not which knowledge source to query.

**Legacy:** `mcp/router.py` (`SemanticRetrievalRouter`) is an older centroid router (memory / rag / hybrid / none). It is retained for reference; production chat uses `CognitiveRouterV4` exclusively.

---

## Configuration & feature flags

| Flag / setting | Location | Effect |
|----------------|----------|--------|
| `USE_COGNITIVE_ROUTER` | `LLMWorker` | Master switch (default `True`) |
| `USE_ADAPTIVE_ROUTER` | `LLMWorker` | Enables self-tuner weights |
| `USE_COGNITIVE_ROUTER_INTERNET` | `LLMWorker` | Auto-web when router sets `internet_enabled` |
| `mcp_rag_enabled` / `mcp_internet_enabled` | worker MCP flags | Tool availability |
| Discourse grounding | `get_discourse_grounding_enabled()` | Query expansion + follow-up logic |
| Sidecar rewrite / digest | `core/app_settings.py` | Assistive retrieval & compression |
| Custom RAG triggers | settings / `core/rag_trigger_routing.py` | User-defined RAG upgrade patterns |

---

## Diagnostics

| Surface | Purpose |
|---------|---------|
| `mcp/routing_debug.py` + Routing Debug UI | Per-turn decision records, trace, effective route |
| `RouterTelemetryBrain` | Dashboard signals (distribution, latency, sensitivities) |
| `QUBE_ROUTING_DEBUG_LOG=1` | Persist routing records to `~/.qube/logs/routing_debug.log` |
| `tools/view_routing_logs.py` | Tail/filter routing log |
| `tools/analyze_routing_outcomes.py` | Offline summary of `retrieval_outcome` joins (enable `QUBE_ROUTING_DEBUG_LOG=1`) |
| Greppable markers | `[RouterV4]`, `[Tier5Policy]`, `[Tier6RAL]`, `[Discourse]`, `[WebPipeline]` |

Decision dict keys are **additive** across tiers (`tier2_active`, `tier3_band_active`, `memory_score_final`, `trace.winning_reason`, …). Older telemetry consumers continue to work when new keys appear.

### Joined retrieval-outcome telemetry (schema v2)

After tool execution and post-retrieval downgrade, `LLMWorker` merges a `retrieval_outcome` block into the routing-debug record via `merge_retrieval_outcome_into_latest()`. Each block joins:

- `router_route` vs `execution_route_pre_downgrade` vs `execution_route_final`
- `downgrade_fired`, per-lane hit counts, sidecar rewrite fields
- Router confidence (`top_intent`, `top_score`, `chat_score`)

Persist with `QUBE_ROUTING_DEBUG_LOG=1`, then run `python tools/analyze_routing_outcomes.py` for offline summaries.

---

## Design constraints (intentional)

From the Qube execution model:

- **No LLM inside the router** — scoring is substring + embedding math; target &lt;10 ms.
- **No multi-hop tool chains** — at most one primary route class per turn (HYBRID runs two retrievers in parallel, not sequentially).
- **No DAG planners** — single decision, deterministic priority tree.
- **Sidecar is assistive only** — never sets `execution_route`.
- **Safety fallback** — low confidence → `none` (self-knowledge), not aggressive retrieval.

---

## Offline evaluation framework

Regression-test routing against a labeled corpus without running the full UI or LLM stack.

| Artifact | Purpose |
|----------|---------|
| [`eval/router_corpus/v1_baseline.json`](../eval/router_corpus/v1_baseline.json) | 100-case baseline corpus (general knowledge, memory, RAG, web, follow-ups, ambiguous, adversarial) |
| [`eval/router_corpus.schema.json`](../eval/router_corpus.schema.json) | Corpus JSON schema (`qube.router_corpus.v1`) |
| [`core/router_evaluation.py`](../core/router_evaluation.py) | Harness: simulate LLMWorker overrides, summaries, regression compare |
| [`tools/evaluate_router.py`](../tools/evaluate_router.py) | CLI runner |
| [`eval/README.md`](../eval/README.md) | Corpus format, metrics, regression workflow |

```bash
# Substring-only router (no GGUF embedder)
python3 tools/evaluate_router.py --no-embeddings

# Full Tier-2 centroids + optional LanceDB hits
python3 tools/evaluate_router.py --with-retrieval

# Regression gate vs a saved run
python3 tools/evaluate_router.py --no-embeddings \
  --baseline eval/runs/baseline/run.json --fail-on-regression

# Seed fixture library + memories and run retrieval-backed eval
venv/bin/python tools/evaluate_router.py --eval-fixtures

# Full observability report (strict + family accuracy, failure causes)
venv/bin/python tools/evaluate_router.py --eval-fixtures --report

# Shadow-mode routing stability clustering (no routing changes)
venv/bin/python tools/evaluate_router.py --eval-fixtures --routing-stability-analysis --report

# Paraphrase invariance stress test (shadow mode)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --report

# Hysteresis boundary simulation on perturbation variants (shadow mode)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --simulate-hysteresis --report

# Canonical route learner + boundary sweep (shadow mode)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --canonicalization-analysis --report

# Continuous retrieval propensity model (shadow mode)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --retrieval-propensity-analysis --report

# Continuous recall-fusion pilot candidate (shadow mode)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --continuous-pilot-routing --simulate-hysteresis --canonicalization-analysis --report

# Full architectural validation of continuous pilot (shadow mode)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --continuous-arch-validation --canonicalization-analysis --simulate-hysteresis --report

# Shadow LLMWorker retrieval policy replay (observational execution layer)
venv/bin/python tools/evaluate_router.py --eval-fixtures --route-perturbation-analysis --shadow-retrieval-policy-analysis --report
```

Each run writes `eval/runs/<run_id>/results.csv` and `run.json` (`qube.router_eval_run.v1`) with per-case metrics (`router_route`, `execution_route_pre_retrieval`, `execution_route_final`, scores, hit counts, downgrade/rewrite flags) plus category-level accuracy, confusion matrix, and retrieval hit rates.

---

## Related docs & tests

- [`docs/sidecar_tasks.md`](sidecar_tasks.md) — sidecar tasks that assist retrieval
- [`docs/logging_and_diagnostics.md`](logging_and_diagnostics.md) — env vars and log markers
- [`docs/rag_relevance_and_router_T4_plan.md`](rag_relevance_and_router_T4_plan.md) — T4 recall/chat margin design notes
- [`docs/memory_manual_qa.md`](memory_manual_qa.md) — manual QA scenarios (web veto, empty downgrade)
- [`eval/README.md`](../eval/README.md) — router evaluation corpus and regression workflow

**Test surface:** `tests/test_cognitive_router_tier{2,3,4,5,6}.py`, `tests/test_cognitive_router_margin.py`, `tests/test_router_tuner_router_contract.py`, `tests/test_rag_relevance_gate.py`, `tests/test_web_veto_fallback.py`, `tests/test_routing_debug.py`, `tests/test_router_evaluation.py`.

---

*Describes the codebase as of the Cognitive Router V4 tiered architecture. Tiers 4–6 remain observability-only until a future release wires policy overrides into execution.*
