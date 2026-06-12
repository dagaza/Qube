# Sidecar Model — Task Reference

This document lists every job the CPU sidecar performs in Qube, as defined by the codebase. The sidecar is a dedicated auxiliary LLM (default: **Qwen3 1.7B Q6_K** under `~/.qube/models/cognition/`) that runs on `SidecarLlmWorker` (`workers/sidecar_llm_worker.py`) and is accessed through `SidecarLlmClient` (`core/sidecar_llm.py`).

**Design constraints (enforced in code):**

- Runs on CPU only (`n_gpu_layers=0`); does not contend with the primary chat model for GPU.
- Priority command queue — one inference at a time; foreground tasks preempt background work.
- Assistive only — never chooses cognitive routes; never changes user-visible chat text.
- Foreground tasks honor `qube.sidecar.foreground_timeout_ms` (default 1500 ms) and fall back on timeout/failure.

**Canonical task enum:** `core/sidecar_types.py` → `SidecarTask` (8 values).

**Prompts, parsers, and inference params:** `core/sidecar_prompts.py`.

---

## Task index

| Task | Timing | Caller | Subsystem |
|------|--------|--------|-----------|
| `title` | Background | `ui/main_window.py` | Chat UI |
| `contradiction_judge` | Background | `workers/enrichment_worker.py` | Memory |
| `reflection_label` | Background | `workers/memory_reflection_worker.py` | Memory |
| `episode_summary` | Background | `workers/enrichment_worker.py` (via `generate()`) | Memory |
| `query_rewrite` | Foreground | `workers/llm_worker.py` → `core/sidecar_query_rewrite.py` | Retrieval |
| `source_digest` | Foreground | `workers/llm_worker.py` → `core/source_digest.py` | Retrieval / prompt |
| `ingest_blurb` | Background | `workers/ingestion_worker.py` | Library / RAG |
| `companion_line` | Background | `ui/companion/companion_verbal_scheduler.py` | Companion UI |

There is also a legacy **`raw_prompt`** execution path (not a `SidecarTask` enum member) used by `SidecarLlmClient.generate()` for pre-built prompts.

---

## 1. `title` — Chat session titling

**What it does:** After the first user+assistant exchange in a new conversation, generates a 2–5 word sidebar label (topic name, not a sentence).

**Caller:** `MainWindow._check_for_titling()` in `ui/main_window.py`.

**Trigger:** `LLMWorker.response_finished` when session history length is exactly 2 (one user message + one assistant reply).

**Execution:**

- Fire-and-forget: `SidecarLlmClient.enqueue_title(user_prompt, session_id, assistant_reply=...)`
- Worker op: `SidecarLlmWorker._do_title()`
- On success: `db.rename_session(session_id, title)` → emits `title_generated(session_id, title)`
- UI listens on `title_generated` to refresh the conversations history list

**Inference:** max_tokens 128, temperature 0.1.

**Output validation:** Heavy post-processing in `core/sidecar_prompts.py` (`_finalize_title_text`) — rejects verbatim user copies, sentence-like titles, reasoning blocks; extracts proper-noun phrases (e.g. “Lord of the Rings”).

**Settings:** Requires sidecar enabled and model on disk (`qube.sidecar.enabled`).

---

## 2. `contradiction_judge` — Memory fact relationship classifier

**What it does:** When enrichment finds a semantically similar existing memory row, classifies the relationship between the old and new fact as one of:

- `duplicate` — reinforce strength on existing row; do not insert
- `contradiction` — delete old row; insert new
- `complement` — keep old row; insert new (shared cluster)

**Caller:** `EnrichmentWorker._judge_contradiction()` in `workers/enrichment_worker.py`.

**Memory pipeline context:** Called from `_store_facts()` when vector similarity search returns an existing row above the dedup threshold. This is the **Phase B two-stage judge**:

1. **Stage 1 (no sidecar):** Jaccard token similarity ≥ 0.85 → `duplicate`
2. **Stage 2 (sidecar):** `cognition_llm.complete(SidecarTask.contradiction_judge, ...)` with 60 s timeout

**Wiring:** `main.py` passes `cognition_llm=self.sidecar_client` into `EnrichmentWorker`. If sidecar is unavailable, `cognition_llm` falls back to `extraction_llm` (primary model) only when explicitly set to `None` at construction — in production boot it is always the sidecar client.

**Fallback on LLM failure:** Rule-based `_is_contradiction()` heuristics.

**Inference:** max_tokens 8, temperature 0.1. Single-word answer expected.

**Settings:** Gated by `get_enable_memory_enrichment()` on the enrichment worker.

---

## 3. `reflection_label` — Memory audit labeling

**What it does:** Labels a single stored memory row for the reflection audit cycle. Valid labels:

- `durable_user_fact`
- `third_party_stub`
- `system_claim`
- `transient`
- `unclear`
- `tier_mismatch` (set deterministically before LLM — no sidecar call)
- `orphan_knowledge` (set deterministically before LLM — no sidecar call)

Anything other than `durable_user_fact` sets `flagged_for_review = True` on the LanceDB payload. The worker **never auto-deletes** flagged rows.

**Caller:** `MemoryReflectionWorker._reflect_one()` → `_call_llm()` in `workers/memory_reflection_worker.py`.

**Wiring:** `main.py` passes `llm=self.sidecar_client` into `MemoryReflectionWorker`.

**Cadence:**

- Wakes every 6 hours when candidates exist (`REFLECT_INTERVAL_SEC`)
- Extends to 24 hours when no candidates (`REFLECT_INTERVAL_IDLE_SEC`)
- Processes up to 10 memories per cycle (`BATCH_SIZE`)
- Skips rows reflected within the last 7 days (`MIN_REFLECT_AGE_SEC`)
- Priority: never-reflected → `unclear` label → oldest `last_reflected_at`

**Execution:** Blocking `complete(SidecarTask.reflection_label, prompt=..., timeout_sec=120.0)`. The full audit prompt is built in `_build_prompt()` and passed via the `prompt` kwarg (prebuilt path in `build_prompt_for_task`).

**Inference:** max_tokens 64, temperature 0.1. Strict JSON output.

**Settings:** Gated by `get_enable_memory_enrichment()` on the reflection worker.

---

## 4. `episode_summary` — Session episode summarization

**What it does:** Summarizes a bounded window of recent conversation into a single paragraph plus topic keywords, stored as a `qube_memory::episode::%` LanceDB row. Enables later “what have we been working on?” recall.

**Caller:** `EnrichmentWorker._summarise_session_now()` in `workers/enrichment_worker.py`, invoked by `_maybe_summarise_session()` after atomic fact extraction.

**Cadence triggers (either):**

- Turn counter reaches `EPISODE_SUMMARY_TURN_CADENCE` since last summary, or
- Session idle for `EPISODE_SUMMARY_IDLE_SEC` then next turn fires

**Execution path:** Blocking `cognition_llm.complete(SidecarTask.episode_summary, conversation=..., timeout_sec=120.0)`. Prompt and parser live in `core/sidecar_prompts.py` (`build_prompt_for_task` + `parse_task_output`).

**Expected output format:**

```
SUMMARY: <one paragraph, ≤120 words>
TOPICS: <comma-separated keywords>
```

Or `SUMMARY: SKIP` for trivial conversations.

**Post-processing:** Rejects thin content, assistant failure messages, negative-list matches; embeds summary vector; deduplicates near-identical episodes.

**Inference (enum defaults if using structured path):** max_tokens 220, temperature 0.2. Raw path uses max_tokens 256, temperature 0.2.

**Related (no sidecar):** `_maybe_daily_episode_rollup()` merges existing episode rows into a calendar-day summary deterministically — no LLM call.

---

## 5. `query_rewrite` — Assistive follow-up query expansion

**What it does:** For active deictic follow-ups (“what about its music?”, “tell me more”), proposes an expanded retrieval query using discourse entity/aspect and recent history. Used to improve RAG and memory hybrid retrieval without changing what the user typed.

**Caller:** `propose_query_expansion()` in `core/sidecar_query_rewrite.py`, called from `LLMWorker` during turn routing (after discourse state and follow-up classification).

**Inputs passed to sidecar:**

- `original_query`
- `retrieval_query` (discourse-expanded search string; read-only context)
- `tentative_route` (cognitive router execution route after overrides; read-only context)
- `topic` (conversation entity via `resolve_sidecar_discourse_context()` → `rewrite_referent_target()`)
- `active_aspect` (current discourse facet)
- `follow_up_kind`
- `history_tail` (last 4 turns, ≤1200 chars)

**Timing:** called after routing overrides finalize (not during early discourse prep) so `tentative_route` reflects the real execution lane.

**Optional JSON output (telemetry only):**

- `recommended_target` — `chat|memory|rag|web|none`; logged as `sidecar_recommended_target` on the routing decision; never changes execution route

**Post-inference guards:**

- Confidence must be ≥ `qube.sidecar.min_rewrite_confidence` (default 0.6)
- `expansion_adds_unanchored_proper_nouns()` rejects hallucinated proper names
- Unchanged or empty expansion → discarded

**Downstream effect:** `LLMWorker` runs primary retrieval on the original query, then optionally runs auxiliary retrieval on the expanded query and merges via `core/dual_query_retrieval.py` (`merge_memory_search_results`, `merge_rag_search_results`). Original query is always preserved in telemetry (`query_expansion_confidence`, `query_expansion_source` on routing decision).

**Execution:** Foreground blocking `complete()` with timeout = `foreground_timeout_ms / 1000`.

**Inference:** max_tokens 120, temperature 0.15. Strict JSON output.

**Settings:** `qube.sidecar.query_rewrite_enabled` (requires sidecar enabled).

---

## 6. `source_digest` — Retrieved source compression

**What it does:** Compresses retrieved memory or RAG sources into claim-oriented bullets before injection into the primary model prompt. Each source citation id `[N]` must be preserved.

**Callers:**

| Function | Called from | When |
|----------|-------------|------|
| `digest_memory_context()` | `LLMWorker` | MEMORY route returned `memory_context` + `memory_sources` |
| `digest_rag_context()` | `LLMWorker` | RAG route returned `tool_context` + `sources` |

Both live in `core/source_digest.py`.

**Execution:** Foreground blocking `complete(SidecarTask.source_digest, ...)` with `foreground_timeout_ms` timeout. Returns a `DigestResult` with `chars_before`, `chars_after`, `source_count`, and `skip_reason`.

**Conditional activation:** Digest runs only when retrieved context length is ≥ `qube.sidecar.source_digest_min_chars` (default 4096). Smaller, already-compact contexts pass through unchanged (`skip_reason=below_threshold`).

**Validation:** `parse_task_output` fails with `citation_ids_missing` if any expected `[id]` token is absent from the digest.

**Fallback:** On timeout, failure, below-threshold skip, or disabled setting → original raw context passed unchanged to the primary model.

**Telemetry:** Turn events record compression stats; `summarize()` exposes `digest.memory_skipped_below_threshold`, avg chars before/after.

**Inference:** max_tokens 400, temperature 0.2.

**Settings:** `qube.sidecar.source_digest_enabled` (requires sidecar enabled).

---

## 7. `ingest_blurb` — Library document one-liner

**What it does:** At document ingest time, generates one sentence (≤30 words) describing what the uploaded file is about.

**Caller:** `IngestionWorker` in `workers/ingestion_worker.py`, after chunks are embedded and written to LanceDB.

**Trigger:** After each file is indexed, if `get_sidecar_ingest_blurb_enabled()` and first chunk sample is available:

```python
self.sidecar_worker.enqueue_ingest_blurb(source, sample)  # sample = chunks[0][:2500]
```

**Execution:** Fire-and-forget queue op → `SidecarLlmWorker._do_ingest_blurb()` → emits `ingest_blurb_ready(filename, blurb)`.

**Persistence:** `main.py` `_on_ingest_blurb_ready()` → `db_manager.update_document_blurb(filename, blurb)` → refreshes Library view.

**Inference:** max_tokens 48, temperature 0.2.

**Settings:** `qube.sidecar.ingest_blurb_enabled` (requires sidecar enabled).

---

## 8. `companion_line` — Companion verbal captions

**What it does:** Generates short companion UI captions (idle quips, ingest acknowledgements, download acknowledgements). Output is JSON `{"line": "...", "kind": "idle_quip|ingest_ack|download_ack|skip"}` with quality gates (`is_acceptable_companion_line`).

**Callers:**

| Path | Module | Trigger |
|------|--------|---------|
| Legacy direct | `CompanionVerbalScheduler._request_line()` | When cognition v2 is off |
| Cognition v2 orchestrator | `CompanionVerbalScheduler._process_cognition()` | When orchestrator returns a `sidecar` payload (expression tier allows rewrite/generate) |
| Settings test | `CompanionVerbalScheduler.process_test_preview()` / `CompanionVerbalTestWorker` | `preview_companion_line()` blocking call |

**Event hooks that may enqueue sidecar work:**

- `idle` — periodic idle timer (30 s check)
- `ingest_complete` — after Library ingestion finishes
- `download_complete` — after model download completes
- `startup`, `model_loaded`, `milestone`, `usage_pattern` — cognition v2 orchestrator paths (may route to local templates instead of sidecar depending on capability tier)

**Execution:** Fire-and-forget `request_companion_line(payload)` → queue op `companion_line` → emits `companion_line_ready(line, kind, trigger)`.

**Capability gating:** `core/companion_cognition/capability.py` maps sidecar model basename to `ExpressionCapabilityTier`; smaller models stay on templates, larger models get `SIDECAR_REWRITE` or `FULL_GENERATE`. Telemetry can downgrade tier when companion line success rate drops below 60%.

**Inference:** max_tokens 64, temperature 0.35.

**Settings:** Companion verbal toggles (`get_companion_verbal_enabled`, cognition v2, expression freedom, trait preset, system prompt) affect whether and how sidecar is invoked.

---

## Memory subsystem — sidecar roles (detailed)

The sidecar is wired into **three memory workers/paths**. It is **not** used for atomic fact extraction, salvage, promotion, or consolidation.

### Uses sidecar

| Memory function | Worker | Sidecar API | Purpose |
|-----------------|--------|-------------|---------|
| Contradiction judge | `EnrichmentWorker` | `complete(contradiction_judge)` | Decide duplicate vs replace vs complement when dedup finds a near-match |
| Episode summarization | `EnrichmentWorker` | `generate(prompt)` | Write `qube_memory::episode::%` rows on cadence/idle |
| Reflection labeling | `MemoryReflectionWorker` | `complete(reflection_label)` | Audit stored memories; flag non-durable rows for user review |

### Does **not** use sidecar (primary model or deterministic)

| Memory function | Worker | Engine |
|-----------------|--------|--------|
| Atomic fact extraction (JSON) | `EnrichmentWorker` | Primary `LLMWorker` via `extraction_llm.generate(task=PrimaryEngineTask.memory_extraction)` |
| v7 salvage extraction | `EnrichmentWorker` | Primary `LLMWorker` via `_generate_memory()` |
| Daily episode rollup | `EnrichmentWorker` | Deterministic merge of existing episode rows |
| Promotion gates | `MemoryPromotionWorker` | Deterministic (`core/memory_promotion.py`) |
| Consolidation staging | `MemoryConsolidationWorker` | Deterministic (`core/memory_consolidation.py`) |
| Embedding / vector search | All memory workers | `EmbeddingModel` + LanceDB |

### Memory-related retrieval assist (foreground)

When the chat route includes MEMORY retrieval, the sidecar may also:

1. **`query_rewrite`** — expand deictic follow-ups for a second hybrid memory search pass
2. **`source_digest`** — compress `memory_context` before the primary model prompt

Both are invoked from `LLMWorker`, not from memory workers directly.

---

## Infrastructure (not user-facing tasks)

These are sidecar lifecycle operations, not `SidecarTask` values:

| Operation | Entry point | Purpose |
|-----------|-------------|---------|
| Model load | `SidecarLlmWorker.run()` | Load GGUF from `resolve_active_cognition_path()` on CPU |
| Hot reload | `reload_from_settings()` | Reload after Settings cognition model / chat format change |
| Stale override migration | `migrate_stale_sidecar_override()` | Clear invalid persisted `qube.sidecar.model_path` at boot |
| Priority queue | `core/sidecar_engine_queue.py` | `interactive` (query_rewrite, source_digest) > `background` > `control` |
| Queue wait telemetry | `submitted_at` / `dequeued_at` | Per-job `wait_ms` split from `inference_ms` in telemetry |
| Burst caps | `enqueue_companion_line` / `enqueue_ingest_blurb` | Defer companion when depth ≥ 8; coalesce + cap ingest blurbs (max 12 pending) |
| Telemetry | `SidecarTelemetryBrain` | Per-task latency, wait_ms, queue depth by priority, defer/coalesce counters |
| Degraded drain | `_run_degraded_queue_loop()` | Fail queued commands immediately when model unavailable |

**Model resolution:** `core/auxiliary_cognition.py` — bundled default, user override, 2 GB size cap, protected default file.

**Chat format / Qwen3:** `core/cognition_prompt_adapter.py` — format inference, `/no_think` injection for Qwen3 models.

---

## Settings keys

| Key | Default | Affects |
|-----|---------|---------|
| `qube.sidecar.enabled` | `true` | Master gate (also requires GGUF on disk) |
| `qube.sidecar.model_path` | `""` | Optional override GGUF path |
| `qube.sidecar.chat_format` | `auto` | Chat template (`chatml`, `llama-3`, `phi`, `gemma`) |
| `qube.sidecar.query_rewrite_enabled` | `true` | Task 5 |
| `qube.sidecar.source_digest_enabled` | `true` | Task 6 |
| `qube.sidecar.source_digest_min_chars` | `4096` | Task 6 threshold |
| `qube.sidecar.min_rewrite_confidence` | `0.6` | Task 5 acceptance threshold |
| `qube.sidecar.foreground_timeout_ms` | `1500` | Tasks 5 and 6 timeout |
| `qube.sidecar.ingest_blurb_enabled` | `true` | Task 7 |

Memory enrichment/reflection additionally require `get_enable_memory_enrichment()`.

---

## Source files (quick map)

| Area | Primary files |
|------|---------------|
| Types | `core/sidecar_types.py` |
| Client | `core/sidecar_llm.py` |
| Worker | `workers/sidecar_llm_worker.py` |
| Prompts / parsers | `core/sidecar_prompts.py` |
| Model path | `core/auxiliary_cognition.py` |
| Query rewrite | `core/sidecar_query_rewrite.py` |
| Source digest | `core/source_digest.py` |
| Telemetry | `core/sidecar_telemetry.py` |
| Settings | `core/app_settings.py`, `assets/config/settings.schema.json` |
| Boot wiring | `main.py` |
| Memory | `workers/enrichment_worker.py`, `workers/memory_reflection_worker.py` |
| Retrieval | `workers/llm_worker.py`, `core/dual_query_retrieval.py` |
| Library | `workers/ingestion_worker.py` |
| Companion | `ui/companion/companion_verbal_scheduler.py`, `core/companion_cognition/` |

---

*Generated from codebase state. Enum definition: `core/sidecar_types.py`.*
