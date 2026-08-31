# Qube — Logging & Diagnostics (Developer / Tester Guide)

This document is the canonical reference for **where logs go**, **how to turn observability on**, **CLI flags**, **environment variables**, and **what each layer records**. It covers chat inference, routing, memory, sidecar cognition, and output sanitization.

---

## Quick reference

| Destination | Logger name | Default path | Terminal? | Always created? |
|-------------|-------------|--------------|-----------|-----------------|
| **Application log** | `Qube.*` (except dedicated debug loggers) | `~/.qube/logs/qube.log` | Yes | Yes (INFO default; disable with `QUBE_APP_LOG=0`) |
| **LLM debug file** | `Qube.NativeLLM.Debug` | `~/.qube/logs/llm_debug.log` | No (file only) | Yes (sink attached at boot) |
| **Routing debug file** | `Qube.RoutingDebug` | `~/.qube/logs/routing_debug.log` | No (file only) | Yes (sink attached at boot) |
| **Skills debug file** | `Qube.SkillsDebug` | `~/.qube/logs/skills_debug.log` | No (file only) | Yes (sink attached at boot) |

**Note:** The terminal still receives all loggers (including third-party DEBUG noise when `logging.basicConfig(level=DEBUG)`). The **application log file** records `Qube.*` at **INFO** by default so post-mortems stay readable without duplicating LLM/routing/skills debug files.

**Important:** LLM and routing debug log files live under **`~/.qube/logs/`** (see `core/paths.logs_dir()`). They are **not** written to `<repo>/logs/` — a stale `~/.qube/logs/llm_debug.log` in the repo checkout is not updated by the running app. Other user data (SQLite, LanceDB, exports, model overrides) also uses `~/.qube`.

Rotating file sinks: **10 MB** per file, **5** backups (`core/app_log_sink.py`, `core/llm_debug_sink.py`, `core/routing_debug_sink.py`).

For the **canonical LLM trace debugging stack** (Truth Diff, golden traces, fingerprinting, trace diff UI), see **[canonical_llm_trace_debugging.md](canonical_llm_trace_debugging.md)**.

---

## Boot-time defaults in `main.py`

These are set in code today (no env var needed):

```python
os.environ["QUBE_LLM_DEBUG"] = "1"
os.environ["QUBE_LOG_RAW_COMPLETION"] = "1"
```

Also at startup:

- `logging.basicConfig(level=logging.DEBUG, …)` — most `Qube.*` loggers print to the **terminal**
- `init_app_logging()` — routes general `Qube.*` → `~/.qube/logs/qube.log` (INFO default; terminal unchanged)
- `init_llm_debug_logging()` — routes `Qube.NativeLLM.Debug` → `~/.qube/logs/llm_debug.log` only
- `init_routing_debug_logging()` — routes `Qube.RoutingDebug` → `~/.qube/logs/routing_debug.log` only
- `init_skills_debug_logging()` — routes `Qube.SkillsDebug` → `~/.qube/logs/skills_debug.log` only

To **disable** hardcoded flags, remove or comment the `os.environ[…]` lines in `main.py`, or override in the shell **before** launch (later assignment in `main.py` wins if it runs first — edit `main.py` for a permanent off).

---

## CLI flags

Parsed by `core/boot_args.py`:

| Flag | Example | Effect |
|------|---------|--------|
| `--routing-debug` | `python3 main.py --routing-debug` | Opens a **detached Routing Debug side window** at startup. Does **not** enable file logging by itself. |
| `--winget-validation` | `Qube.exe --winget-validation` | CI / WinGet smoke only: defer CUDA loads, mock bootstrap, write `.winget-validation-boot-trace.jsonl`. |
| `--bootstrap-trace` | `Qube.exe --bootstrap-trace` | Granular bootstrap/launch JSONL + stderr (see below). Also sets `QUBE_BOOTSTRAP_TRACE=1`. |

No other CLI logging flags exist today. All other diagnostics use environment variables or `main.py` edits.

### Bootstrap trace (`--bootstrap-trace`)

For first-run / splash debugging on a test machine (especially Windows):

**Terminal A — launch with trace:**

```powershell
& "$env:LOCALAPPDATA\Programs\Qube\Qube.exe" --bootstrap-trace
```

**Terminal B — follow the trace file live:**

```powershell
.\scripts\diagnostics\tail_bootstrap_trace.ps1
# or manually:
Get-Content "$env:LOCALAPPDATA\Qube\logs\bootstrap-trace.jsonl" -Wait -Tail 30
```

Each line is JSON with `event`, `timestamp`, and step-specific fields (`pending_models`, `repo_file`, `phase`, errors, …). The latest event is also mirrored to `bootstrap-state.json`.

Equivalent without CLI flag: `$env:QUBE_BOOTSTRAP_TRACE = "1"` before launch.

Also bumps `QUBE_APP_LOG_LEVEL` to `DEBUG` for richer `qube.log` capture. General app log path: `%LOCALAPPDATA%\Qube\logs\qube.log`.

---

## Developer CLI tools

Run from the repo root:

### View application log

```bash
tail -f ~/.qube/logs/qube.log
grep -i "voice capture" ~/.qube/logs/qube.log
grep -i "Manual voice" ~/.qube/logs/qube.log
```

Also available in **Settings → Diagnostics** (“Application log”) and **Settings → Privacy & data** (audit logs).

### View LLM debug log

```bash
python3 tools/view_llm_logs.py --last 200
python3 tools/view_llm_logs.py --follow
python3 tools/view_llm_logs.py --filter llm_completion_output_trace --last 50
python3 tools/view_llm_logs.py --filter token_trace --last 500
python3 tools/view_llm_logs.py --filter llm_prompt_validation --last 100
```

Direct file access (equivalent):

```bash
tail -f ~/.qube/logs/llm_debug.log
grep llm_completion_output_trace ~/.qube/logs/llm_debug.log
```

### View routing debug log

Requires `QUBE_ROUTING_DEBUG_LOG=1` (see below) for JSONL lines to appear.

```bash
python3 tools/view_routing_logs.py --last 200
python3 tools/view_routing_logs.py --follow --filter HYBRID
```

### Prompt parity (LM Studio vs Qube)

1. Enable `QUBE_LLM_DEBUG=1` and optionally set `QUBE_LLM_DEBUG_FILE=/path/to/prompts.log`
2. Export LM Studio’s rendered prompt to a text file
3. Diff:

```bash
python3 tools/llm_prompt_diff.py studio_prompt.txt qube_prompt.txt
# Optional: DIFF_CONTEXT=5 python3 tools/llm_prompt_diff.py ...
```

### Load-time prompt ablation (diagnostic only)

Does **not** run inside normal chat; loads a GGUF and runs harness scenarios. May write `~/.qube/model_overrides.json`.

```bash
python3 -m tools.run_ablation --model /path/to/model.gguf --message "Hello" --json-out ablation_report.json
```

---

## In-app UI surfaces

| Surface | How to enable | What it shows |
|---------|---------------|---------------|
| **Telemetry** screen | Always in nav | Hardware, pipeline latency, router metrics, sidecar summary |
| **Routing Debug window** | `python3 main.py --routing-debug` | Per-turn routing records (in-memory buffer, up to 100 turns) |
| **LLM debug log panel** | `QUBE_LLM_LOG_UI=1` + restart | Tail of `~/.qube/logs/llm_debug.log` on Telemetry screen (developer-only) |

The Routing Debug **UI** and **file log** are independent: the UI works without `QUBE_ROUTING_DEBUG_LOG`; the file log requires that env var.

---

## Environment variables (complete list)

Truthy values are usually: `1`, `true`, `yes`, `on` (case-insensitive unless noted).

### General application log

| Variable | Default | Log destination | Purpose |
|----------|---------|-----------------|---------|
| `QUBE_APP_LOG` | **`1` (on)** | `~/.qube/logs/qube.log` | Master switch for the general app log file |
| `QUBE_APP_LOG_LEVEL` | `INFO` | same | File level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `QUBE_APP_LOG_FILE` | (unset) | custom path | Override log file location |

Logger scope: all `Qube.*` except `Qube.NativeLLM.Debug`, `Qube.RoutingDebug`, and `Qube.SkillsDebug` (those stay in their dedicated files).

### Native LLM / chat inference

| Variable | Default | Log destination | Purpose |
|----------|---------|-----------------|---------|
| `QUBE_LLM_DEBUG` | **`1` in `main.py`** | `~/.qube/logs/llm_debug.log` | Reconstructed prompts, stop merge, `[LLM-DEBUG]` blocks; enables prompt validation JSON when combined with inference |
| `QUBE_LLM_DEBUG_FILE` | (unset) | **Additional file** you specify | Appends raw reconstructed prompt per native request |
| `QUBE_LLM_PROMPT_VALIDATE` | off | `~/.qube/logs/llm_debug.log` | JSON `llm_prompt_validation` events (also on when `QUBE_LLM_DEBUG=1`) |
| `QUBE_LLM_REFERENCE_JSON` | (unset) | (read-only input) | LM Studio parity baseline path; used by validation / causality / counterfactual |
| `QUBE_LOG_RAW_COMPLETION` | **`1` in `main.py`** | `~/.qube/logs/llm_debug.log` | JSON `llm_completion_output_trace` — raw vs filtered vs presented text |
| `QUBE_LOG_RAW_COMPLETION_MAX_CHARS` | `0` (unlimited) | same | Truncate each text field in completion trace; adds `*_full_len` metadata |
| `QUBE_LLM_TOKEN_TRACE` | off | `~/.qube/logs/llm_debug.log` | Token-level JSON: `llm_token_trace`, `llm_token_trace_live`, `llm_token_trace_ground_truth` |
| `QUBE_LLM_TOKEN_TRACE_N` | `20` | same | Max tokens in complete trace |
| `QUBE_LLM_TOKEN_TRACE_EARLY` | `5` | same | Early-phase token trace threshold |
| `QUBE_LLM_CAUSALITY` | off | `~/.qube/logs/llm_debug.log` | JSON `llm_execution_causality_report` (post-inference, observer-only) |
| `QUBE_LLM_COUNTERFACTUAL` | off | `~/.qube/logs/llm_debug.log` | JSON `llm_counterfactual_simulation` (analytical, no extra inference) |
| `QUBE_ENGINE_INPUT_TRACE` | off | Routing Debug record + `[LLM-DEBUG][engine_input]` preview | Captures llama.cpp `create_completion(prompt=…)` boundary; preview in llm_debug when `QUBE_LLM_DEBUG=1` |
| `QUBE_LOG_NATIVE_CHAT` | off | **Terminal** (`Qube.NativeLlamaInference`) | Lightweight message summary per native chat call (not full prompt) |

**Native post-inference pipeline order** (each gated by its env flag; all observer-only):

1. Prompt validation / `QUBE_LLM_DEBUG` baseline  
2. `QUBE_LLM_TOKEN_TRACE` — live vs post-hoc vs sampler ground truth  
3. `QUBE_LLM_CAUSALITY` — influence scores, first-token cause  
4. `QUBE_LLM_COUNTERFACTUAL` — one intervention at a time  

Logger: `Qube.NativeLLM.Debug` → `~/.qube/logs/llm_debug.log`.

**Load-time only (not per chat turn):**

- `llm_model_behavior_profile` JSON after model load (from `core/model_behavior.py` + ablation harness)
- Ablation CLI: `python -m tools.run_ablation` → optional `~/.qube/model_overrides.json`

### Routing & cognitive router

| Variable | Default | Log destination | Purpose |
|----------|---------|-----------------|---------|
| `QUBE_ROUTING_DEBUG_LOG` | off | `~/.qube/logs/routing_debug.log` | One compact JSONL record per turn (`schema_version`, route, tier signals, …) |
| `QUBE_ROUTING_DEBUG_LOG_VERBOSE` | off | same | Larger payload in JSONL |
| `QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY` | off | same | Replace query with `sha256:` digest in file log |

**Terminal** (always when those code paths run): `Qube.CognitiveRouterV4` emits selective INFO lines, e.g.:

- `[Tier5Policy] …` when policy ≠ accept/no_action  
- `[Tier6RAL] …` when conflict flags fire  
- Web veto: grep `Cognitive router picked WEB but internet tool is disabled`

**In-memory:** `mcp/routing_debug.py` buffer (100 records) feeds the Routing Debug UI and merge hooks from `LLMWorker`.

### Discourse / follow-up routing

| Variable | Default | Log destination | Purpose |
|----------|---------|-----------------|---------|
| `QUBE_DISCOURSE_DEBUG` | off | **Terminal** (`Qube.LLM`) | `[Discourse] follow_up=… topic=… wrapper=… core_memory_suppressed=…` per turn |

Structured discourse events in `llm_debug.log` (always when discourse grounding runs):

- `discourse_referent_trace` — referent promoted after assistant reply
- `discourse_referent_rejected` — promotion/history candidate rejected (`reject_reason`, `prior_referent`)
- `discourse_rewrite_validation_failed` — possessive rewrite rejected after substitution
- `discourse_query_rewrite` / `discourse_prompt_rewrite` — inference grounding applied

See `docs/discourse_grounding_referent_stability_plan.md` for QA scenarios.

### Sidecar (CPU assistive cognition)

| Variable | Default | Log destination | Purpose |
|----------|---------|-----------------|---------|
| `QUBE_SIDECAR_DEBUG` | off | **Terminal** (`Qube.Sidecar.Telemetry`) | Full `[SidecarTelemetry] {event}` JSON per sidecar task; otherwise DEBUG one-liners |

Sidecar aggregate summary is also shown on the **Telemetry** screen (no env var required).

### UI / markdown debugging

| Variable | Default | Log destination | Purpose |
|----------|---------|-----------------|---------|
| `QUBE_DUMP_MARKDOWN_HTML` | off | **stderr** | Dumps markdown/HTML length diagnostics from chat bubble rendering |
| `QUBE_LLM_LOG_UI` | off | In-app panel | Embeds `LLMDebugLogPanel` on Telemetry |

### Feature flags (not logging, but env-gated)

| Variable | Purpose |
|----------|---------|
| `QUBE_COMPANION` | Enable companion subsystem (`core/app_settings.py`) |
| `QUBE_COMPANION_FORCE_TIER` | Force companion capability tier |
| `QUBE_REDUCED_MOTION` | Companion animation reduction |
| `QUBE_HF_OFFICIAL_BRANDING` | HF publisher branding in Model Manager (default on) |

---

## JSON events in `~/.qube/logs/llm_debug.log`

| `event` field | Enabled by |
|---------------|------------|
| `llm_prompt_validation` | `QUBE_LLM_DEBUG` or `QUBE_LLM_PROMPT_VALIDATE` |
| `llm_completion_output_trace` | `QUBE_LOG_RAW_COMPLETION` |
| `llm_token_trace` / `llm_token_trace_live` / `llm_token_trace_ground_truth` | `QUBE_LLM_TOKEN_TRACE` |
| `llm_execution_causality_report` | `QUBE_LLM_CAUSALITY` |
| `llm_counterfactual_simulation` | `QUBE_LLM_COUNTERFACTUAL` |
| `llm_model_behavior_profile` | Model load (internal engine) |
| `llm_debug_exchange_begin` / `llm_debug_exchange_end` | `QUBE_LLM_DEBUG` (always on in `main.py`) |
| `llm_engine_job_timing` | `QUBE_LLM_DEBUG` — per native-engine job (chat + background) |
| `llm_engine_background_job_timing` | `QUBE_LLM_DEBUG` — background `chat_once` jobs (e.g. memory extraction) |
| `llm_engine_queue_snapshot` | `QUBE_LLM_DEBUG` — queue depth on each enqueue |
| `discourse_referent_trace` | Referent promoted after assistant turn |
| `discourse_referent_rejected` | Candidate referent rejected by stability/validation policy |
| `discourse_rewrite_validation_failed` | Inference query rewrite failed post-substitution sanity check |
| `discourse_query_rewrite` | Successful deictic/possessive query substitution |
| `discourse_prompt_rewrite` | Prompt grounding prefix or resolved user line applied |

**Exchange timing fields** (`llm_debug_exchange_end`, when `QUBE_LLM_DEBUG=1`):

- `worker_prep_ms` — `LLMWorker.run()` start → `enqueue_generation` (routing, RAG, prompt build)
- `engine_queue_wait_ms` — engine job `submitted_at` → dequeue (contention with background extraction)
- `engine_inference_ms` — first model token → last token (`create_completion` boundary)
- `exchange_total_ms` — exchange begin → exchange end

**Engine job timing fields** (`llm_engine_job_timing`):

- `queue_wait_ms`, `engine_prep_ms`, `inference_ms`, `total_ms`
- `queue_depth_at_submit`, `queue_depth_at_start`, `queued_behind` (e.g. `["memory_extraction"]`)
- `cancelled`, `preempted_by` (background jobs cancelled when chat preempts)

**Marker semantics (important):**

- `[QUBE INFERENCE BEGIN/END]` — prompt logging only (before `create_completion`)
- `[QUBE INFERENCE TOKEN BEGIN/END]` — true GPU/token generation boundary

Plus plain-text blocks: `[LLM-DEBUG] …`, `[CompletionOutputTrace] …`, `[LLM-DEBUG][engine_input] …`.

---

## Terminal loggers (by domain)

These go to **stdout/terminal** via root logging (not the dedicated log files):

| Logger | Examples |
|--------|----------|
| `Qube.LLM` | `[PromptLayout]`, `[Discourse]`, `[LLM] SSE/native errors, routing centroid install |
| `Qube.NativeLLM` | `[OutputValidation]`, `[ResponseQuality]`, `[ChatContract]`, `[Native]`, `[ModelPerformance]` |
| `Qube.NativeLlamaInference` | `[Native][chat]` summary when `QUBE_LOG_NATIVE_CHAT=1` |
| `Qube.CognitiveRouterV4` | Tier 5/6 policy INFO, web veto |
| `Qube.PromptTemplateRouter` | `[LLM-TEMPLATE]`, `[LLM-PROMPT-ROUTER]`, `[LLM-TEMPLATE-OVERRIDE]`, `[LLM-SELF-HEAL-APPLY]` |
| `Qube.Sidecar.Telemetry` | Sidecar task events |
| `Qube.Memory*` workers | Enrichment, promotion, consolidation, reflection |
| `Qube.UI.*` | Chat UI, TTS lifecycle, stop button |
| `Qube.Core` / `Qube.Main` | Boot, shutdown |

**Note:** `Qube.NativeLLM.Debug` and `Qube.RoutingDebug` are **file-only** (`propagate=False`).

---

## Output sanitization vs logging

Qube strips non-user-facing model text before UI, TTS, and SQLite. Layers (native path):

1. **`RedactedThinkingStreamFilter`** — `<think>`, `<thinking>`, …  
2. **`LeadingMetaInstructionStripper`** — “Provide final answer”-style openers  
3. **`strip_harmony_oss_artifacts()`** — Harmony/OSS tokens, scratchpad tails, Mistral markers (`core/output_artifact_strip.py`)  
4. **`validate_output` / adaptive retry** — may replace output on structural issues (`core/output_validation.py`, `core/adaptive_retry.py`)  
5. Final harmony strip in `LLMWorker.run()` before `response_finished`  

**External HTTP** path: raw SSE deltas during stream; harmony strip at end of SSE loop (no thinking/meta stream filters).

### Completion output trace (raw vs presented)

When `QUBE_LOG_RAW_COMPLETION=1`, each chat turn logs:

| Field | Meaning |
|-------|---------|
| `raw_text` | Verbatim model output before worker sanitization |
| `after_worker_filters` | Full-string pass through thinking/meta/harmony filters (native) |
| `streamed_incremental` | Text built during incremental streaming before end reconcile (native) |
| `worker_return_text` | What `LLMWorker` returns (written to SQLite) |
| `presented_text` | After final strip in `run()` (UI `response_finished`) |
| `stages_changed` | Which consecutive stages differed |
| `removed_char_count` | `len(raw) - len(presented)` |

Implementation: `core/completion_output_trace.py`, wired from `workers/llm_worker.py`.

---

## Routing debug: UI vs file vs terminal

```
User query
    → CognitiveRouterV4.route()  → decision dict (tier1…tier6 keys)
    → LLMWorker executes tools + LLM
    → RoutingDebugBuffer (in-memory, max 100)
         ├─ Routing Debug UI (--routing-debug window)
         └─ QUBE_ROUTING_DEBUG_LOG=1 → ~/.qube/logs/routing_debug.log (JSONL)
```

Merge hooks also attach **model router**, **chat contract**, and **engine input trace** snapshots into the latest routing record when available.

---

## Persistent diagnostic artifacts (`~/.qube`)

Not log files, but relevant when debugging inference:

| Path | Content |
|------|---------|
| `~/.qube/model_overrides.json` | Self-heal learned stop tokens / assistant anchor |
| `~/.qube/prompt_layout_overrides.json` | Per-model prompt layout overrides |
| `~/.qube/exports/memory_*.md` | Memory Manager exports |
| `~/.qube/system_data/` | Capability detection DB/JSON |

---

## Recommended workflows

### “What was stripped from the model answer?”

1. Ensure `QUBE_LOG_RAW_COMPLETION=1` (default via `main.py` today)  
2. Reproduce a chat turn  
3. `python3 tools/view_llm_logs.py --filter llm_completion_output_trace --last 5`  
4. Compare `raw_text` vs `presented_text` in the JSON line  

### “Why did routing pick WEB / HYBRID / NONE?”

1. `python3 main.py --routing-debug`  
2. Optionally `export QUBE_ROUTING_DEBUG_LOG=1` for JSONL history  
3. Inspect tier fields in the UI detail pane or `~/.qube/logs/routing_debug.log`  

### “Prompt doesn’t match LM Studio”

1. `QUBE_LLM_DEBUG=1` + `QUBE_LLM_DEBUG_FILE=./qube_prompts.log`  
2. Export LM Studio prompt  
3. `python3 tools/llm_prompt_diff.py lm_studio.txt qube_prompts.log`  
4. Optional: `QUBE_LLM_REFERENCE_JSON=ref.json` for automated parity scoring  

### “First token / template leakage”

1. `export QUBE_LLM_TOKEN_TRACE=1`  
2. `export QUBE_LLM_CAUSALITY=1`  
3. Reproduce on **internal engine**  
4. Filter `~/.qube/logs/llm_debug.log` for `llm_token_trace` and `llm_execution_causality_report`  

### “Follow-up / discourse routing”

1. `export QUBE_DISCOURSE_DEBUG=1`  
2. Watch terminal for `[Discourse]` lines during multi-turn chat  

---

## Greppable markers

| Marker / substring | Location |
|--------------------|----------|
| `llm_completion_output_trace` | Raw vs presented completion |
| `[CompletionOutputTrace]` | One-line completion trace summary |
| `[LLM-DEBUG]` | Native prompt reconstruction |
| `llm_prompt_validation` | Prompt structure validation JSON |
| `[OutputValidation]` | Terminal — validation issue labels |
| `[ChatContract] CHAT CONTRACT VIOLATION` | Terminal — template marker leakage |
| `[Tier5Policy]` / `[Tier6RAL]` | Terminal — router policy observability |
| `Cognitive router picked WEB but internet tool is disabled` | Web veto (also in llm_debug if router logs there) |
| `[Discourse]` | Follow-up classification |
| `[SidecarTelemetry]` | Sidecar task detail (`QUBE_SIDECAR_DEBUG=1`) |
| `retrieval_outcome` | Per-turn JSONL block: router vs final route, hits, downgrade, sidecar rewrite |
| `[PromptLayout]` | Rendered layout + roles per turn |

---

## Module map

| Module | Role |
|--------|------|
| `core/logging_bootstrap.py` | Attaches file sinks at boot |
| `core/app_log_sink.py` | `~/.qube/logs/qube.log` rotating handler (general `Qube.*`) |
| `core/llm_debug_sink.py` | `~/.qube/logs/llm_debug.log` rotating handler |
| `core/routing_debug_sink.py` | `~/.qube/logs/routing_debug.log` rotating handler |
| `core/native_llm_debug.py` | Prompt reconstruction logging |
| `core/completion_output_trace.py` | Raw vs presented completion JSON |
| `core/prompt_integrity_validator.py` | Prompt validation + LM Studio parity |
| `core/native_token_trace.py` | Token trace JSON |
| `core/llm_execution_causality.py` | Causality report JSON |
| `core/llm_counterfactual.py` | Counterfactual simulation JSON |
| `core/engine_input_trace.py` | Engine-boundary prompt capture |
| `mcp/routing_debug.py` | Routing record buffer + file serialization |
| `workers/llm_worker.py` | Turn orchestration, completion trace stash, routing persist |
| `workers/native_llama_engine.py` | Post-inference debug hooks |
| `tools/view_llm_logs.py` | Tail/filter LLM debug file |
| `tools/view_routing_logs.py` | Tail/filter routing debug file |
| `tools/analyze_routing_outcomes.py` | Summarize joined `retrieval_outcome` telemetry (schema v2) |
| `tools/evaluate_router.py` | Offline router eval against labeled corpus; CSV + regression `run.json` |
| `tools/seed_router_eval_library.py` | Index `eval/fixtures` into isolated LanceDB for automated retrieval eval |
| `tools/llm_prompt_diff.py` | Unified diff for prompt dumps |
| `tools/run_ablation.py` | Offline ablation CLI |

---

## Related docs

- `.cursor/rules/native_engine.mdc` — native inference debug pipeline detail  
- `.cursor/rules/tools-diagnostics.mdc` — CLI tool contracts  
- `docs/memory_manual_qa.md` — manual QA checklist (includes `QUBE_DISCOURSE_DEBUG` scenario)  
- `docs/rag_relevance_and_router_T4_plan.md` — routing downgrade / grep notes  

---

*Last updated to reflect the general application log (`~/.qube/logs/qube.log`) and `QUBE_APP_LOG*` env vars.*
