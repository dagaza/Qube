# Advanced Telemetry

## Common questions

- Where do I see CPU and GPU usage?
- How do I check LLM latency or time-to-first-token?
- What is router intelligence?
- What is the sidecar cognition panel?
- What does the inference stack card show?
- How do I correlate telemetry with a slow chat reply?

## What it is

**Advanced Telemetry** (nav label **Telemetry**) is a diagnostic dashboard for runtime performance and routing transparency. It surfaces live hardware graphs, speech/LLM/TTS latency, native model capability detection, cognitive router statistics, sidecar worker health, and compile-time inference stack details — useful when responses feel slow, VRAM feels tight, or retrieval behaves unexpectedly.

For **how to read each card** (what is measured, what is not, and how to diagnose slow replies), see [Advanced Telemetry — interpreting the dashboard](../faq/advanced-telemetry-interpreting.md). This page lists every control; that FAQ explains Qube-specific behaviour.

This view is for monitoring and troubleshooting; everyday chat does not require opening it. Per-turn timing also appears inline on assistant replies in **Conversations** (**STT**, **TTFT**, **TTS**, **TPS** — **TPS** is chat-only, not on the Telemetry dashboard).

Press **?** next to the page title for a spatial guided tour (`telemetry`). This page summarizes controls in retrieval-friendly prose; the tour walks the layout card-by-card.

## Where to find it

Click **Telemetry** in the left navigation (tachometer icon). The page title reads **Advanced Telemetry**. Press **?** beside the title for the guided tour.

## Also called

telemetry dashboard, performance monitor, routing diagnostics, hardware graphs, advanced telemetry, diagnostics page, pipeline latency, inference stack

## How to…

1. **Watch hardware load** — Review **System Load Timeline (%)** for CPU, RAM, and GPU over the last 60 seconds (legend updates every second).
2. **Check pipeline timing** — Read **Pipeline Latency** for **Whisper STT**, **LLM TTFT**, and **TTS Generation** (values show `-- ms` until a stage completes).
3. **Confirm model capabilities** — Inspect **Native LLM — Model capability** for the loaded Internal Engine model: identity, reasoning support, execution mode, detection confidence, and publisher guidance when available.
4. **Understand routing** — Use **Router Intelligence** for route mix, average retrieval phase latency, memory/RAG route shares, adaptive tuner weights, and rule-based health flags. For what each route means (and **HYBRID** vs **● HYBRID**), see [Cognitive Router — how routing works](../faq/cognitive-router-how-routing-works.md).
5. **Monitor the sidecar** — Check **Sidecar Cognition** for runtime status, queue depth, success rate, foreground p95 latency, query-rewrite effectiveness, and health summary.
6. **Review web discovery** — Check **Web discovery** for privacy tier, DDG budgets, pacing, and discovery health (mirrors **Settings → Privacy & data** and **Settings → Knowledge → Web search discovery**).
7. **Review session integrations** — Check **Session integrations** for MCP/integration capability calls on the **current conversation** (open the chat first). See [Integrations settings](../features/settings/integrations.md).
8. **See compute paths** — Open **Inference stack** for llama.cpp build info, hardware profile heuristics, and which compute path native chat, embeddings, and sidecar use (configuration transparency, not live VRAM or timing).
9. **Correlate with chat** — Compare dashboard stats with per-message **STT** / **TTFT** / **TTS** / **TPS** on assistant replies in **Conversations** (**TPS** only on bubbles). Follow [Advanced Telemetry — interpreting the dashboard](../faq/advanced-telemetry-interpreting.md) for a slow-reply workflow. For grounded answers, open **Sources** on the reply, then **INSPECT RETRIEVAL** when that button appears (per-turn retrieval trace — not the router summary card).

## Controls

Grouped top-to-bottom like the **Advanced Telemetry** layout. Hover metric titles or the **ⓘ** info buttons on rows for longer measurement notes.

### Page header

| Control | What it does |
|---------|----------------|
| **Advanced Telemetry** | Page title |
| **?** (guided tour) | Starts the Advanced Telemetry tour |

### System Load Timeline (%) (hardware card)

| Element | What it shows |
|---------|----------------|
| **System Load Timeline (%)** | Section header — rolling 60-second chart |
| **CPU:** legend | Live processor utilization across all cores |
| **RAM:** legend | Live system memory in use |
| **GPU:** legend | Live graphics processor **compute** utilization (not VRAM allocated) |
| Load chart | CPU (green), RAM (blue), and GPU (purple) lines over the last minute; chart is read-only (no pan/zoom) |

### Pipeline Latency

| Metric | Subtext | Value |
|--------|---------|-------|
| **Whisper STT** | Voice-to-Text inference time | `-- ms` until measured |
| **LLM TTFT** | Time To First Token | `-- ms` until measured — last completed stream; also on chat bubbles |
| **TTS Generation** | Text-to-Speech synthesis time | `-- ms` until measured |

### Native LLM — Model capability

Populated when an Internal Engine (native) model is loaded; fields show **—** when unavailable.

| Metric | Subtext |
|--------|---------|
| **Model** | Loaded native model identity |
| **Reasoning-capable** | Thinking token capability |
| **Execution mode** | Resolved policy execution mode |
| **Confidence** | Model capability classification confidence |
| **Publisher guidance** | README/curated publisher contract when present |

### Router Intelligence

| Metric | Subtext |
|--------|---------|
| **Routes** | Current route distribution |
| **Avg retrieval phase** | Mean retrieval latency across turns (before token streaming; rolling ~200 turns) |
| **MEMORY route share** | Portion of turns routed to memory |
| **RAG route share** | Portion of turns routed to RAG |
| **Tuner weights** | Adaptive router weight state (hybrid / memory / rag sensitivities) |
| **System health** | Rule-based router health summary |

### Sidecar Cognition

| Metric | Subtext |
|--------|---------|
| **Status** | Runtime availability (online / degraded / disabled) |
| **Queue depth** | Pending sidecar jobs |
| **Success rate** | Completed sidecar calls vs attempts |
| **Foreground p95** | Rewrite + digest latency (95th percentile) |
| **Query rewrite** | Assistive follow-up expansion applied vs attempted |
| **System health** | Rule-based sidecar health summary |

### Web discovery

Live web search discovery policy (R10). Refreshes about once per second while Telemetry is open. Same underlying settings as **Settings → Knowledge → Web search discovery**.

| Metric | Subtext |
|--------|---------|
| **Privacy tier** | Active SERP discovery tier |
| **Primary provider** | Current primary route; includes DDG backoff text when paused |
| **DDG burst budget** | Live DuckDuckGo calls in burst window (cache hits excluded) |
| **DDG session budget** | Live DuckDuckGo calls in session window |
| **Pacing** | Minimum gap between live DDG queries |
| **System health** | Stable vs budget exhausted vs backoff vs conservative pacing |

For session privacy review without JSONL, see [Audit session privacy](../faq/audit-session-privacy.md).

### Session integrations

Lists **integration capability** calls recorded for the **active Conversations session** (MCP and other capability providers). Refreshes when you open Telemetry with a chat selected.

| Element | What it shows |
|---------|----------------|
| **Session integrations** | Section title |
| Summary lines | Provider/namespace, capability group, tier, allowed/denied status |
| Raw tool id | Shown when **Advanced** settings are unlocked (same rule as integration egress formatting) |

Configure servers under **Settings → Knowledge → Custom sources**; grant permissions under **Settings → Integrations**. See [Connect an MCP server](../workflows/connect-mcp-server.md).

### Inference stack

Does not measure VRAM usage or timing — shows compile-time backend and configured compute paths.

| Metric | Subtext |
|--------|---------|
| **llama.cpp build** | Wheel backend and GPU offload support |
| **Hardware profile** | GPU memory kind and layer cap heuristics |
| **Native chat** | Loaded model and requested GPU layers |
| **Embeddings** | RAG embedder compute path |
| **Sidecar** | Auxiliary cognition compute path (CPU) |

### LLM debug log (developer only)

When the environment variable **`QUBE_LLM_LOG_UI=1`** is set before launch, an **LLM debug log (developer — QUBE_LLM_LOG_UI)** panel appears below the cards and tails `~/.qube/logs/llm_debug.log`. This is not shown in normal installs.

## Related

- [Advanced Telemetry — interpreting the dashboard](../faq/advanced-telemetry-interpreting.md) — educative guide (TTFT, GPU vs VRAM, router, diagnose slow replies)
- [Cognitive Router — how routing works](../faq/cognitive-router-how-routing-works.md) — pathways, overrides, Web vs Hybrid Internet Mode
- [AI & Models settings](../features/settings/ai-models.md) — engine mode, GPU layers, and hardware knobs
- [Knowledge settings](../features/settings/knowledge.md) — retrieval pipeline options
- [Conversations](../features/conversations.md) — per-message timing, **Sources**, and **INSPECT RETRIEVAL**
- [Audit session privacy](../faq/audit-session-privacy.md) — Telemetry + INSPECT session review
- [Integrations settings](../features/settings/integrations.md) — MCP capability permissions
- [Web discovery privacy tiers](../faq/web-discovery-privacy-tiers.md) — tier egress table
- [Model won't load troubleshooting](../troubleshooting/model-wont-load.md) — native engine load failures
