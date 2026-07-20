# Advanced Telemetry — interpreting the dashboard

## Common questions

- What does **LLM TTFT** mean and what is a “good” value?
- Why does **GPU** show 0% even when a model is loaded?
- What is **Router Intelligence** showing?
- How do I tell if slowness is retrieval vs generation?
- What is **INSPECT RETRIEVAL**?

## What Advanced Telemetry is

**Advanced Telemetry** (**Telemetry** in the left nav) is Qube’s **live diagnostic dashboard** — hardware graphs, voice/LLM/TTS timing, routing statistics, sidecar health, and inference-stack transparency. It helps when replies feel slow, GPU memory feels tight, or retrieval behaves unexpectedly.

It is **not** required for everyday chat. Per-turn labels (**STT**, **TTFT**, **TTS**, **TPS**) also appear on assistant messages in **Conversations**.

## What Telemetry does not measure

| You might expect | What the dashboard actually shows |
|------------------|-----------------------------------|
| VRAM used (GB) | **GPU %** = compute utilization, not memory allocated |
| Token counts per turn | **TPS** appears on **chat bubbles**, not on the Telemetry page |
| Full routing trace | Summary stats only — use **Sources → INSPECT RETRIEVAL** on a reply |
| Historical logs (days) | **Rolling in-memory windows** (see below); resets when Qube exits |

For VRAM tuning, use **Settings → AI & Models → Inference stack** / **Hardware tuning** and [Hardware tuning FAQ](hardware-tuning-internal-engine.md).

## Where to find it

Click **Telemetry** (tachometer icon). Page title: **Advanced Telemetry**. Press **?** for the spatial guided tour. Full control names: [Advanced Telemetry feature doc](../features/telemetry.md).

## System Load Timeline (%)

A **60-second rolling chart** refreshed about **once per second**:

| Line | Source | How to read it |
|------|--------|----------------|
| **CPU** (green) | `psutil` — all cores | Spikes during embedding, CPU-only inference, or heavy retrieval |
| **RAM** (blue) | System memory in use | Rises with large **Context limit**, RAG batches, or multiple loaded models |
| **GPU** (purple) | NVML / AMD sysfs / Windows perf | **Compute busy %**, not VRAM free/total |

**Interpretation:** If replies are slow but **GPU stays near 0%**, inference may be CPU-bound — try raising **GPU offload layers** (Internal Engine) or check External Server host settings. If **RAM pegs high**, lower **Context limit**, **Chat history**, or model quant size.

The chart is read-only (no pan/zoom) so page scrolling is not blocked.

## Pipeline Latency

Shows the **most recently completed** timing for each stage (`-- ms` until measured):

| Metric | What Qube measures |
|--------|-------------------|
| **Whisper STT** | Wall-clock from entering transcription to final text in **STTWorker** |
| **LLM TTFT** | Wall-clock from stream request start to **first emitted token** in **LLMWorker** |
| **TTS Generation** | Wall-clock from sentence synth start to first playable PCM in **TTSWorker** |

**TTFT** is the usual “how long until text starts appearing” metric. High TTFT often means:

- Large prompt (long **Chat history**, big Library retrieval, many `@` attachments)
- **Internal Engine** cold load or model reload after **Context limit** / **GPU layers** change
- **External Server** queueing or an undersized host context window (see [Internal vs External FAQ](internal-engine-vs-external-server.md))
- CPU-only or partial GPU offload

The same **TTFT** value is mirrored on the latest assistant reply in **Conversations** (shown in seconds there). **TPS** (tokens per second) appears **only on chat bubbles**, estimated from first token to stream end — not on this card.

## Native LLM — Model capability

Populated when an **Internal Engine** `.gguf` is loaded:

| Field | Meaning |
|-------|---------|
| **Model** | Loaded file identity (basename + detected profile name when they differ) |
| **Reasoning-capable** | Whether Qube detected thinking-token support for **Think** mode |
| **Execution mode** | Resolved policy / profile execution mode |
| **Confidence** | Classifier confidence for capability detection |
| **Publisher guidance** | README-derived publisher contract when available |

When **External Server** is selected or no native model is loaded, fields show **—** (engine mode may still appear). This card does **not** describe the remote LM Studio/Ollama model.

## Router Intelligence

Summarizes the **last ~200 routed chat turns** in memory (`RouterTelemetryBrain`). Resets when Qube restarts. Shows **⚪ Idle** until you send messages that trigger routing.

| Metric | Meaning in Qube |
|--------|-----------------|
| **Routes** | Counts by **execution route** (for example `MEMORY`, `RAG`, `HYBRID`, `WEB`, `NONE`) |
| **Avg retrieval phase** | Mean milliseconds from route start through Memory/RAG/web retrieval assembly **before token streaming** |
| **MEMORY route share** | Turns routed with Memory retrieval / total sampled turns |
| **RAG route share** | Turns routed with Library RAG / total sampled turns |
| **Tuner weights** | Live `h` / `m` / `r` sensitivities from **AdaptiveRouterSelfTunerV2** (clamped ~0.4–2.0) |
| **System health** | Rule-based flags — for example HYBRID overuse (>60%), weak memory weight (<0.6), high avg latency (>1200 ms) |

**Interpretation:** High **Avg retrieval phase** with normal **TTFT** suggests retrieval/indexing slowness — check **Settings → Knowledge** search models and Library size. High **TTFT** with low retrieval latency points to **generation** (model size, GPU layers, External host). **HYBRID**-heavy counts here mean **Memory + Library RAG** on recent turns — **not** the **● HYBRID** dot (**Hybrid Internet Mode** / auto web). See [Cognitive Router — how routing works](cognitive-router-how-routing-works.md).

## Sidecar Cognition

The **auxiliary CPU cognition worker** (background titling, contradiction judge, **query rewrite**, **source digest**, etc.). Sidecar tasks **do not change routing**; they add assistive processing.

| Metric | Meaning |
|--------|---------|
| **Status** | Online / degraded / disabled (+ custom model basename when not bundled default) |
| **Queue depth** | Pending jobs on **SidecarLlmWorker** |
| **Success rate** | Completed inference attempts / tries (queue deferrals excluded from denominator when shown) |
| **Foreground p95** | 95th percentile latency for **query rewrite** and **source digest** |
| **Query rewrite** | Applied / attempted on discourse follow-up turns (confidence-gated) |
| **System health** | Rule-based summary from queue, failures, and foreground latency |

Elevated **Queue depth** or **Foreground p95** can add slight delay before the main LLM turn is fully prepared on follow-ups, but main **TTFT** still reflects the primary chat model stream.

## Inference stack

**Configuration transparency**, not live timing or VRAM metering (same family of data as **Settings → AI & Models → Inference stack**):

| Row | Shows |
|-----|-------|
| **llama.cpp build** | Wheel backend, GPU offload support flag, package version |
| **Hardware profile** | Detected GPU memory kind, budget heuristic, safe layer cap |
| **Native chat** | Loaded model depth vs **requested GPU layers** (not measured runtime offload) |
| **Embeddings** | Library embedder GPU/CPU path |
| **Sidecar** | Always **CPU** (`n_gpu_layers=0`) when loaded |

Use this card to confirm Qube **thinks** it configured — cross-check against [Hardware tuning FAQ](hardware-tuning-internal-engine.md).

## Diagnose a slow reply (workflow)

1. **Conversations** — read **TTFT** and **TPS** on the slow assistant message.
2. **Telemetry → Pipeline Latency** — confirm whether **LLM TTFT** or **Whisper STT** / **TTS** dominated.
3. **System Load Timeline** — check CPU/RAM/GPU during the same period.
4. **Router Intelligence → Avg retrieval phase** — if high, retrieval/indexing likely contributed.
5. **Sources** on the reply → **INSPECT RETRIEVAL** (when shown) for adapter timing, preset, and trace detail for that turn.
6. **Generation / hardware settings** — [Generation parameters](generation-parameters.md), [Hardware tuning](hardware-tuning-internal-engine.md), or External host context if TTFT stays high with low retrieval latency.

## INSPECT RETRIEVAL

On assistant replies with retrieved evidence, open **Sources**, then **INSPECT RETRIEVAL** (when present). Opens the retrieval inspector with the stored bundle trace — adapters used, preset id, and phase detail for **that turn**. Telemetry’s router card is session-wide aggregate; the inspector is per-reply forensics.

## Developer panel (optional)

When **`QUBE_LLM_LOG_UI=1`** is set before launch, an **LLM debug log** panel tails `~/.qube/logs/llm_debug.log`. Normal installs do not show this.

## Also called

telemetry interpretation, TTFT meaning, router intelligence explained, pipeline latency, diagnose slow chat, retrieval latency, GPU telemetry

## Related

- [Advanced Telemetry feature doc](../features/telemetry.md) — full control catalog
- [Conversations](../features/conversations.md) — per-message **STT** / **TTFT** / **TTS** / **TPS**
- [Generation parameters FAQ](generation-parameters.md) — context and reply caps affecting TTFT
- [Hardware tuning FAQ](hardware-tuning-internal-engine.md) — GPU layers and VRAM
- [Internal engine vs external server](internal-engine-vs-external-server.md) — External context quirks
- [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) — route vocabulary and HYBRID naming
