# Competitive landscape

**Purpose:** Position Qube fairly against common local-AI tools. Use when refining the README, landing page, or launch messaging.  
**Last updated:** July 2026  
**Not a bash piece** — each project excels at different jobs.

---

## One-line positioning

| Product | What it optimizes for |
|---------|----------------------|
| **Qube** | Voice-first **desktop assistant** — one app for chat, Library, curated memory, and cited research |
| **LM Studio** | **Model runner** — discover, load, and serve GGUF/MLX models (GUI + API + MCP) |
| **SillyTavern** | **LLM frontend for power users** — prompts, lorebooks, characters, extensions |
| **Odysseus** | **Self-hosted AI workspace** — chat, agents, email, calendar, research in Docker |

Qube can use **LM Studio or Ollama as its inference backend** while still providing routing, voice, Library, and memory that those tools do not bundle as a single assistant experience.

---

## Feature matrix

Legend: **●** Strong · **◐** Partial · **○** Weak / not a focus · **—** Not applicable

| Capability | Qube | LM Studio | SillyTavern | Odysseus |
|------------|:----:|:---------:|:-----------:|:--------:|
| **Native local inference (GGUF)** | ● | ● | — | ◐ |
| **Use external OpenAI-compatible server** | ● | ● (is server) | ● | ● |
| **Voice pipeline (STT + TTS + wake word + barge-in)** | ● | ○ | ◐ | ○ |
| **Unified document Library (ingest + search + preview)** | ● | ◐ | ◐ | ● |
| **Automatic turn routing** (memory vs Library vs web vs chat) | ● | ○ | ○ | ◐ |
| **Editable long-term memory** (review / edit / delete) | ● | ○ | ◐ | ● |
| **Institutional Live Sources** (SEC, case law, PubMed, etc.) | ● | ◐ | ○ | ◐ |
| **Per-turn composer `@` tools** (`@library`, `@evidence`, …) | ● | ○ | ○ | ◐ |
| **Custom knowledge presets** (`@[tool:user:…]`) | ● | ○ | ○ | ◐ |
| **Reasoning skills** (orthogonal to retrieval routing) | ● | ○ | ○ | ● |
| **Async deep research reports** | ● | ○ | ○ | ● |
| **In-app help as searchable Library corpus + `@help`** | ● | ○ | ○ | ○ |
| **Built-in Hugging Face Model Manager** | ● | ● | ○ | ● |
| **Desktop Companion (floating voice orb)** | ● | — | — | — |
| **MCP tool servers** | ◐ | ● | ◐ | ● |
| **Autonomous agents (bash, multi-step plans)** | ◐ | ◐ | ◐ | ● |
| **Roleplay / lorebooks / character cards** | — | — | ● | ○ |
| **Image generation integration** | — | ○ | ● | ◐ |
| **Email / calendar / notes workspace** | — | — | ○ | ● |
| **Multi-user / Docker-first deploy** | ○ | ◐ | ◐ | ● |
| **Native desktop shell (not browser-only UI)** | ● | ● | ○ | ○ |
| **First-run bundled stack** (sidecar + embed + voice + optional main LLM) | ● | ○ | — | ○ |
| **Spatial guided tours in-app (`?` on each screen)** | ● | ○ | ◐ | ◐ |
| **Searchable in-app help corpus (not just external docs)** | ● | ○ | ○ | ○ |
| **Native UI shell (PyQt — not Electron / browser tab)** | ● | ○ | ◐ | ○ |
| **In-process inference path (internal engine, no chat HTTP hop)** | ● | ◐ | — | ◐ |
| **Local observability dashboard (no cloud analytics)** | ● | ○ | ○ | ○ |
| **Per-turn retrieval inspector + opt-in audit logs** | ● | ○ | ○ | ◐ |
| **Privacy-tiered web discovery (DDG default, BYO SearXNG)** | ● | ○ | ◐ | ● |
| **Shipped institutional Live Source adapters (58+)** | ● | ○ | ○ | ◐ |
| **User custom sources / presets (`@[tool:user:…]`)** | ● | ○ | ◐ | ● |
| **Automatic fact extraction from chat (no manual lore authoring)** | ● | ◐ | ○ | ● |
| **Dedicated memory editor (browse / edit / delete / export)** | ● | ◐ | ○ | ◐ |
| **Delete blocks re-extraction (negative list)** | ● | ◐ | ○ | ○ |

**How to read this:** SillyTavern and Odysseus are not “worse” — they target different primary workflows. LM Studio is the natural **engine** many Qube users already run; Qube is the **assistant layer** on top.

---

## Verified: `@` composer tools

**None of the three competitors ship Qube-style `@` composer routing** in the chat input. This was checked against official docs and READMEs (July 2026).

| Product | How tools / retrieval are chosen | Equivalent to `@library` per message? |
|---------|----------------------------------|--------------------------------------|
| **Qube** | User attaches **`@[tool:…]`**, **`@[file:…]`**, **`@[skill:…]`** in the composer; cognitive router fills gaps | **Yes** — explicit, visible tokens |
| **LM Studio** | **MCP** servers + model **function calling** via API; chat UI document RAG is thread-level, not `@`-token routing ([LM Studio MCP docs](https://lmstudio.ai/docs/developer/core/mcp)) | **No** — model/API-driven or manual doc attach |
| **SillyTavern** | **Slash commands** (`/…`), **World Info** keyword lorebooks, **Data Bank** + Vector Storage extension (enable + configure embedding source) ([Data Bank docs](https://docs.sillytavern.app/usage/core-concepts/data-bank/)) | **No** — prompt injection / extension config |
| **Odysseus** | **Agent tool toggles** (bash, files, web, memory); **slash command** subsystem in codebase; chat composer tours via UI walkthrough — not `@` tokens in messages ([Odysseus README](https://github.com/pewdiepie-archdaemon/odysseus)) | **No** — session/agent-level tools |

**Takeaway:** Qube’s `@` palette is closer to “attach a capability to *this* message” than to MCP tool lists, ST slash commands, or Odysseus agent switches. That is a real product difference, not marketing.

---

## Verified: in-app assisted help

| Product | In-app help | Searchable from chat? | Guided UI tours |
|---------|-------------|----------------------|-----------------|
| **Qube** | **Library → Qube** corpus + **`@[tool:help]`** retrieval | **Yes** | **`?` buttons** on Conversations, Library, Settings, etc. |
| **LM Studio** | External [lmstudio.ai/docs](https://lmstudio.ai/docs/app) | No | First-run onboarding wizard only |
| **SillyTavern** | External [docs.sillytavern.app](https://docs.sillytavern.app/) linked from Welcome screen | No | Onboarding (persona + API setup); **Force onboarding** in settings — not per-screen spatial tours |
| **Odysseus** | External [setup guide](https://github.com/pewdiepie-archdaemon/odysseus/blob/main/docs/setup.md) + landing-page hover demo | No | Slash-command **tours** in app (community docs mention `/tour`-style flows); not a unified `@help` corpus |

**Takeaway:** Qube treats help as **product data** (same Library/embed pipeline as user docs). Competitors rely on **external wikis** and community docs — fine for enthusiasts who already search GitHub, weaker for “how do I turn on GPU layers?” inside the app.

---

## Verified: onboarding & “enthusiast vs approachable”

Your read is **directionally right**, with nuance per product.

### LM Studio — approachable for **running a chat model**, not for a full assistant

- **Pros:** Polished GUI; first-run **onboarding wizard** can download a **starter LLM** (e.g. DeepSeek R1, Llama small) in a few clicks ([install guides](https://lmstudio.ai/docs/app)).
- **Gap vs Qube:** Does **not** bundle a coordinated stack (sidecar cognition, **embedding preset for Library**, **Whisper STT**, **Kokoro TTS**) in one consent flow. Document RAG exists; voice assistant pipeline does not. MCP and Developer tab assume technical comfort.
- **Audience:** “Easiest path to local LLM” — still **chat + models**, not voice-first assistant.

### SillyTavern — explicitly power-user

- Official vision: *“embracing the steep learning curve as part of the fun”* ([SillyTavern docs](https://docs.sillytavern.app/)).
- Quick Start assumes you **install Node app**, open **API Connections**, pick **AI Horde or OpenAI**, select models — **no local model bundle** ([Quick Start](https://docs.sillytavern.app/usage/quick-start/)).
- RAG: enable **Vector Storage** extension, configure **embedding provider**, attach files — multiple manual steps ([Data Bank](https://docs.sillytavern.app/usage/core-concepts/data-bank/)).
- **Audience:** Prompt engineers, roleplay, extension tinkerers — not “install and talk to your PDFs.”

### Odysseus — powerful workspace, **operator mindset**

- **Docker Compose** or native Python venv + **connect Ollama/API yourself** ([README](https://github.com/pewdiepie-archdaemon/odysseus)).
- **Cookbook** gives hardware-aware model picks (similar *idea* to Qube Model Manager fit hints) — but you still wire providers and services.
- Breadth (email, calendar, agents) rewards users who want a **homelab control panel**, not a single-purpose assistant installer.
- **Audience:** Self-hosters comfortable with containers, logs, and multi-panel UI.

### Qube — first-run **Recommended preset** (verified in `core/bootstrap_manifest.py`)

On first launch, Qube’s consent dialog offers a **Recommended** bundle (with disk/RAM feasibility checks), not “go find models yourself”:

| Component | Default in Recommended | Role |
|-----------|------------------------|------|
| Qwen 1.7B sidecar | Required (locked) | Titles, cognition, core features |
| Balanced search preset | Required (locked) | **Embeddings** for Library + memory retrieval |
| Whisper Small | Recommended | **STT** / voice input |
| Kokoro TTS | Recommended | **Voice output** |
| Qwen 3.5 9B (or alternates) | Recommended main LLM | **Conversation** on ~16 GB RAM |

Optional **Advanced** path exposes alternates (Nemotron Nano, Gemma, lighter sidecar). Chat weights are chosen in consent — not a blank shell.

**Honest caveat:** Qube still expects **16 GB RAM** for a good experience; bootstrap warns on tight disk/memory. It is “approachable” relative to **assemble-your-stack** workflows, not relative to **cloud ChatGPT zero-setup**.

---

## User-friendliness summary

| Dimension | Qube | LM Studio | SillyTavern | Odysseus |
|-----------|------|-----------|-------------|----------|
| **Stated audience** | Desktop assistant user | Local LLM user / developer | Power user / hobbyist | Self-hosted workspace operator |
| **Setup mental model** | One app, one consent, phased downloads | Download app → pick chat model | Install → connect API → configure prompts | Deploy stack → connect brain → explore panels |
| **Docs when stuck** | `@help` + tours | Website docs | Website + Discord | Setup guide + slash tours |
| **Best if you…** | Want voice + Library + memory without wiring 5 tools | Want the best model browser / API server | Want maximum prompt/character control | Want agents + email + research in one homelab hub |

---

## Verified: runtime, UI shell, and RAM

This section checks the claim that Qube’s **Python-native desktop stack** is lighter and more efficient than competitors that wrap the UI in **Electron** or a **browser tab**. Investigated July 2026.

### What Qube actually is (precise wording)

| Claim | Verdict |
|-------|---------|
| “100% Python app” | **Overstated.** Qube is **Python-led** with a **PyQt6** native UI and **native extensions**: `llama-cpp-python` (GGUF inference), Faster-Whisper, Kokoro-ONNX, LanceDB, OpenWakeWord. Python orchestrates; heavy work runs in compiled code. |
| “No APIs” | **Overstated for the whole product.** In **Internal Engine** mode, chat inference runs on a **dedicated worker thread** with a queue handoff to the UI — **no localhost HTTP round-trip** for generation. **External Server** mode deliberately talks to LM Studio / Ollama over OpenAI-compatible **localhost HTTP**. Optional dev tooling includes a small FastAPI surface for model-capability probes — not the main chat path. |
| “Faster inference because Python” | **Mostly false.** Token speed is dominated by **llama.cpp / GPU**, not UI language. LM Studio and Qube both lean on llama.cpp for GGUF. Qube’s win is **RAM budget and UI integration**, not magically faster matmul. |
| “Electron competitors eat RAM just for the UI” | **Directionally true for LM Studio; mixed for others** (see below). |

Sources: [Qube architecture stack](../architecture/stack.md), [pipeline and UI](../architecture/pipeline-and-ui.md), LM Studio [NamuWiki / stack notes](https://en.namu.wiki/w/LM%20Studio), [SillyTavern README](https://github.com/SillyTavern/SillyTavern), [Odysseus README](https://github.com/pewdiepie-archdaemon/odysseus).

### UI/runtime stack comparison

| Product | UI layer | Inference / server | Typical process picture |
|---------|----------|-------------------|-------------------------|
| **Qube** | **PyQt6** native widgets (frameless desktop shell) | **Internal:** `llama-cpp-python` in-process worker thread. **External:** localhost OpenAI API to another host. | One main desktop process (+ optional **Desktop Companion** orb). Workers on `QThread`s — UI stays responsive without bundling Chromium for the shell. |
| **LM Studio** | **Electron** + React/TypeScript ([confirmed](https://en.namu.wiki/w/LM%20Studio)) | Bundled **llama.cpp** / MLX engines; v0.4+ **`llmster`** daemon can run headless without GUI | **Multi-process Electron app** (main + renderer + GPU helpers). Industry rule of thumb: **~150–300 MB idle** baseline for minimal Electron apps before your model weights load ([performance guides](https://emadibrahim.com/electron-guide/performance)). LM Studio has had real-world macOS GPU/window-server issues tied to Electron version lag ([bug tracker #1119](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1119)). |
| **SillyTavern** | **Browser SPA** served by **Node.js/Express** on `:8000` ([install docs](https://docs.sillytavern.app/installation/)) — **not** an official Electron app | **None** — proxy to external backends (Horde, OpenAI, Ollama, Kobold, …) | **Node server + your browser** (Chrome/Edge/Firefox). ST itself is lightweight on CPU, but you often pay **Chromium again** in the browser tab. Community wrappers like [EazySillyTavern](https://github.com/yuman07/EazySillyTavern) add **Electron + bundled Node + ST** — the heaviest combo. RAG extensions can push Node heap to **multi-GB RSS** under load. |
| **Odysseus** | **Vanilla JS + HTML/CSS** in the **browser** ([repo languages](https://github.com/pewdiepie-archdaemon/odysseus)) | **FastAPI** (Python) + SQLite + ChromaDB; models via Ollama/llama.cpp/vLLM | **Python API server + browser tab** (or Docker stack with extra services). Backend is Python like Qube, but the **UI is still web**, not a native desktop shell. |

### Where the RAM actually goes on a 16 GB machine

On a typical local-AI setup, **the chat model weights dominate** (often 4–8+ GB for a 9B quant). UI shell differences matter at the margin — but margins matter when you are already tight:

| Layer | Qube (internal) | LM Studio | SillyTavern + Chrome | Odysseus + browser |
|-------|-----------------|-----------|----------------------|-------------------|
| UI shell (no model loaded) | PyQt6 — typically **tens to low hundreds of MB** for the app process | Electron baseline **~150–300+ MB** before chat | Node **+** browser tab — often **similar or worse** than single native app | FastAPI Python **+** browser tab |
| Embeddings / sidecar / voice | **Bundled in bootstrap** (sidecar ~1.4 GB, embed ~130 MB, voice ~900 MB) | Optional embed models; no full voice loop | Via extensions — extra config | Via Chroma + configured providers |
| Main LLM | Same GGUF math as LM Studio when both use llama.cpp | Same | External process (Ollama, etc.) | External or configured host |

**Takeaway:** Qube is designed around a **strict on-device memory budget** (documented in [system requirements](system-requirements.md): ~10–15 GB usable for models + context on 16 GB). A **native PyQt shell** keeps more of that budget for **weights and context**, especially vs **Electron-wrapped** LM Studio. That is a **credible differentiator** — not “Python is faster,” but **“we did not spend hundreds of MB on Chromium for the settings panel.”**

### What Qube should *not* claim in marketing

- ❌ “100% Python” — native inference and Qt are core to the product.
- ❌ “No APIs ever” — External Server mode and optional localhost integrations exist by design.
- ❌ “Always faster tokens than LM Studio” — same backends, same physics; compare on your hardware with the same quant.
- ❌ “SillyTavern is Electron” — default ST is **Node + browser**; only third-party bundles add Electron.

### What *is* fair to say

- ✅ **Native desktop shell (PyQt6), not an Electron wrapper** — aligns with Qube’s RAM-first positioning.
- ✅ **Internal Engine keeps chat on an in-process worker thread** — fewer moving parts than browser UI → Node → HTTP → Ollama → back.
- ✅ **Single assistant binary** — voice, Library embeddings, memory, routing, and chat in one install (vs ST’s interface-only + separate inference app).
- ✅ **LM Studio users can still pair with Qube** — External Server mode uses LM Studio as the engine while Qube supplies the leaner assistant UI layer.

---

## Verified: observability, transparency, and privacy certainty

**Important naming note:** In Qube, **“Telemetry”** means a **local diagnostic dashboard** (CPU/RAM/GPU, router stats, sidecar health) — **not** cloud analytics. There is **no vendor usage-reporting pipeline** in the open-source tree (no PostHog/Sentry-style product telemetry found in code review).

### What Qube gives users to *see* and *verify*

| Surface | Where | What you learn | Leaves your machine? |
|---------|-------|----------------|----------------------|
| **Advanced Telemetry** | Left nav **Telemetry** | Live hardware graph, **Pipeline Latency** (STT/TTFT/TTS), **Router Intelligence**, **Sidecar Cognition**, **Inference stack** card | **No** — in-memory rolling windows; cleared on exit |
| **Per-message timing** | **Conversations** bubbles | **STT**, **TTFT**, **TTS**, **TPS** on assistant replies | **No** |
| **Sources + INSPECT RETRIEVAL** | Reply **Sources** panel | Which adapters/preset ran, phase timing, discovery provider for **that turn** | **No** — stored with the message bundle |
| **Diagnostic logs (5 files)** | **Settings → Advanced** | Rotating logs: app, LLM debug, routing JSONL, web-search audit, skills debug — each with **recording toggle**, in-app viewer, **Clear log** | **No** — `~/.qube/logs/` (or `%LOCALAPPDATA%\Qube\logs\`) |
| **Knowledge → Diagnostics** | **Settings → Knowledge** | **Last retrieval trace** (from web-search audit JSONL) | **No** |
| **Provider status** | **Settings → Knowledge → Source status** | Per-adapter health, quota hints, circuit-breaker state | **No** — reflects local HTTP client metrics |
| **Privacy-tier labels** | **Settings → Knowledge → Web search discovery** | Which discovery path is active (Private / Balanced / SearXNG / …) | N/A — configuration transparency |

**Learning path:** `@help` articles cover [Telemetry](../assets/help/en/features/telemetry.md), [diagnostic logs](../assets/help/en/faq/diagnostic-logs-advanced-settings.md), and [interpreting router behaviour](../assets/help/en/faq/cognitive-router-how-routing-works.md). **`?` tours** exist on Telemetry and Settings sections.

**Before sharing logs:** Web-search and routing logs can contain **full query text** unless you launch with redact flags (`QUBE_WEB_SEARCH_AUDIT_REDACT=1`, `QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY=1`). LLM debug logs may include **prompt excerpts** — documented in help, not hidden.

**Open-source auditability:** Unlike closed-source runners, a skeptical user can grep the repo and run Wireshark to confirm Qube is not phoning home with chat content. **Network traffic you should expect when *you* opt in:** Hugging Face model downloads, Live Source adapter HTTP, DuckDuckGo/SearXNG/Brave discovery, and optional external LLM servers — all user-configured paths, not silent exfiltration of chat history.

### Competitor comparison (observability & privacy posture)

| Dimension | Qube | LM Studio | SillyTavern | Odysseus |
|-----------|------|-----------|-------------|----------|
| **Product “telemetry”** | Local dashboard only | [Privacy policy](https://www.lmstudio.ai/app-privacy): **no chat/document exfil**; update/model-hub requests only. Third-party audits note possible **opt-out usage stats** in some builds — **closed source**, so verify with firewall/Wireshark | No dashboard; Node proxy logs optional; **you** choose backends | Claims **no telemetry**; open source; Docker logs |
| **Per-turn retrieval forensics** | **INSPECT RETRIEVAL** + optional JSONL audit | Document RAG; less turn-level adapter provenance in UI | Extension-dependent; no first-class inspector | Agent/tool traces; research reports — different shape |
| **Audit logs on disk** | Five toggled log files + env redaction | Developer/API focus; less user-facing log UX | Filesystem chat logs; server console | SQLite + service logs in Docker |
| **Code auditability** | **Open source (AGPL)** | Proprietary binary | Open source (AGPL) | Open source |

**Honest caveat:** Qube’s depth of observability is a **double-edged sword** — power users get certainty; casual users must still **read privacy tiers** before enabling `@internet` / Hybrid Internet Mode. Default web-discovery tier is **`private`** (DuckDuckGo HTML + Wikipedia — no API-key SERP vendors).

---

## Verified: Live Sources, custom adapters, and anonymous search

### Bundled institutional adapters (Qube-specific breadth)

Qube ships **58 live knowledge adapters** across scientific, finance, and legal domains ([inventory](../live_knowledge_adapters.md)) — PubMed, arXiv, SEC EDGAR, CourtListener, Crossref, OpenAlex, and dozens more — with:

- **Per-adapter toggles** in **Settings → Knowledge → Live sources**
- **Anonymous vs key-required** modes documented per adapter
- **Provider credentials** UI + **Source status** health/quota panel
- Shared **HTTP resilience** (rate limits, circuit breakers, negative cache)

LM Studio can reach similar endpoints **via MCP** if you install/configure servers. SillyTavern relies on **extensions** and external APIs. Odysseus has research tools and MCP — **less curated institutional catalog out of the box**.

### User-built tools: presets, custom sources, connectors

| Mechanism | What it does |
|-----------|--------------|
| **My knowledge presets** | Bundle adapter ids or web-fetch domains → appear as **`@[tool:user:…]`** in the composer ([workflow](../assets/help/en/workflows/create-knowledge-preset.md)) |
| **Custom sources** | **REST JSON**, **GraphQL**, **RSS/Atom**, **SQLite**, **PostgreSQL**, **filesystem**, **MCP** connectors — define base URL, search path, test connection ([Settings → Knowledge](../assets/help/en/features/settings/knowledge.md)) |
| **Knowledge pack import/export** | Backup/transfer presets + custom sources between installs |
| **Composer `@` tokens** | Explicit per-turn routing to `@evidence`, `@finance`, `@legal`, `@internet`, etc. |

**Extension model comparison:** Odysseus and LM Studio lead on **general MCP server ecosystems**. Qube’s angle is **curated Live Sources + user presets/custom connectors** wired into the same citation-oriented retrieval pipeline — not “install any MCP and figure out prompts yourself.”

### Web search privacy: DuckDuckGo, tiers, and bring-your-own SearXNG

General-web discovery (`@internet`, Hybrid Internet Mode) follows a **privacy-first policy** ([design doc](../web_discovery_privacy_resilience_plan.md)):

| Privacy tier | Default? | Discovery path |
|--------------|----------|----------------|
| **Private search** | **Yes** | **DuckDuckGo HTML** + **Wikipedia** — no API keys, no third-party SERP vendors |
| **Private + API fallback** | Opt-in | Same primary; **Brave Search API** only when configured and tier allows |
| **Maximum reliability** | Opt-in | Prioritizes configured API fallbacks after DDG blocks |
| **Self-hosted SearXNG** | Opt-in | Queries your **`SearXNG base URL`** when set; falls back like balanced private if unset |

Additional user controls:

- **DDG pacing** — slow-down toggle, session/burst limits, live usage counter, backoff after bot challenges
- **Web search audit log** (opt-in) — records trigger, query, URLs, `retrieval_trace` for **Knowledge → Diagnostics**
- **No silent commercial SERP default** — API vendors require explicit tier + credentials

**vs Odysseus:** Docker deploy **bundles SearXNG** as an internal service — strong default for homelab operators. Qube targets **desktop users** who may point at an existing instance via **Settings → SearXNG base URL** without running a full compose stack.

**vs SillyTavern:** Web search is extension/API-dependent; no equivalent **privacy-tier** UX or DDG pacing policy in core product.

**vs LM Studio:** Web/RAG via chat + MCP; no shipped **58-adapter catalog** or DDG/SearXNG discovery tiers — different product layer.

### What still leaves your machine (be explicit in messaging)

When users enable online features, queries and context snippets go to **the endpoints they chose**:

- DuckDuckGo / your SearXNG / Brave (if configured) for `@internet`
- Each Live Source adapter’s public API (PubMed, SEC, …)
- Optional cloud LLM if user configures External Server beyond localhost

Qube’s transparency win is **making that visible** (Sources, INSPECT RETRIEVAL, privacy tier labels, audit logs) — not pretending the app is air-gapped while Hybrid Internet Mode is on.

---

## Verified: Desktop Companion (floating orb)

**None of the three primary competitors ship an integrated always-on-top desktop companion** like Qube’s **Desktop Companion**. Checked July 2026 against product docs, READMEs, and community threads.

### What Qube ships (in-product)

The **Desktop Companion** is a **native PyQt6 overlay window** — optional, always-on-top “orb” for **quick voice turns** and **glanceable status** (idle / listening / working / speaking) without raising the main window ([help](../assets/help/en/features/settings/desktop-companion.md)).

| Capability | Qube Companion |
|------------|----------------|
| **Integrated with same app** | Same voice pipeline (wake word, PTT, TTS, **barge-in**), routing, and memory as the main window |
| **Always-on-top overlay** | Yes — floats over other apps |
| **Fullscreen suppression** | **Hide during fullscreen apps** (games, slides) |
| **Quick voice vs full chat** | Companion for fast hands-free turns; main window for `@` attachments, Library, long threads ([FAQ](../assets/help/en/faq/companion-vs-main-window.md)) |
| **Commentary / cognition** | Optional captions (**Companion Cognition v2**), personality and frequency controls |
| **Look & feel** | **Sphere** or **Qube** cube styles, idle glow, position snap compass |
| **Platform notes** | Wayland experimental overlay + edge-dock mode ([docs/companion_wayland.md](../companion_wayland.md)) |

### Competitors

| Product | Built-in floating companion? | What exists instead |
|---------|------------------------------|---------------------|
| **LM Studio** | **No** | Main chat window + API. **LM Studio Bionic** (separate app) adds **voice keyboard** dictation into any app — not an always-on-top Qube-style orb ([Bionic blog](https://lmstudio.ai/blog/introducing-lm-studio-bionic)). |
| **SillyTavern** | **No** | **Live2D** extension renders avatars **inside the browser UI only** — not a desktop overlay ([Live2D docs](https://docs.sillytavern.app/extensions/live2d/)). [GitHub Discussion #4996](https://github.com/SillyTavern/SillyTavern/discussions/4996) (Jan 2026): users ask for floating pet mode; **no official ST solution**. |
| **Odysseus** | **No** | **Browser/PWA workspace** dashboard — chat, agents, email, etc. No documented always-on-top orb in the [README](https://github.com/pewdiepie-archdaemon/odysseus). |

### Third-party “desktop companion” ecosystem (not the same as Qube)

Similar *ideas* exist as **separate tools** users wire to LM Studio / Ollama / ST:

| Project | Notes |
|---------|--------|
| [NeuralCompanion](https://github.com/Rakile/NeuralCompanion) | Windows desktop companion; **Companion Orb Overlay** + LM Studio provider — third-party, not LM Studio itself |
| [Bruno](https://github.com/rithulkamesh/bruno) | macOS floating orb; local voice + multi-provider (incl. LM Studio) |
| [Sidekick](https://github.com/ast-ry/sidekick) | macOS screen-aware overlay for LM Studio — prototype |
| [AnySoul Desktop Pet](https://docs.anysoul.ai/guides/desktop-pet/) | Transparent Live2D pet overlay (Electron) — avatar-forward, separate product |
| Commercial pets (e.g. itch.io “Desktop AI Companion”) | Always-on-top overlays with avatars; connect to LM Studio/Ollama via API |

**Takeaway:** A **floating voice orb** is **not unique in the abstract** — the ecosystem has avatar pets and LM Studio add-ons. Qube’s difference is a **first-party, integrated companion** tied to the **same assistant stack** (voice loop, cognitive router, Library, memory, citations) — not “install another overlay and point it at localhost:1234.”

**Honest gaps:**

- Qube’s orb is **minimal UI** — not Live2D/VRM avatar theater. SillyTavern + external pet tools win on **character presentation**.
- Full composer features (`@library`, file attach, long history browsing) still live in the **main window**, not the orb alone.

---

## Verified: long-term memory

Qube treats **Memory** (distilled personal facts) and **Library** (your documents) as **separate pipelines** — a deliberate assistant design. Competitors overlap pieces of this, but rarely with the same editorial stack ([architecture](../architecture/memory-system.md), [Memory Manager help](../assets/help/en/features/memory-manager.md)).

### What Qube’s memory system is

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Library (document RAG)** | LanceDB chunks from PDF/EPUB/TXT/MD | Passage-level citations from **your files** — `@library`, `@file` |
| **Long-term atomic memory (v6–v7.1)** | LanceDB `qube_memory::%` namespace | Short **facts about you** — preferences, projects, knowledge — distilled from chat |

**Automatic capture:** Background **Enrichment Worker** extracts durable facts asynchronously (never blocks chat). Skips bad turns (failures, web errors, stream guard trips). **Explicit remember** (“please remember that…”) bypasses the extractor LLM but still seeds the fact.

**Structural tiers:** `preference` · `knowledge` · `episode` · `context` — filterable in **Memory Manager** with `PREF` / `KNOW` / `EP` / `CTX` badges.

**Quality & trust mechanics:**

| Mechanism | What it does |
|-----------|--------------|
| **Typed schema + server validation** | Rejects assistant failures, thin stubs, missing provenance |
| **Negative list** | Delete in Memory Manager → fact is **blocked from re-extraction** (`~/.qube/memory_negatives.json`) |
| **Contradiction judge** | Duplicate / contradict / complement pairs resolved deterministically + micro-LLM |
| **Self-reflection worker** | Sidecar flags suspect rows → **Flagged for review** (never auto-deletes) |
| **Usage decay + promotion (opt-in)** | Rows that earn retrieval float up; optional context→preference promotion |
| **Provenance + document expansion** | Thin memory can expand to originating Library chunk (“Who is Alice?”) |
| **Cognitive router + `@memory`** | Recall-intent turns auto-route to memory; composer override available |

**User-facing surface:** Dedicated **Memory Manager** nav — edit, flag, delete, export Markdown, bulk delete visible, recurring-themes card, promotion/consolidation sections (when enabled in **Settings → Memory**).

### Competitor comparison

| Dimension | Qube | LM Studio | SillyTavern | Odysseus |
|-----------|------|-----------|-------------|----------|
| **Built-in long-term user memory** | **Yes** — first-class | **No in core app** — document chat RAG only ([docs](https://lmstudio.ai/docs/app)) | **Not core** — lorebooks + extensions | **Yes** — **Brain** (ChromaDB hybrid retrieval) |
| **Memory vs documents** | **Explicit split** — Library ≠ Memory ([FAQ](../assets/help/en/faq/memory-vs-library.md)) | Document RAG in chat threads | Data Bank = files; World Info = keyword lore | Library + Brain separate panels |
| **How facts get in** | **Automatic extraction** + explicit remember | **Plugins** (model must call tools) — e.g. [persistent-memory](https://lmstudio.ai/dirty-data/persistent-memory), [simplememory](https://lmstudio.ai/zeroinn/simplememory) | **Manual lorebooks**; optional [CharMemory](https://github.com/bal-spec/sillytavern-character-memory) extension → Data Bank markdown | **Automatic** persistence into vector store + agent memory tools |
| **Editor UI** | **Memory Manager** — tiers, search, edit/delete/export | Edit plugin DB / markdown file externally | Edit lorebook entries or Data Bank markdown files | **Brain** UI — browse/search memories; less atomic “fact card” editorial model |
| **Delete = stay gone** | **Negative list** blocks re-extraction | Plugin-dependent (many support delete/update tools) | Delete lore/file manually; no unified negative list | Delete varies; no documented Qube-style negative list |
| **Routing integration** | Cognitive router + **`@memory`** + recall-intent detection | Plugin injects context if model calls recall | World Info **keyword** activation; Vector Storage **semantic** chunk retrieval | Hybrid RAG in agent/chat loops |
| **Primary use case** | **Assistant remembers you** — preferences, standing facts | Model runner + optional plugin memory | **Character/lore** continuity, roleplay | **Workspace brain** — broad persistence across agents |

### LM Studio — plugins, not product memory

LM Studio’s core product optimizes **model loading, chat, document RAG, MCP**. Long-term memory arrives via **community plugins** (SQLite, markdown files, MCP servers) that depend on **model tool-calling discipline** — fine for power users, not a guided assistant pipeline. Qube ships extraction, validation, UI, and routing **in the box**.

### SillyTavern — lore and RAG, not Memory Manager

Three different “memory-like” systems, none equivalent to Qube’s atomic store:

1. **World Info / lorebooks** — author keyword→text injections (excellent for **characters**, not “remember I prefer metric units”).
2. **Data Bank + Vector Storage** — **document** RAG ([docs](https://docs.sillytavern.app/usage/core-concepts/data-bank/)); must enable extension + embedding provider.
3. **Chat vectorization** — retrieves **raw past messages**, not distilled facts ([extension docs](https://github.com/SillyTavern/SillyTavern-Docs/blob/main/extensions/Chat-vectorization.md)).

CharMemory and similar extensions close part of the gap by extracting markdown into Data Bank — still **character-scoped**, extension-dependent, and without Qube’s negative list / reflection / tier editor.

### Odysseus — strong persistence, different editorial model

Odysseus **Brain** uses **ChromaDB + hybrid retrieval** for cross-session memory ([reviews](https://www.xda-developers.com/tried-pewdiepie-open-source-ai-workspace-odysseus-weirdly-great/)). **Skills** can evolve; **Library** handles documents. Some forks/commits explore **bi-temporal knowledge graphs** — advanced, still maturing upstream.

**Where Qube differs:** atomic **fact cards** with provenance, tier filters, negative list, reflection flagging, and **`@memory` / router recall-intent** tuned for a **single-user desktop assistant** — not a multi-panel homelab workspace. Odysseus may store **more**; Qube optimizes for **curated, deletable, inspectable** personal facts.

### Fair summary

| Best if you want… | Lean toward… |
|-------------------|--------------|
| Editorial control over **personal facts** with delete-that-sticks | **Qube** |
| **Character lore** and prompt injection craft | **SillyTavern** |
| Memory via **plugins** on your existing LM Studio workflow | **LM Studio + Hub plugins** |
| **Workspace-wide brain** across agents, email, research | **Odysseus** |

**Honest Qube caveats:**

- Memory automation has **many toggles** (enrichment, promotion, consolidation) — powerful but not zero-config.
- **Episodes** (session summaries) and **knowledge** tiers add concepts competitors don’t expose — learning curve.
- Odysseus’s graph/bi-temporal direction may surpass Qube for **temporal “what was true when”** queries as it matures.

---

## Where Qube is genuinely different

### 1. Composer routing (`@` attachments)

In **Conversations**, you attach **`@[tool:…]`**, **`@[file:…]`**, **`@[chat:…]`**, or **`@[skill:…]`** tokens to steer a *single turn* — without rewriting system prompts.

| Token family | Examples | What it does |
|--------------|----------|--------------|
| **Tools** | `@library`, `@evidence`, `@finance`, `@legal`, `@research`, `@internet`, `@memory`, `@help` | Routes retrieval to the right backend for this message |
| **Files** | `@file:my-report.pdf` | Ground on one Library document |
| **Chats** | `@chat:previous-thread` | Pull context from another conversation |
| **Skills** | `@skill:research_synthesis` | Adds reasoning framework *without* changing route |
| **Presets** | `@tool:user:biology` | Your saved Live Source bundles from Settings |

LM Studio chat is one thread + optional document RAG. SillyTavern uses **slash commands** and World Info keyword lore. Odysseus uses **agent tool toggles** and slash commands — not per-message `@` tokens. **Qube’s composer is explicit, per-message, and citation-oriented** ([verified comparison](#verified-composer-tools)).

### 2. Cognitive router (automatic, still overridable)

Qube scores each turn for **Memory**, **Library**, **web**, or **plain chat** before the model replies — then applies relevance gates and empty-source downgrade so the model is not forced to hallucinate citations.

You can still override with `@` attachments, toggles, or trigger phrases. Competitors generally rely on **manual** RAG toggles, **keyword lorebooks**, or **agent tool selection**.

### 3. Library as a first-class corpus

- Ingest **PDF, EPUB, TXT, MD** into folders; preview reconstructed text from the vector index.
- **Chat with document** opens Conversations with `@file` prefilled.
- **Library → Qube** holds the shipped help manual — same pipeline as your docs, searchable via **`@help`** without polluting normal Library search.
- **NLP Auto-Activator** + custom trigger phrases generalize from a few examples (no exact phrase repetition).

LM Studio added document chat; SillyTavern’s Data Bank is per-character/context; Odysseus has document tools — **none combine user Library + embedded product help + composer `@library` in one LanceDB pipeline with route-aware injection.**

### 4. Live Sources beyond “web search”

Built-in adapters for **Wikipedia/trusted**, **scientific literature**, **SEC EDGAR**, **U.S. case law**, and more — with optional **knowledge presets** to bundle domains for repeat research.

LM Studio can reach similar data **via MCP** if you configure servers. Qube ships **curated institutional paths** as first-class `@` tools.

### 5. Memory you can audit

Long-term **atomic memory** with **Memory Manager** — tier filters, edit/flag/delete, export, negative list so deleted facts stay gone; automatic extraction with reflection flagging and optional promotion. Distinct from chat history and from Library documents ([verified comparison](#verified-long-term-memory)).

Odysseus has persistent **Brain** memory; SillyTavern has World Info / vector lore; LM Studio relies on **plugins** for cross-chat memory. **Qube optimizes for factual preference memory with provenance and editorial control in one native UI.**

### 6. Voice + integrated Desktop Companion

Wake word, push-to-talk, streaming TTS, **barge-in**, and an optional **Desktop Companion** always-on-top orb for quick voice turns while other apps are focused — **built into the same product**, not a third-party overlay ([verified comparison](#verified-desktop-companion-floating-orb)).

### 7. Observability you can audit (local, not cloud)

**Advanced Telemetry**, per-reply **INSPECT RETRIEVAL**, five **opt-in diagnostic logs** with redaction flags, and **Knowledge → Source status** — designed so users can **learn how routing works** and **verify what left the machine** on a given turn. Open source supports independent audit; “Telemetry” here means **on-device diagnostics**, not vendor analytics ([details](#verified-observability-transparency-and-privacy-certainty)).

### 8. Live Sources + privacy-tiered web discovery

**58+ institutional adapters** with toggles and credential health UI; **My knowledge** presets and **Custom sources** (REST, GraphQL, MCP, …) for user-built tools; default **`private`** web tier (**DuckDuckGo** + Wikipedia) with optional **bring-your-own SearXNG** ([details](#verified-live-sources-custom-adapters-and-anonymous-search)).

---

## Honest gaps (Qube is not trying to win these)

| You want… | Better fit today |
|-----------|------------------|
| Maximum prompt / character / lore control | SillyTavern |
| Headless API server or MLX speed on Mac | LM Studio (`llmster`, MLX) |
| Full agent workspace with email, calendar, Docker | Odysseus |
| MCP ecosystem as the primary extension model | LM Studio / Odysseus |
| Bundled SearXNG in one Docker command (no separate config) | Odysseus |
| Live2D / VRM avatar desktop pet (via extensions or third-party tools) | SillyTavern + external pets / AnySoul |
| Image generation pipelines | SillyTavern |
| Multi-user team deployment | Odysseus / Open WebUI |

Qube’s bet: **one native assistant** for people who want voice, grounded answers, curated memory, and institutional research **without assembling a stack**.

---

## Messaging snippets (reuse in README / site)

**Short comparison (README-scale):**

> LM Studio excels at running models; SillyTavern at prompt craft and roleplay; Odysseus at a broad self-hosted workspace. **Qube is something else:** a voice-first desktop assistant that routes each turn to the right knowledge — your Library, live institutional sources, or memory — with `@` composer tools and citations you can inspect, while inference stays on your machine.

**Composer hook:**

> Attach `@library`, `@evidence`, or `@research` to a message when you want that turn grounded — or let the cognitive router infer it from how you ask.

**Library hook:**

> Your files and Qube’s own help docs live in the same Library pipeline — ingest PDFs, then ask in plain language or attach `@file` for surgical grounding.

---

## Maintenance

Re-run this matrix when competitors ship major features (e.g. LM Studio RAG upgrades, Odysseus voice). Update [Readme.md](../../Readme.md) only when positioning or pillar features change — details stay here and in `@help`.

See also: [launch documentation guidelines](../launch_documentation_guidelines.md) · **[Competitive roadmap](../competitive_roadmap.md)** (developer priorities: parity, moats, non-goals).
