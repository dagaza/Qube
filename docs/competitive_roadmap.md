# Competitive roadmap (developer guide)

**Purpose:** Turn [competitive landscape](user/competitive-landscape.md) analysis into **actionable product/engineering priorities** — where to reach **parity**, where to **deepen moats**, and what to treat as **intentional non-goals**.

**Audience:** Maintainers, contributors planning features, launch triage.  
**Last updated:** July 2026  
**Source of truth for positioning:** [docs/user/competitive-landscape.md](user/competitive-landscape.md)

---

## How to use this document

1. **Pick a theme** (voice, memory, MCP, …) from the tables below.
2. Check **Status:** `Moat` · `Parity` · `Deepen` · `Non-goal`.
3. Link PRs/issues to the theme so the matrix stays honest over time.
4. **Re-run the competitive doc** when LM Studio, SillyTavern, or Odysseus ship major features (quarterly or before launch).

**Priority legend**

| Tier | Meaning |
|------|---------|
| **P0** | Closes a credibility gap or protects a core moat before/at launch |
| **P1** | Meaningful parity or moat depth; next 1–2 release cycles |
| **P2** | Nice parity or experimental; only when P0/P1 are stable |

---

## Strategic frame

Qube is **not** trying to become LM Studio (model runner), SillyTavern (roleplay frontend), or Odysseus (homelab workspace). The bet:

> **One native, voice-first desktop assistant** — grounded answers, curated memory, institutional research, and privacy transparency — **without assembling a stack**.

Roadmap work should **strengthen that job**, not chase every competitor checkbox.

---

## Moats — defend and deepen

These are areas where Qube is already ahead. Work here should ** widen the gap** or **reduce friction** so users feel the advantage on day one.

| Moat | Why it matters | Deepen (suggested work) | Priority |
|------|----------------|-------------------------|----------|
| **`@` composer + cognitive router** | Per-turn, citation-oriented routing is rare ([verified](user/competitive-landscape.md#verified-composer-tools)) | Palette discoverability (empty-state hints, recent `@` tokens); router explainability in **INSPECT RETRIEVAL**; golden eval for route regressions | P0 |
| **Voice loop + Desktop Companion** | Integrated orb + wake/PTT/barge-in vs third-party overlays ([verified](user/competitive-landscape.md#verified-desktop-companion-floating-orb)) | Companion screenshot + launch polish; Wayland overlay hardening ([companion_wayland.md](companion_wayland.md)); optional quick actions from orb (open last thread, mute TTS) | P0 |
| **Memory Manager editorial stack** | Atomic facts, negative list, tiers — vs plugins/lore ([verified](user/competitive-landscape.md#verified-long-term-memory)) | **Simplify defaults** (fewer toggles visible until “Advanced”); onboarding tour for Memory; `@help` for “why wasn’t this remembered?” | P0 |
| **In-app `@help` + guided tours** | Help as product data, not external wiki ([verified](user/competitive-landscape.md#verified-in-app-assisted-help)) | `@help` usage analytics → corpus priorities ([in_app_help_knowledge_base.md](in_app_help_knowledge_base.md) §13); tour coverage gaps on new settings | P1 |
| **Live Sources + privacy-tiered web** | 58+ adapters, DDG default, BYO SearXNG ([verified](user/competitive-landscape.md#verified-live-sources-custom-adapters-and-anonymous-search)) | Finish discovery **R5/R10** ([web_discovery_privacy_resilience_plan.md](web_discovery_privacy_resilience_plan.md)); adapter inventory hygiene ([live_knowledge_adapters.md](live_knowledge_adapters.md)); “what left my machine” summary panel | P1 |
| **Local observability** | Telemetry = on-device diagnostics, not cloud ([verified](user/competitive-landscape.md#verified-observability-transparency-and-privacy-certainty)) | Session **network egress summary** (adapters/domains contacted); one-click “privacy report” export; default log redaction presets in Settings | P1 |
| **First-run bootstrap + native shell** | Recommended stack vs assemble-yourself ([verified](user/competitive-landscape.md#verified-onboarding--enthusiast-vs-approachable)) | 8 GB RAM path messaging; bootstrap progress UX; keep PyQt RAM budget documented in [system-requirements.md](user/system-requirements.md) | P0 |
| **Library + help unified pipeline** | Same LanceDB path for user docs and shipped help | EPUB edge cases; “Chat with document” discoverability; Library performance on large corpora | P1 |

---

## Parity — close credible gaps

Work that stops “why doesn’t Qube have X?” objections **without** changing product identity.

### vs LM Studio (model runner + MCP hub)

| Gap | Qube today | Target parity | Suggested work | Priority |
|-----|------------|---------------|----------------|----------|
| **MCP interoperability** | ◐ Custom source MCP connector; not a first-class client UX | Users can attach **standard MCP servers** as grouped, permissioned **capabilities** alongside Live Sources | [MCP & Capability Integrations plan](mcp_capability_integrations_plan.md) — Integrations UI, capability registry, composer `@` search, INSPECT steps | P1 |
| **MLX / Apple Silicon fast path** | Metal GGUF via internal engine | Acknowledge MLX gap; optimize **External Server** story for LM Studio on Mac | Model Manager copy + help: “use LM Studio MLX host”; detect localhost MLX endpoint | P2 |
| **Headless / API-only serving** | Qube is desktop-first | Partner, don’t duplicate **llmster** | Document “Qube UI + LM Studio `llmster` on server” pattern for advanced users | P2 |
| **Plugin/memory ecosystem** | Built-in memory beats plugins for assistants | Optional **import** from LM Studio persistent-memory SQLite/markdown (one-way migration tool) | P2 |

### vs SillyTavern (power-user frontend)

| Gap | Qube today | Target parity | Suggested work | Priority |
|-----|------------|---------------|----------------|----------|
| **Prompt / character depth** | — Not a goal for roleplay | Minimal **system prompt override** + saved profiles (assistant personas, not character cards) | Settings → AI: named profiles; no lorebook complexity | P2 |
| **Extension ecosystem** | Custom sources + presets | **Knowledge pack** sharing (export/import exists — document + community template repo) | P1 |
| **Image / TTS variety** | Voice-first with Kokoro | Optional third-party TTS voice packs | P2 |

### vs Odysseus (self-hosted workspace)

| Gap | Qube today | Target parity | Suggested work | Priority |
|-----|------------|---------------|----------------|----------|
| **Multi-step agents** | ◐ `@research`, skills; not general bash agent | **Scoped agent mode** for research/file tasks with explicit user approval | Agent plan UI; reuse retrieval inspector for tool steps | P1 |
| **Bundled SearXNG** | BYO URL | **One-click local SearXNG** wizard (detect Docker, test URL, set tier) | P1 |
| **Skills that evolve** | `@skill` frameworks static | Skill templates user-editable; optional “refine skill from thread” | P2 |
| **Bi-temporal memory** | Episodes + decay; no knowledge graph | **Valid-at timestamps** on memory rows; “what did I prefer then?” query | P2 |
| **Email / calendar / notes** | — | **Non-goal** (see below) | — | — |
| **Docker / multi-user** | ○ Desktop single-user | Linux **official binary** or documented Flatpak beats Docker for desktop audience | P1 |

### Cross-cutting parity

| Gap | Suggested work | Priority |
|-----|----------------|----------|
| **Linux distribution** | Official install path (Flatpak/AppImage) on [system-requirements.md](user/system-requirements.md) roadmap | P1 |
| **Companion presentation** | Optional **minimal avatar** (keep performance); not Live2D scope | P2 |
| **Memory complexity** | “Simple memory mode” hides promotion/consolidation until user opts in | P0 |

---

## Surpass — go beyond parity in core job

Items that make Qube **clearly best-in-class** for the assistant job, not just “has a checkbox.”

| Theme | Surpass vision | Concrete initiatives | Links |
|-------|----------------|----------------------|-------|
| **Grounded answers** | Best **inspectable** citations in class | INSPECT RETRIEVAL on every grounded path; empty-source downgrade tests; citation integrity telemetry | [cognitive_router.md](cognitive_router.md), `core/citation_integrity_telemetry.py` |
| **Privacy certainty** | User can **audit a session** without reading JSONL | “This session” egress + adapter list in Settings or Telemetry | [web_discovery_privacy_resilience_plan.md](web_discovery_privacy_resilience_plan.md) |
| **Research** | Institutional research in one `@research` flow | Tiered scientific fan-out default; eval harness for adapters | [live_knowledge_adapters.md](live_knowledge_adapters.md), `tools/evaluate_retrieval.py` |
| **Onboarding** | Fastest path to **talking assistant with Library** | Bootstrap + one `@help` guided “first research” workflow | [bootstrap_manifest.py](../core/bootstrap_manifest.py), help workflows |
| **Companion** | Only orb tied to **full routing stack** | Voice turn from orb runs same router/memory/Library as main window (verify + document) | [companion-vs-main-window FAQ](../assets/help/en/faq/companion-vs-main-window.md) |

---

## Intentional non-goals

Do **not** prioritize these for parity unless strategy explicitly changes.

| Area | Better fit | Rationale |
|------|------------|-----------|
| **Roleplay / lorebooks / character cards** | SillyTavern | Different user job; would dilute assistant UX |
| **Primary MCP-as-everything architecture** | LM Studio / Odysseus | Qube leads with **curated `@` tools + Live Sources**; MCP is complement |
| **Email, calendar, homelab workspace** | Odysseus | Scope explosion; desktop assistant focus |
| **Multi-tenant / team deploy** | Odysseus / Open WebUI | Single-user desktop first |
| **Image generation hub** | SillyTavern | Out of core assistant mission |
| **Becoming the model runner** | LM Studio | **External Server** integration is the partnership model |
| **Electron / browser shell** | — | PyQt native shell is a deliberate RAM/UX choice |

When users ask for non-goals, point them to **External Server** (LM Studio/Ollama) or document **intentional boundaries** in `@help`.

---

## Suggested release themes (rolling)

Use as **epic buckets** for issues/PRs. Adjust per maintainer capacity.

### Theme A — Launch credibility (P0)

- [ ] Bootstrap + first-run copy aligned with [bootstrap_manifest.py](../core/bootstrap_manifest.py)
- [ ] Memory “simple mode” UX; Memory Manager tour
- [ ] Companion screenshot + Wayland notes in help
- [ ] Competitive matrix row audit (this doc + [competitive-landscape.md](user/competitive-landscape.md))
- [ ] Social preview + GitHub Pages ([launch_documentation_guidelines.md](launch_documentation_guidelines.md) Phase 4)

### Theme B — Trust & transparency (P1)

- [ ] Session egress / privacy summary UI
- [ ] Web discovery R10 metrics in Telemetry ([web_discovery_privacy_resilience_plan.md](web_discovery_privacy_resilience_plan.md))
- [ ] Default log redaction guidance in Settings → Advanced
- [ ] `@help` articles for privacy tiers and memory delete behavior

### Theme C — Extensibility compatibility (P1)

See [MCP & Capability Integrations plan](mcp_capability_integrations_plan.md).

- [ ] Capability registry + MCP server integrations (settings + permission model)
- [ ] SearXNG setup wizard (detect/test/configure)
- [ ] Knowledge pack templates documented + example pack in repo
- [ ] Scoped research agent with plan + INSPECT steps

### Theme D — Moat depth (P1–P2)

- [ ] Router + retrieval golden eval expansion
- [ ] Adapter eval CI smoke ([live_knowledge_adapters.md](live_knowledge_adapters.md))
- [ ] `@help` query-driven corpus updates
- [ ] Optional memory valid-at / timeline view (Odysseus graph alternative, assistant-scoped)

### Theme E — Platform (P1)

- [ ] Linux official install path
- [ ] 8 GB / low-RAM bootstrap path validation

---

## Tracking competitiveness over time

| When | Action |
|------|--------|
| **Quarterly** | Re-read competitor release notes / READMEs; update [feature matrix](user/competitive-landscape.md#feature-matrix) |
| **Before major release** | Run Theme A checklist; update CHANGELOG positioning bullets |
| **When closing a parity item** | Move row from ○/◐ → ● in matrix; note date in this file’s changelog |
| **When deferring** | Mark **Non-goal** or **P2** here so issues don’t reopen the same debate |

### Roadmap changelog

| Date | Change |
|------|--------|
| 2026-07 | Initial roadmap from competitive landscape deep-dive (composer, help, onboarding, runtime, observability, Live Sources, Companion, memory) |
| 2026-07-20 | MCP row → interoperability framing; link [mcp_capability_integrations_plan.md](mcp_capability_integrations_plan.md) |

---

## Related documents

| Doc | Role |
|-----|------|
| [user/competitive-landscape.md](user/competitive-landscape.md) | Evidence and matrix (user + marketing) |
| [launch_documentation_guidelines.md](launch_documentation_guidelines.md) | Launch pass checklist |
| [architecture/README.md](architecture/README.md) | Technical deep dives |
| [live_knowledge_adapters.md](live_knowledge_adapters.md) | Adapter inventory + expansion |
| [web_discovery_privacy_resilience_plan.md](web_discovery_privacy_resilience_plan.md) | Discovery privacy slices |
| [memory-system.md](architecture/memory-system.md) | Memory architecture |
| [in_app_help_knowledge_base.md](in_app_help_knowledge_base.md) | Help corpus strategy |
| [mcp_capability_integrations_plan.md](mcp_capability_integrations_plan.md) | MCP interoperability — capability registry, permissions, composer, INSPECT |
