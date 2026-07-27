# Web Discovery Privacy & Resilience — Design & Implementation Plan

**Status:** Phase 1 (R1–R4) + Phase 2 (R6–R9) implemented · Phase 3 (R5) planned · R10 partial (Telemetry + manual QA doc; structured `[Discovery]` logs ongoing)  
**Date:** 2026-07-15  
**Parent:** [Web Content Fetch Plan](./web_content_fetch_plan.md)  
**Related:** [Knowledge Adapter HTTP Resilience Plan](./knowledge_adapter_http_resilience_plan.md), [Live Knowledge Adapters](./live_knowledge_adapters.md), [Cognitive Router](./cognitive_router.md)

This document is the **source of truth** for making Qube’s **general-web discovery** (SERP / URL finding) both **privacy-first by default** and **resilient against bot challenges** — without forcing users through commercial search APIs.

---

## 0. Executive summary

Qube’s general-web pipeline discovers URLs via **DuckDuckGo HTML** (`html.duckduckgo.com/html/`), then optionally fetches pages. DDG frequently returns **HTTP 202 + anomaly-modal** (bot challenge) instead of SERP HTML when traffic looks automated. This is a **discovery-layer** problem, not a fetch/relevance regression.

**Product north star:** Qube is a **privacy-first** app. Online search must work **without API keys** as the default path. API-based providers (Brave, Bing, etc.) and self-hosted options (SearXNG) are **opt-in upgrades** for users who choose higher reliability over maximum privacy.

**Core shift:** Move from “retry DDG harder” to **“shape discovery traffic like a careful human, route around blocks gracefully, and let users choose their privacy/reliability trade-off explicitly.”**

### Implementation order (recommended)

| Slice | Focus | Privacy impact |
|-------|--------|----------------|
| **R0** | Reactive resilience (backoff, cache, fallbacks, UI) | Neutral — **shipped** |
| **R1** | Unify all DDG HTTP through discovery registry | **shipped** |
| **R2** | Global discovery pacing (token bucket + jitter) | **shipped** |
| **R3** | Per-session DDG budget (proactive cap) | **shipped** |
| **R3b** | Burst DDG budget (6 / 10 min) + advanced limit UX | **shipped** |
| **R4** | Profile-aware + normalized discovery cache | **shipped** |
| **R5** | Session query dedup / near-duplicate reuse | Planned (Phase 3) |
| **R6** | Privacy-first discovery policy (Settings + tiers) | **shipped** |
| **R7** | Smart opt-in fallback routing | **shipped** |
| **R8** | Bring-your-own SearXNG provider | **shipped** |
| **R9** | Adaptive conservative pacing | **shipped** |
| **R10** | Metrics, Inspector surfacing, manual QA matrix | Partial — Telemetry card + manual QA doc shipped |

**Non-goal:** CAPTCHA solving, proxy rotation farms, or weakening relevance gates to mask empty SERPs.

---

## 1. Product principles

### 1.1 Privacy-first default

1. **No API key required** for core `@internet` / Hybrid web search.
2. **Default discovery path** uses providers that do not require user accounts or paid API registration:
   - DuckDuckGo HTML (primary SERP)
   - Wikipedia API (structured fallback; no key)
3. **API-key providers are optional upgrades**, never the silent default:
   - Brave Search API, Bing Web Search, Google Programmable Search, SerpAPI, etc.
4. **User choice is explicit** — Settings must show *what leaves the machine*, *who sees the query*, and *what improves reliability*.
5. **Self-hosted / BYO endpoints** (SearXNG) are first-class *optional* providers for advanced users who want privacy **and** control.

### 1.2 Reliability without abandoning privacy

Free HTML scraping will always be somewhat fragile. Qube should:

- **Reduce block probability** (pacing, cache, dedup, unified routing)
- **Recover gracefully** when blocked (fallback chain, 30-min DDG pause, user notification, countdown)
- **Offer opt-in reliability** without making it the default

### 1.3 Transparency

Every web turn should be able to answer (Inspector / routing debug):

- Which discovery provider ran (DDG, Brave, Wikipedia, SearXNG, backoff-skipped)
- Whether result came from cache, fallback, or live HTTP
- Whether an API-key provider was used (and which)

---

## 2. Problem statement

### 2.1 Failure mode

| Signal | Meaning | Current behavior |
|--------|---------|------------------|
| HTTP 202 + `anomaly-modal` | DDG bot challenge | Mark 30-min backoff; skip DDG HTTP; fallbacks |
| Empty parse / no SERP rows | Possible soft block or query mismatch | Typed `SearchOutcome`; may fallback |
| Rapid unique queries | Burst automation pattern | **Not yet paced** — high block risk |

### 2.2 What increases DDG call volume (more than retrieval profile)

Retrieval profiles (Fast / Balanced / Thorough / Evidence-first) each make **one** `discover_full` call per general-web turn. Profiles change **fetch depth after SERP**, not DDG call count.

**Real volume drivers:**

| Driver | DDG impact |
|--------|------------|
| Hybrid mode approving many turns | 1 SERP per approved turn |
| Force-web / `@internet` on every message | 1 SERP per message |
| Deep Research (up to 3 sub-queries) | Up to 3 retrievals if routed to web |
| Trusted knowledge path | **Extra** `search_duckduckgo` call; bypasses registry today |
| Distinct queries (paraphrases) | Cache miss → fresh DDG HTTP |
| VPN / datacenter / shared NAT IP | Higher challenge rate (environmental) |

### 2.3 Architectural gap (today)

Two DDG entry paths exist:

| Path | Module | Registry cache | Registry backoff | Fallback chain |
|------|--------|----------------|------------------|----------------|
| **Discovery registry** | `discover_full` → `DuckDuckGoDiscovery` | Yes (5 min) | Yes (30 min) | Brave → Wikipedia |
| **Legacy direct adapter** | `search_duckduckgo` in `pipeline_trusted.py` | No | No | None |

This split can cause **extra DDG traffic during pause windows** and inconsistent user-visible behavior.

---

## 3. Current state (R0 — shipped baseline)

The following is **already implemented** and should be treated as the foundation for subsequent slices.

| Capability | Location | Notes |
|------------|----------|-------|
| Typed `SearchOutcome` + Inspector visibility | `core/knowledge/search_outcome.py`, UI | P3 |
| Wikipedia fallback on `bot_challenge` | `discovery/wikipedia.py`, `registry.py` | M9 |
| Brave Search API optional fallback | `discovery/brave_search.py`, Settings UI | M10; requires key |
| DDG 30-min backoff after challenge | `discovery/backoff.py` | `QUBE_DDG_BACKOFF_SECONDS` |
| 5-min discovery cache (success only) | `discovery/cache.py` | `QUBE_DISCOVERY_CACHE_TTL` |
| Browser-like DDG headers | `mcp/internet_tool.py` | UA, Referer, Accept-Language |
| User notification + top-bar countdown | `notification_types.py`, `main_window.py`, `llm_worker.py` | On new backoff |
| Settings discovery policy summary | `knowledge_web_discovery.py` | Primary / paused / fallbacks |

**Policy today:** DDG is always primary when not in backoff; on `bot_challenge` → Brave (if keyed) → Wikipedia.

---

## 4. Design goals & non-goals

### 4.1 Goals

1. **Maximize free, private discovery success rate** under real desktop usage (Hybrid chat, `@internet`, recipes).
2. **Minimize DDG HTTP calls** without reducing answer quality when users need the web.
3. **Single discovery gateway** for all code paths that need SERP URLs.
4. **Configurable privacy/reliability tiers** — private default, enhanced opt-in.
5. **Observable** — logs, Inspector, Settings status align with actual provider used.

### 4.2 Non-goals

- Defeating CAPTCHAs with Playwright or third-party solving services
- Rotating datacenter proxy pools
- Replacing DDG HTML with undocumented/scraping-only endpoints as default
- Making Brave/Bing the silent default primary
- Weakening SERP relevance gate to hide discovery failures

---

## 5. Privacy & provider model

### 5.1 Provider classes

| Class | Examples | API key | Query seen by | Default role |
|-------|----------|---------|---------------|--------------|
| **Private SERP** | DDG HTML | No | DDG (no Qube account) | **Primary** |
| **Private structured** | Wikipedia API | No | Wikimedia | Fallback |
| **Opt-in API SERP** | Brave, Bing, Google CSE | Yes | API vendor | User-enabled fallback / alternate |
| **BYO meta-search** | SearXNG (self-hosted) | Optional instance auth | User’s instance + its engines | User-enabled optional |
| **Page fetch** | HTTP/Playwright to target URLs | No | Destination site | Post-discovery (separate concern) |

### 5.2 User-facing discovery tiers (Settings)

**Tier A — Private (default)**  
Label: *“Private search (recommended)”*

- Primary: DuckDuckGo HTML
- Fallbacks: Wikipedia only (no API SERP)
- Pacing: on (R2)
- Session budget: conservative (R3)
- Brave/Bing/SearXNG: disabled unless user explicitly enables

**Tier B — Balanced private**  
Label: *“Private search + optional API fallback”*

- Same as Tier A, but if user has configured Brave key → use as fallback on `bot_challenge` (current behavior)
- Clear badge: “API fallback enabled”

**Tier C — Enhanced reliability**  
Label: *“Maximum reliability (uses API providers when configured)”*

- User opts in explicitly
- May enable: API fallback, SearXNG URL, optional alternation / site-bias routing (R7)
- Inspector always shows when an API provider handled the query

**Tier D — Bring your own (advanced)**  
Label: *“Self-hosted SearXNG”*

- User supplies base URL (+ optional API key/header)
- Qube talks only to user’s instance
- Privacy depends on instance config (document clearly)

### 5.3 Copy principles (Settings + notifications)

- Default tier copy emphasizes: *“No account required. Queries go directly to DuckDuckGo and Wikipedia — not through Qube servers.”*
- API tier copy emphasizes: *“Your queries are sent to [Provider] under their terms. Improves reliability when DDG blocks automated access.”*
- Never imply API keys are required for web search to work.

---

## 6. Architecture

### 6.1 Target: single discovery gateway

All SERP acquisition flows through:

```
core/knowledge/discovery/registry.py
  discover_full()           # primary entry (no fallback)
  discover_full_with_fallback()
```

**Consumers to migrate:**

| Consumer | Today | Target |
|----------|-------|--------|
| `pipeline_general_web.py` | `discover_full` | unchanged |
| `pipeline_trusted.py` | `search_duckduckgo` direct | `discover_full_with_fallback` or shared helper |
| Legacy `run_legacy_web_retrieval` | direct adapter | deprecate / delegate to registry |
| Future Deep Research web paths | varies | registry only |

### 6.2 Layered resilience stack

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 0 — Policy (privacy tier, user toggles)               │
├─────────────────────────────────────────────────────────────┤
│ Layer 1 — Proactive shaping (pacing, session budget, dedup) │
├─────────────────────────────────────────────────────────────┤
│ Layer 2 — Cache (normalized query, profile-aware TTL)       │
├─────────────────────────────────────────────────────────────┤
│ Layer 3 — Provider execution (DDG / Brave / Wiki / SearXNG)│
├─────────────────────────────────────────────────────────────┤
│ Layer 4 — Reactive recovery (backoff, fallback chain)       │
├─────────────────────────────────────────────────────────────┤
│ Layer 5 — UX (notification, countdown, Settings, Inspector) │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Fallback chain (privacy-aware)

Default chain (Tier A):

```
DDG (if not in backoff and budget allows)
  → on bot_challenge: Wikipedia (site_bias stripped)
```

Tier B+ (user enabled API fallback):

```
DDG → Brave (if keyed) → Wikipedia
```

Tier D (SearXNG configured):

```
User-selectable:
  - SearXNG only (skip DDG entirely) — advanced
  - DDG → SearXNG → Wikipedia — hybrid
  - SearXNG as fallback only — recommended BYO placement
```

**Invariant:** DDG remains the **default primary** for Tier A/B unless user explicitly changes policy in Settings.

### 6.4 New module sketch (implementation-time)

| Module | Responsibility |
|--------|----------------|
| `discovery/pacing.py` | Global min interval + jitter between outbound discovery HTTP |
| `discovery/session_budget.py` | Per-process DDG call counters + rolling window |
| `discovery/query_normalization.py` | Canonical cache/dedup keys |
| `discovery/policy.py` (extend) | Privacy tier, chain selection, user settings |
| `discovery/searxng.py` | Optional BYO provider |
| `app_settings.py` | `discovery_privacy_tier`, pacing toggles, SearXNG URL |

---

## 7. Implementation slices

### R1 — Unify DDG entry points

**Goal:** Every DDG SERP call respects cache, backoff, pacing (once R2 lands), and typed outcomes.

**Work:**

1. Refactor `pipeline_trusted.py` to use `discover_full_with_fallback` (or `discover_full` + allowlist filter on candidates).
2. Audit repo for `search_duckduckgo(` / `execute_internet_search(` outside `discovery/duckduckgo.py`.
3. Deprecate `run_legacy_web_retrieval` for production paths (keep test parity helper if needed).
4. Ensure trusted path records `discovery_provider` in provenance.

**Exit criteria:**

- With DDG in backoff, trusted pipeline does **not** call DDG HTTP.
- Single cache hit benefits trusted + general-web for identical queries.
- Tests: trusted path respects backoff; no direct `search_duckduckgo` in pipelines.

**Risk:** Low. Behavior change is more consistent pause semantics.

---

### R2 — Global discovery pacing

**Goal:** Space DDG HTTP requests to resemble human pacing; reduce burst-triggered challenges.

**Design:**

- Process-wide scheduler for **discovery HTTP** (separate from scientific `host_scheduler.py`).
- Default: min **3.0s** between DDG requests + **0.5–1.5s jitter** (tunable).
- Applies before `execute_internet_search` (DDG only initially; extensible).
- Blocking wait in worker thread is acceptable; cap max wait (e.g. 30s) then skip DDG → fallback with `SearchOutcome` reason `pacing_timeout`.
- Env: `QUBE_DISCOVERY_PACE_MIN_SEC`, `QUBE_DISCOVERY_PACE_ENABLED`.

**Settings (R6):**

- Toggle: “Slow down web searches slightly to reduce blocking (recommended)” — **default ON** for Tier A/B.

**Exit criteria:**

- Rapid 5-turn Hybrid session spaces DDG calls; logs show `discovery_pace_wait_ms`.
- Inspector shows when pacing delayed discovery.
- Unit tests: scheduler enforces interval; jitter bounded.

**Privacy:** Fully aligned — no third parties; reduces automated fingerprint.

---

### R3 — Per-session DDG budget

**Goal:** Proactive cap on DDG SERP calls per rolling window; prevents hour-long chat sessions from hammering DDG.

**Design:**

- Rolling window: default **30 DDG SERP calls / 60 minutes** (env-tunable).
- Counter increments only on **live DDG HTTP** (not cache hits, not backoff skips).
- When exhausted: skip DDG for remainder of window → fallbacks only; surface status in Settings + optional one-shot notification (deduped).
- Resets independently of 30-min bot backoff (complementary).

**Settings:**

- Show read-only usage: burst (6 / 10 min) + session (30 / 60 min) counters
- Heuristic disclaimer (not official DDG quotas)
- Advanced panel (Prestige unlock): session limit override with confirm-on-raise above default

**Exit criteria:**

- 31st live DDG call in window uses Wikipedia/Brave without DDG HTTP.
- Burst limit blocks DDG before session limit when burst exhausts first.
- Countdown/budget visible in Settings → Web search discovery.
- Tests: counter, window rollover, cache hit does not decrement incorrectly.

---

### R3b — Burst DDG budget + advanced limit UX

**Goal:** Catch rapid Hybrid bursts that trigger bot challenges before the hourly session cap; gate dangerous limit overrides behind advanced UX.

**Design:**

- Rolling burst window: default **6 live DDG SERP calls / 10 minutes** (env-tunable).
- Checked **before** session budget in `_skip_primary_ddg_result()`.
- Distinct synthetic outcomes: `burst_budget_exhausted` vs `session_budget_exhausted`.
- Settings: read-only burst + session counters; session override spinbox only in advanced panel.
- Confirm dialog when raising session limit above default (30); stronger warning above 100.

**Env:**

- `QUBE_DDG_BURST_BUDGET` (default `6`)
- `QUBE_DDG_BURST_WINDOW_SEC` (default `600`)
- `QUBE_DDG_BURST_BUDGET_ENABLED` (default `true`)

**Exit criteria:**

- 7th live DDG call within 10 min skips DDG even if session budget remains.
- Advanced toggle + confirm-on-raise wired in Settings handlers.
- Tests: burst exhaustion, registry skip, block-reason ordering.

---

### R4 — Profile-aware & normalized discovery cache

**Goal:** Fewer repeat DDG calls; Fast/Local-first profiles benefit from longer SERP cache.

**Design:**

1. **Query normalization** before cache key:
   - lowercase, collapse whitespace, strip trailing `?`, optional punctuation normalization
   - site_bias still part of key
2. **Profile-aware TTL:**

| Profile | Suggested SERP cache TTL |
|---------|-------------------------|
| Fast | 10 min |
| Local-first | 10 min |
| Balanced | 5 min (current default) |
| Thorough | 5 min |
| Evidence-first | 5 min |

3. Respect `cache_policy` from `RetrievalProfileSpec` (`aggressive` → longer TTL).

**Exit criteria:**

- Paraphrase-normalized duplicate queries hit cache within TTL.
- Fast profile retains SERP cache longer than Balanced in tests.
- Inspector: `discovery_cache_hit=true`.

---

### R5 — Session query dedup / near-duplicate reuse

**Goal:** Hybrid follow-ups (“tell me more”, “what about X”) avoid redundant SERP when safe.

**Design (conservative v1):**

- Per-session LRU of last **N** discovery results (default N=8, TTL 15 min).
- Reuse prior SERP when:
  - normalized query matches, OR
  - token overlap ≥ **0.85** AND same `site_bias` AND same session
- Never reuse across sessions (privacy: no cross-chat bleed).
- User message explicitly containing “search again” / new `@internet` pin bypasses dedup.

**Future v1.1:** embedding similarity gate (opt-in; local embed only).

**Exit criteria:**

- Follow-up turn with near-identical query does not trigger DDG HTTP.
- Different site_bias forces fresh discovery.
- Tests: reuse hit/miss cases.

---

### R6 — Privacy-first discovery policy (Settings)

**Goal:** Make the privacy/reliability contract visible and configurable.

**UI location:** Settings → Knowledge → Web search discovery (extend existing section).

**Add:**

1. **Privacy tier** dropdown (Tier A/B/C/D) with plain-language descriptions.
2. **Pacing** toggle (linked to R2).
3. **DDG hourly budget** display + advanced override (R3).
4. **Provider rows** with privacy badges:
   - DDG: “Free · No API key · Direct”
   - Wikipedia: “Free · No API key”
   - Brave: “Optional API key · Third-party”
   - SearXNG: “Self-hosted · Advanced”
5. **“What leaves your device”** expandable help panel (short, non-legalistic).

**Persistence:** `app_settings` keys, e.g.:

- `knowledge.discovery_privacy_tier` — default `private`
- `knowledge.discovery_pacing_enabled` — default `true`
- `knowledge.discovery_api_fallback_enabled` — default `false` for Tier A, `true` for Tier B

**Exit criteria:**

- Fresh install = Tier A, pacing on, no API fallback required.
- Changing tier updates `discovery/policy.py` chain at runtime.
- Manual QA: tier copy matches actual provider used in Inspector.

---

### R7 — Smart opt-in fallback routing

**Goal:** Improve reliability for users who **choose** API providers, without affecting private default.

**Behaviors (all opt-in via Tier B+):**

1. **Site-biased queries** (`site:bbcgoodfood.com …`): prefer Brave when keyed (Wikipedia strips site bias).
2. **During DDG backoff:** optionally try SearXNG (if configured) before Wikipedia.
3. **Optional alternation** (Tier C only): e.g. “Use API provider for 1 in every K queries” — **off by default** even in Tier C; research whether alternation helps or hurts fingerprinting.

**Exit criteria:**

- Tier A users never hit Brave without explicit enable.
- `@recipe` site-bias turn uses Brave when keyed + Tier B+.
- Provenance records `fallback_from` and `privacy_tier`.

---

### R8 — Bring-your-own SearXNG provider

**Goal:** Power users run meta-search under their control; Qube integrates as optional discovery provider.

**Design:**

- Settings fields: Base URL (required), optional API key / custom header, timeout.
- `SearXNGDiscovery` implements `DiscoveryProvider` using SearXNG JSON API (`/search?q=…&format=json`).
- Privacy copy: “Queries go to **your** SearXNG instance. Which engines it calls depends on your server config.”
- **Not recommended** for default public instance lists (flaky, often blocked, privacy varies).

**Placement in chain (recommended):**

- Tier D: fallback after DDG challenge, or primary if user explicitly selects “SearXNG first”.

**Exit criteria:**

- Valid local SearXNG returns candidates; typed outcomes on HTTP errors.
- Inspector shows `provider=searxng`.
- Tests with mocked JSON responses.

**Note:** If instance uses DDG engine backend, blocks may persist at instance level — document honestly.

---

### R9 — Adaptive conservative pacing

**Goal:** After repeated challenges, automatically tighten pacing without user intervention.

**Design:**

- Track challenge count in rolling 24h (local only).
- After **2** DDG challenges in 24h: double pacing interval (e.g. 3s → 6s) for 24h.
- After **3+**: suggest Tier B in notification (optional action → Settings); do **not** auto-enable API.
- Clears on successful DDG SERP streak or manual “Reset discovery health” in Settings.

**Exit criteria:**

- Simulated challenges increase pacing; Settings shows “Conservative mode active”.
- Does not change privacy tier automatically.

---

### R10 — Observability, metrics & QA

**Goal:** Measure slices; support manual QA and regression detection.

**Add:**

- `[Discovery]` structured logs: `pace_wait_ms`, `cache_hit`, `budget_remaining`, `tier`, `provider`
- Inspector Summary lines: privacy tier, pacing delayed, budget exhausted
- Routing debug: `discovery_policy` block
- Manual QA doc: `docs/manual_qa_web_discovery_resilience.md` (matrix: tier × profile × Hybrid × backoff)

**Exit criteria:**

- Manual QA matrix passes for Tier A default install.
- No silent API provider use in Tier A.

---

## 8. Retrieval profiles — policy interaction

Retrieval profiles control **fetch depth and adapter orchestration**, not DDG call multiplicity.

| Profile | Discovery impact (this plan) |
|---------|------------------------------|
| Fast | Longer SERP cache TTL (R4) |
| Local-first | Longer cache + local-before-remote may skip web entirely |
| Balanced | Baseline |
| Thorough | Same SERP count; more page fetches (destination sites, not DDG) |
| Evidence-first | Ranking hint only; default SERP-only |

**Do not** add profile-specific DDG pacing multipliers in v1 — keep pacing global to avoid confusing UX. Revisit only if metrics show Thorough sessions correlate with blocks (unlikely).

---

## 9. Environment variables (summary)

| Variable | Default | Slice | Purpose |
|----------|---------|-------|---------|
| `QUBE_DISCOVERY_BACKOFF` | `1` | R0 | Enable DDG backoff |
| `QUBE_DDG_BACKOFF_SECONDS` | `1800` | R0 | Pause duration |
| `QUBE_DISCOVERY_CACHE` | `1` | R0/R4 | SERP cache on |
| `QUBE_DISCOVERY_CACHE_TTL` | `300` | R0/R4 | Base TTL seconds |
| `QUBE_DISCOVERY_PACE_ENABLED` | `1` | R2 | Pacing on |
| `QUBE_DISCOVERY_PACE_MIN_SEC` | `3.0` | R2 | Min DDG interval |
| `QUBE_DDG_BURST_BUDGET` | `6` | R3b | Max live DDG calls / burst window |
| `QUBE_DDG_BURST_WINDOW_SEC` | `600` | R3b | Burst budget window |
| `QUBE_DDG_BURST_BUDGET_ENABLED` | `true` | R3b | Enable burst budget |
| `QUBE_DDG_SESSION_BUDGET` | `30` | R3 | Max live DDG calls / window |
| `QUBE_DDG_SESSION_BUDGET_WINDOW_SEC` | `3600` | R3 | Budget window |
| `QUBE_SEARXNG_BASE_URL` | — | R8 | BYO instance (optional env override) |

User Settings should mirror key toggles; env vars remain override for power users and CI.

---

## 10. Testing strategy

| Area | Tests |
|------|-------|
| Registry unification | Trusted path backoff respect (R1) |
| Pacing | Scheduler interval, jitter bounds (R2) |
| Session budget | Exhaustion, cache exclusion (R3) |
| Cache normalization | Key collision / hit (R4) |
| Session dedup | Reuse rules, site_bias isolation (R5) |
| Policy tiers | Chain selection per tier (R6) |
| SearXNG | Mock JSON provider (R8) |
| Notifications | Budget exhausted, conservative mode (R9) |

Keep tests **offline** — mock `execute_internet_search` and provider classes; no live DDG in CI.

---

## 11. Manual QA checklist (abbreviated)

1. **Tier A fresh install:** `@internet` query → DDG primary; no API calls; pacing felt ≤ few seconds on rapid sends.
2. **Bot challenge simulation** (fixture/mock): notification + countdown; Wikipedia fallback; DDG paused.
3. **Trusted knowledge** during backoff: no DDG HTTP (post-R1).
4. **Tier B + Brave key:** challenge → Brave used; Inspector shows provider.
5. **SearXNG local:** configured URL returns results; tier D copy accurate.
6. **Session budget:** after threshold, fallbacks only + Settings counter accurate.
7. **App restart during backoff:** countdown resumes; no duplicate notification.

Full matrix → [manual_qa_web_discovery_resilience.md](manual_qa_web_discovery_resilience.md).

---

## 12. Research backlog (no implementation commitment)

Items to evaluate before building:

| Option | Privacy | Notes |
|--------|---------|-------|
| Additional HTML SERP sources (Startpage, Mojeek, etc.) | High if no key | Legal/ToS fragility similar to DDG |
| Tor / proxy support | High anonymity | UX complexity; latency; out of scope for v1 |
| Client-side query embedding dedup | High (local) | R5 v1.1; requires embed model |
| Shared discovery cache across app restarts | Neutral | Persisted cache file — privacy vs convenience trade-off |
| Official DDG API / partnership | N/A today | No public anonymous API equivalent to HTML SERP |

---

## 13. Relationship to other plans

| Plan | Relationship |
|------|--------------|
| [Web Content Fetch Plan](./web_content_fetch_plan.md) | Fetch/extract after discovery; profiles gate `fetch_url_count` |
| [Knowledge Adapter HTTP Resilience](./knowledge_adapter_http_resilience_plan.md) | Scientific **API** adapters; separate scheduler — reuse patterns, not shared bucket with DDG HTML |
| [Cognitive Router](./cognitive_router.md) | Hybrid frequency drives discovery volume — future: optional “pace Hybrid web approvals” (out of scope here) |

---

## 14. Suggested delivery phases

### Phase 1 — Consistency & proactive shaping (privacy-neutral, high ROI)

- R1 Unify entry points
- R2 Pacing
- R3 Session budget
- R4 Cache improvements

**Outcome:** Default private path is harder to block; behavior is consistent across pipelines.

### Phase 2 — User contract & opt-in reliability

- R6 Settings privacy tiers
- R7 Smart API fallback routing
- R9 Adaptive conservative pacing

**Outcome:** Users understand and control privacy vs reliability.

### Phase 3 — Advanced & observability

- R8 SearXNG BYO
- R5 Session dedup
- R10 Metrics + manual QA doc

**Outcome:** Power-user path; measurable regression safety net.

---

## 15. Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-15 | DDG HTML remains default primary | Free, no API key, aligns with privacy-first |
| 2026-07-15 | API providers are opt-in fallbacks only | Avoid silent third-party query logging |
| 2026-07-15 | Pacing + budget before new providers | Cheapest anti-block lever; no privacy cost |
| 2026-07-15 | SearXNG as BYO optional, not hosted by Qube | Ops burden + privacy varies per instance |
| 2026-07-15 | Retrieval profiles do not multiply DDG calls | Evidence from pipeline code; fetch ≠ SERP |

---

## 16. Open questions (resolve during implementation)

1. **Default pacing interval** — 3s vs 5s? Validate with real Hybrid sessions.
2. **Tier A fallback** — Wikipedia-only vs limited DDG retries after soft empty parse?
3. **Session budget defaults** — 30/hour too low for power users?
4. **SearXNG** — fallback-only vs allow “SearXNG primary” in Tier D?
5. **Persist discovery cache to disk** — convenience vs forensic footprint on shared machines?

---

*When implementation starts, update slice statuses in this document and cross-link shipped modules in [Live Knowledge Adapters](./live_knowledge_adapters.md) if discovery providers become part of that inventory.*
