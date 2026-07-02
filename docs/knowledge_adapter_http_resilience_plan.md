# Knowledge Adapter HTTP Resilience — Design & Implementation Plan

**Status:** Slices 1–6 implemented  
**Date:** 2026-06-25  
**Parent:** [External Knowledge Platform Plan](./external_knowledge_platform_plan.md)  
**Related:** [Phase 6c — Scientific Discipline Packs](./phase6c_scientific_discipline_packs.md), [Logging & diagnostics](./logging_and_diagnostics.md), [Eval retrieval corpus](../eval/retrieval_corpus/README.md)

This document is the **source of truth** for shaping outbound HTTP traffic to external knowledge APIs: observability, credentials, per-host rate limiting, header-aware retries, caching, and circuit breaking. It consolidates internal analysis and external architecture review feedback into an implementation roadmap tailored to Qube’s current codebase.

---

## 0. Executive summary

Live scientific retrieval (OpenAlex, PubMed, arXiv, PubChem, INSPIRE-HEP, etc.) occasionally returns **429** (rate/quota exceeded) or **503** (server overload). Today, adapters call `requests.get` directly with minimal coordination. Retries are ad hoc (OpenAlex only). Eval harness flakes and repeated dev runs can amplify the problem.

**Core shift:** Move from “retry harder” to **“shape traffic before it leaves Qube.”**

**Implementation order (merged priority):**

1. HTTP observability (measure before optimizing)
2. API keys where providers offer them (OpenAlex, NCBI) — env + backend resolution first
3. Shared per-host scheduler / rate limiter
4. Header-aware retry policy (`Retry-After`, OpenAlex budget headers)
5. **User-managed credentials & source status UI** (Settings → Knowledge; see §8)
6. Strengthen caching (TTL, optional DOI/metadata layers)
7. Circuit breaker per host
8. In-app quota/limit notifications & upgrade path (hybrid model; see §8.2)
9. Eval pacing improvements (if live eval still flakes)
10. Tiered adapter fan-out (behavior change; defer until metrics justify)

**Product principle (credentials):** Everything works **without keys** where providers allow anonymous access. Keys are an optional upgrade for higher quotas and reliability — never a paywall for core `@evidence` usage. See **§8 — User-managed credentials & source management**.

**Non-goal:** Introduce a parallel `RetrievalPlanner` abstraction stack. Extend the existing pipeline (`scientific_query_planner`, `pipeline_scientific`, adapters) with a thin shared HTTP layer underneath.

---

## 1. Problem statement

### 1.1 Failure modes are different

| HTTP code | Meaning | Correct client behavior |
|-----------|---------|-------------------------|
| **429** | Quota or rate ceiling hit | Slow down; honor `Retry-After`; if daily budget exhausted, **stop retrying** until reset |
| **503** | Temporary overload / unavailability | Exponential backoff + limited retries |

Lumping 429 and 503 into one retry loop wastes time (credit exhaustion) or hammers sick endpoints (503 storms).

### 1.2 Observed symptoms in Qube

- Live eval flakes on OpenAlex-primary queries (`cross_001`, `soc_001`, `polisci_001`) with 429/503.
- Scientific pipeline fires up to **3 adapters in parallel** per cache miss.
- Only `core/knowledge/adapters/openalex.py` has retry logic (one 3s sleep, max 2 attempts).
- No OpenAlex or NCBI API keys wired in adapters (RePEc key pattern exists via `QUBE_REPEC_API_KEY`).
- Query-level evidence cache exists (`~/.qube/evidence_cache/`, 1h TTL) but no HTTP-level metrics.

### 1.3 What is *not* the main problem

**Cross-host parallelism is fine.** PubMed + OpenAlex + INSPIRE in parallel hit different hosts and do not share rate-limit buckets.

**Same-host uncoordinated traffic is the real risk.** Example: chemistry discipline pack runs `pubchem` + `pubmed` + `openalex` together — **PubChem and PubMed both use `*.ncbi.nlm.nih.gov`**, potentially 6+ NCBI calls in a short window (PubMed esearch + efetch; PubChem CID + properties + description per name candidate).

---

## 2. Current state (codebase baseline)

### 2.1 Orchestration

| Component | Role |
|-----------|------|
| `core/knowledge/pipeline_scientific.py` | Parallel adapter fan-out (`ThreadPoolExecutor`, max 3 workers); evidence cache read/write |
| `core/knowledge/scientific_adapters.py` | Discipline-based adapter ordering |
| `core/knowledge/scientific_query_planner.py` | Per-adapter query shaping |
| `core/knowledge/evidence_cache.py` | File-backed query cache (SHA256 key, 3600s default TTL) |
| `core/knowledge/observability.py` | Turn-level `retrieval_trace` (adapter names, latency — **not** HTTP counts) |

### 2.2 Adapter HTTP patterns (approximate calls per invocation)

| Adapter | Host | HTTP calls (typical) | Retry today |
|---------|------|----------------------|-------------|
| OpenAlex | `api.openalex.org` | 1 (+ optional retry) | 429/503, 3s sleep, 2 attempts |
| PubMed | `eutils.ncbi.nlm.nih.gov` | 2 (esearch + efetch) | None |
| PubChem | `pubchem.ncbi.nlm.nih.gov` | 3+ per name candidate | None |
| arXiv | `export.arxiv.org` | 1 | None |
| INSPIRE-HEP | `inspirehep.net` | 1 | None |
| Wikipedia | `en.wikipedia.org` | 2 (search + extract) | None |
| CourtListener | `courtlistener.com` | 1+ | None |
| RePEc (EconBiz) | varies | 1 | None |
| DBLP | `dblp.org` | 1 | None |
| bioRxiv | `api.biorxiv.org` | 1 | None |

**Cold-cache eval estimate:** 12 scientific queries → roughly **40–80+ HTTP requests** (not 12), higher if PubChem name probing or OpenAlex retries fire. **Measure empirically in Slice 1.**

### 2.3 Existing credentials pattern

- `QUBE_REPEC_API_KEY` — RePEc/EconBiz adapter (env only today)
- `QUBE_KNOWLEDGE_FIXTURES=1` — fixture mode for tests/eval offline
- `QUBE_EVIDENCE_CACHE` — toggle evidence cache (default on)

### 2.4 Existing Settings UI (baseline for §8)

**Settings → Knowledge → Preferred sources** (`ui/views/settings/sections/knowledge_sources.py`) already exposes:

- Per-service sections: Scientific literature, Finance, Legal
- Per-discipline UI groups (Science, Biology, Chemistry, …) with **enable/disable checkboxes**
- Catalog metadata via `AdapterCatalogEntry` (`requires_api_key` flag exists but is coarse — treated as “API key required to use at all” for stubs like Semantic Scholar)

**Gap:** No per-provider credential fields, no connection status, no quota display, no “Test connection” flow. Checkboxes duplicate the same adapter id across UI groups (e.g. OpenAlex appears under Science, Biology, Chemistry…) — **credentials must be keyed by provider id once**, not per UI group.

---

## 3. Provider rate-limit reference

Conservative operating targets for v1 scheduler policies (verify against current provider docs before shipping):

| Host / provider | Documented limit | Qube v1 target | Notes |
|-----------------|------------------|----------------|-------|
| **OpenAlex** | 100 req/sec burst; ~$1/day free with API key; search ~$0.001/call | 5–10 req/sec; track `X-RateLimit-*` | Without key: ~$0.10/day. [Auth docs](https://developers.openalex.org/api-reference/authentication) |
| **NCBI E-utilities** (PubMed) | 3 req/sec without key; 10 with key | 2.5 req/sec (3) or 8/sec (10 with key) | 2 calls per search. [E-utilities guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/) |
| **PubChem PUG REST** | 5 req/sec, 400/min, 300s compute/min | 4 req/sec; read `X-Throttling-Control` | Same NCBI IP budget as E-utilities in practice |
| **arXiv** | 1 req / 3 sec, single connection | Strict serialize: 1 req / 3.5 sec | Strictest limit. [arXiv API ToU](https://info.arxiv.org/help/api/tou.html) |
| **INSPIRE-HEP** | 15 req / 5 sec per IP | 2.5 req/sec | Wait ≥5s after 429. [REST API doc](https://github.com/inspirehep/rest-api-doc) |
| **Wikimedia** | 10 req/min (unidentified); 200/min with User-Agent | 3 req/sec with compliant UA | Already sends User-Agent |
| **CourtListener** | 5/min, 50/hr, 125/day (new accounts) | 4/min unless membership tier | Very tight; legal eval only |

Scheduler policies should be **adaptive in design** (reduce rate after 429) but **fixed conservative in v1** until metrics justify tuning.

---

## 4. Design principles

1. **Measure before optimizing.** No sophisticated limiter without HTTP-level counters.
2. **Same-host coordination, not less parallelism.** Keep cross-host fan-out; queue same-host requests.
3. **Centralize HTTP concerns.** Adapters stay thin; one scheduler + one instrumented client.
4. **Distinguish 429 flavors.** Rate-limited (retry after delay) vs budget exhausted (fail fast, use cache).
5. **Every request not made is free.** Extend caching deliberately after measuring duplication.
6. **Desktop-app scope.** Circuit breakers protect user experience and external providers; not multi-tenant crawl scale.
7. **Extend, don’t rewrite.** Bolt resilience under existing pipelines and adapters.

---

## 5. Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Knowledge pipelines (scientific, finance, legal, trusted)  │
│  • existing planners + discipline policy + evidence cache   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Adapters (thin)                                            │
│  search_openalex(), search_pubmed(), …                      │
│  → knowledge_http.get(url, host_policy=...)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  knowledge_http (new module)                                │
│  • instrumented GET/POST                                      │
│  • header-aware retry                                         │
│  • records metrics + emits audit fields                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  HostScheduler (process-wide, per hostname)                 │
│  • acquire(host) → token bucket / mutex (arXiv)               │
│  • release + record outcome                                   │
│  • optional circuit breaker state per host                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 `knowledge_http` module (proposed: `core/knowledge/http_client.py`)

Responsibilities:

- Wrap `requests` with timeouts (default 10–30s).
- Call `HostScheduler.acquire(hostname)` before each request.
- On response: record status, latency, selected headers, retry count.
- Retry policy:
  - **429 + `Retry-After`** → sleep and retry (cap attempts).
  - **429 + OpenAlex `X-RateLimit-Remaining: 0`** → raise `BudgetExhausted` (no retry).
  - **503, 502, 504** → exponential backoff with jitter (max 3–4 attempts).
  - **4xx other than 429** → no retry.
- Inject API keys from settings/env where configured.
- Return a small `HttpResult` dataclass (response, metrics dict).

### 5.2 `HostScheduler` (proposed: `core/knowledge/host_scheduler.py`)

Process-wide singleton (thread-safe):

```python
# Conceptual API — not final code
scheduler.acquire("api.openalex.org")   # blocks until slot available
# ... perform request ...
scheduler.record("api.openalex.org", status=200, headers=...)
scheduler.release("api.openalex.org")
```

Policy table (configurable, defaults from §3):

```yaml
# Conceptual — implement as Python dict / dataclass
api.openalex.org:
  kind: token_bucket
  rate_per_sec: 8
  burst: 16
  backoff_on_429: true
  obey_retry_after: true

export.arxiv.org:
  kind: serialized_interval
  min_interval_sec: 3.5

eutils.ncbi.nlm.nih.gov:
  kind: token_bucket
  rate_per_sec: 2.5   # bump when NCBI API key present

pubchem.ncbi.nlm.nih.gov:
  kind: shared_bucket   # share NCBI budget with eutils
  parent: ncbi
```

**NCBI shared bucket:** PubMed and PubChem share one logical limiter keyed `ncbi` (both subdomains map to the same pool).

### 5.3 Observability schema

Extend retrieval diagnostics (and optional dedicated JSONL logger) with per-turn HTTP summary:

```json
{
  "http_summary": {
    "requests_total": 7,
    "cache_hits_evidence": 0,
    "by_host": {
      "api.openalex.org": {
        "requests": 2,
        "429": 1,
        "503": 0,
        "retries": 1,
        "latency_ms_p95": 840,
        "rate_limit_remaining": 0.94
      },
      "ncbi": {
        "requests": 5,
        "429": 0,
        "503": 0,
        "retries": 0
      }
    }
  }
}
```

Surface in:

- `relevance_diag` on scientific turns (when audit enabled).
- Eval harness stdout / JSON report for `--live` runs.
- Optional: `QUBE_HTTP_METRICS=1` → append to audit log.

### 5.4 Caching layers (phased)

| Layer | Exists today | Proposed |
|-------|--------------|----------|
| **Query → ranked rows** | Yes (`evidence_cache.py`, 1h TTL) | Configurable TTL; eval mode longer TTL |
| **Negative cache** | No | Remember host throttle / empty 503 for N minutes |
| **DOI / work ID → metadata** | No | Slice 5 if metrics show cross-query duplication |
| **Adapter raw response** | No | Optional; lower priority |

Cache keys remain separate from HTTP layer; HTTP client does not cache unless explicitly wired.

### 5.5 Circuit breaker (per host)

States: `closed` → `open` (after N consecutive 503/429 failures) → `half_open` (probe after cooldown).

When open:

- Adapter calls fail fast with warning in trace: `"openalex_circuit_open"`.
- Pipeline may still serve cached evidence or proceed with other adapters.
- Default cooldown: 5 minutes (configurable).

Integrated into `HostScheduler`, not separate service.

---

## 6. Configuration & credentials

### 6.1 Resolution order (backend)

Credentials resolve in priority order:

1. **User settings** (Settings → Knowledge → provider credential store; §8.7)
2. **Environment variables** (dev/CI override; existing pattern)
3. **Anonymous** (no key — scheduler uses anonymous rate policies from §3)

| Credential id | Adapters served | Env override | User settings field |
|---------------|-----------------|----------------|---------------------|
| `openalex` | `openalex` | `QUBE_OPENALEX_API_KEY` | OpenAlex API key |
| `ncbi` | `pubmed`, `pubchem` | `QUBE_NCBI_API_KEY` | NCBI API key (shared) |
| `repec` | `repec` | `QUBE_REPEC_API_KEY` | RePEc / EconBiz key |
| `courtlistener` | `courtlistener` | `QUBE_COURTLISTENER_TOKEN` | CourtListener token |
| `semantic_scholar` | `semantic_scholar` (future) | `QUBE_SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar key |

Adapters never read settings directly — `knowledge_http` / `credential_resolver` injects keys per request. Never log key values; audit logs record `credential_mode: anonymous | user_key | env_key` only.

### 6.2 Env-only phase (Slice 2)

Initial backend wiring may ship with env vars only; user UI follows in Slice 9 (§7).

---

## 7. Implementation slices

### Slice 1 — HTTP observability (Priority 1)

**Goal:** Count every outbound knowledge HTTP request without changing behavior.

**Work:**

- Add `core/knowledge/http_metrics.py` — thread-safe counters by host; rolling latency samples.
- Add thin wrapper or monkey-patch path used by one adapter (OpenAlex) as proof of concept, then migrate others.
- Attach `http_summary` to scientific `relevance_diag` when web audit logging enabled.
- Add eval harness flag `--http-report` (or always print summary in `--live` mode).

**Files (expected):**

- `core/knowledge/http_metrics.py` (new)
- `core/knowledge/pipeline_scientific.py` (diag attachment)
- `tools/evaluate_retrieval.py` (report)
- `tests/test_http_metrics.py` (new)

**Acceptance criteria:**

- Single `@evidence` turn logs request count per host.
- Live 12-query eval prints total HTTP requests and 429/503 breakdown.
- No change to retrieval results in fixture mode.

---

### Slice 2 — API keys (Priority 2)

**Goal:** Wire optional keys for OpenAlex and NCBI; document setup.

**Work:**

- OpenAlex: append `api_key` param in `openalex.py` when `QUBE_OPENALEX_API_KEY` set.
- PubMed: append `api_key` to E-utilities params when `QUBE_NCBI_API_KEY` set.
- Bump scheduler NCBI rate when key detected.
- Update `core/knowledge/adapters/catalog.py` flags (`requires_api_key` for OpenAlex optional note).
- Docs: env vars in eval README + manual QA snippet.

**Acceptance criteria:**

- With keys set, OpenAlex `/rate-limit` shows remaining budget > default no-key tier.
- PubMed accepts burst without 429 at modest eval pace.
- Keys absent → current behavior (no regression).

---

### Slice 3 — Shared per-host scheduler (Priority 3)

**Goal:** Process-wide rate limiting; eliminate ad hoc sleeps in adapters.

**Work:**

- Implement `HostScheduler` with token bucket + serialized interval modes.
- Map hostnames to policies (§3); **NCBI shared bucket** for PubMed + PubChem.
- Route all adapter `requests.get` through `knowledge_http.get`.
- Remove OpenAlex inline `time.sleep(3)` retry (defer to Slice 4 policy).
- arXiv: strict global mutex + 3.5s minimum gap.

**Files (expected):**

- `core/knowledge/host_scheduler.py` (new)
- `core/knowledge/http_client.py` (new)
- All files under `core/knowledge/adapters/*.py` that call `requests` directly

**Acceptance criteria:**

- Unit tests: token bucket enforces rate; arXiv serializes.
- Integration test: chemistry query does not exceed NCBI configured rate in metrics.
- Parallel physics query (arxiv + inspire + openalex) still completes in reasonable time.

---

### Slice 4 — Header-aware retry (Priority 4)

**Goal:** Smart retries; stop burning quota on budget exhaustion.

**Work:**

- Implement retry in `http_client.py` only (not per adapter).
- Honor `Retry-After` (seconds or HTTP-date).
- OpenAlex: parse `X-RateLimit-Remaining`, `X-RateLimit-Reset`; distinguish budget exhausted.
- Exponential backoff + jitter for 503 (base 1s, max ~16s, max 3 retries).
- Log retry reasons in `http_summary`.

**Acceptance criteria:**

- Simulated 429 with `Retry-After: 1` waits ~1s (mocked clock in tests).
- Budget exhausted → no retry loop; trace warning emitted.
- 503 → at most 3 retries then graceful empty adapter result.

---

### Slice 5 — Strengthen caching (Priority 5)

**Goal:** Reduce duplicate HTTP without over-engineering.

**Work:**

- Evidence cache: env `QUBE_EVIDENCE_CACHE_TTL` (default 3600; eval suggestion 86400).
- Optional negative cache entry when host circuit open or budget exhausted (short TTL, e.g. 300s).
- **If Slice 1 metrics show duplication:** add DOI-level cache (`core/knowledge/metadata_cache.py`) keyed by normalized DOI / OpenAlex ID, TTL 7d.

**Acceptance criteria:**

- Repeated identical `@evidence` query within TTL → zero HTTP calls (metrics confirm).
- Cache miss behavior unchanged for novel queries.

---

### Slice 6 — Circuit breaker (Priority 6)

**Goal:** Stop hammering unhealthy hosts.

**Work:**

- Add breaker state to `HostScheduler`.
- Threshold: e.g. 3 consecutive 503 or rate-limit failures within 60s → open 5 min.
- Half-open: allow 1 probe request.
- Expose `host_health` in `http_summary` / trace warnings.

**Acceptance criteria:**

- After forced failures in test, host short-circuits; other adapters still run.
- After cooldown, probe succeeds → closed.

---

### Slice 7 — Eval pacing (Priority 7)

**Goal:** Live eval reliability if Slices 1–4 insufficient.

**Status:** Pacing changes **deferred** pending live eval evidence. Throttle reporting in eval JSON **implemented** (partial).

**Work:**

- Increase `_inter_query_delay_s` for scientific live eval (configurable, default 2 → 5s). **Deferred**
- Optional `--serial-adapters` eval flag: run adapters sequentially (slower, gentler). **Deferred**
- Report throttle events in eval JSON output distinctly from retrieval failure. **Done** (`http_throttle_report.py`, per-query `throttle_report` + `failure_class`)

**Acceptance criteria:**

- 12-query live eval passes discipline gates ≥70% on 3 consecutive runs (same day, with keys). **Not validated in CI** — run manually after Slices 1–6.

---

### Slice 8 — Tiered adapter fan-out (Priority 8, optional)

**Goal:** Call fewer adapters per turn when primary source is sufficient.

**Work:**

- Extend `pipeline_scientific.py`: phase 1 = primary adapter(s) only; phase 2 = fallbacks if `len(candidates) < threshold`.
- Respect existing slot reservations (PubChem, RePEc, arXiv).
- Feature flag: `QUBE_TIERED_SCIENTIFIC_RETRIEVAL=1`.

**Acceptance criteria:**

- OpenAlex-only sociology query makes 1 HTTP call on success path (metrics).
- Medical query still hits PubMed first; quality unchanged on eval corpus.

**Defer until:** Slice 1 metrics show excess fan-out as dominant cost.

---

### Slice 9 — Credential store & resolver (Priority 5, backend)

**Goal:** Single backend module for user/env/anonymous credential resolution; foundation for Settings UI.

**Work:**

- Add `core/knowledge/credentials.py`:
  - `KnowledgeCredentialProfile` dataclass per provider (see §8.6 catalog extension)
  - `resolve_credential(provider_id) -> CredentialBundle(mode, secret?, metadata)`
  - Persist user keys in settings store (`KEY_KNOWLEDGE_PROVIDER_CREDENTIALS`) — see §8.7 security
- Wire `http_client.py` to resolver (Slice 2/3 paths)
- NCBI key shared across PubMed + PubChem adapters
- Unit tests: env overrides user; empty → anonymous; never expose secret in repr/logs

**Acceptance criteria:**

- OpenAlex adapter uses user key from settings when env unset
- PubMed + PubChem both receive NCBI key from one stored value
- Fixture mode ignores live credentials

---

### Slice 10 — Settings: provider credentials UI (Priority 5, UX)

**Goal:** Extend Settings → Knowledge with per-provider credential rows (checkbox + optional key + benefits copy).

**Work:**

- Extend `AdapterCatalogEntry` → add `ProviderCredentialSpec` (or parallel registry keyed by **provider id**, not duplicate catalog rows):
  - `supports_anonymous: bool`
  - `supports_free_api_key: bool`
  - `paid_tier_available: bool`
  - `signup_url: str`
  - `anonymous_benefits_line: str`
  - `key_benefits_line: str`
- New UI subsection **“Provider credentials”** above or within **Preferred sources** (`knowledge_sources.py` or sibling `knowledge_provider_credentials.py`):
  - One row per **unique provider** (dedupe OpenAlex across UI groups)
  - Masked `QLineEdit` (password echo), **Test connection**, **Open provider page**
  - Status line: “Current mode: Anonymous access” / “Connected with your API key”
  - Static benefit text per provider (see §8.3)
- Handlers in `KnowledgeHandlersMixin`: save on blur/Test; emit settings reload signal
- Do **not** require keys for implemented adapters that work anonymously

**Example row (Scientific):**

```
✓ OpenAlex                                    [enabled checkbox — existing prefs]
  API key: [________________]  [Test]  [Get free key ↗]
  Current mode: Anonymous access
  Adding a free API key increases your daily quota and may improve reliability.
```

**Acceptance criteria:**

- App fully functional with all key fields empty
- Saving OpenAlex key persists across restart; Test connection hits `/rate-limit` or lightweight search
- Semantic Scholar row shows “API key required” and disabled retrieval until key + implemented adapter

---

### Slice 11 — Source status panel & limit notifications (Priority 6, UX)

**Goal:** Treat credentials as **source management** — visibility into connection mode, quota, and health; gentle upgrade path when limits hit.

**Work:**

- Add `core/knowledge/provider_status.py`:
  - Aggregate from `http_metrics`, `HostScheduler` breaker state, last credential test, OpenAlex rate-limit headers
  - `ProviderStatus`: `connection_mode`, `quota_label`, `health`, `last_error`, `resets_at`
- **Settings panel — “Source status”** (compact table or cards):

  | Provider | Status | Quota | Health |
  |----------|--------|-------|--------|
  | OpenAlex | Anonymous | ~$0.10/day | Good |
  | PubMed (NCBI) | Anonymous | 3 req/sec | Good |
  | CourtListener | Not configured | — | — |

- **In-app notification** (non-blocking banner or assistant notice) when `BudgetExhausted` / daily quota hit:
  - Copy pattern from §8.4 (no pressure; explain free key; actions: Open provider, Paste key in Settings, Dismiss, Try again tomorrow)
  - Trigger at most once per provider per day (debounce)
- Optional: link from notification directly to Settings → Knowledge credentials anchor

**Acceptance criteria:**

- After simulated OpenAlex budget exhaustion, user sees upgrade message with working Settings deep link
- Status panel refreshes on Test connection and after retrieval turns (when audit enabled)
- Health shows `Degraded` when circuit breaker open

---

## 8. User-managed credentials & source management

This section specifies the **product and UX layer** for API keys — complementary to HTTP resilience (§5–§7). It turns credentials from hidden env vars into transparent **source management**, which can differentiate Qube from tools that silently fail or require manual config files.

### 8.1 Design goals

1. **Works without keys** — Anonymous access everywhere providers allow it; discipline routing and `@evidence` remain usable out of the box.
2. **Transparent upgrade path** — When limits bite, explain why and offer a **free** key where available — not “you must pay.”
3. **One row per provider** — Credentials keyed by provider id (`openalex`, `ncbi`, …), not by catalog UI group duplicates.
4. **Honest labeling** — Distinguish anonymous vs free API key vs optional paid tier in UI copy and docs.
5. **No secret leakage** — Keys stored locally; never in logs, traces, or retrieval audit JSONL.

### 8.2 Hybrid access model

```
┌─────────────────────────────────────────────────────────────┐
│  Default: Anonymous mode                                     │
│  • Scheduler uses anonymous rate policies (§3)               │
│  • Evidence cache minimizes repeat calls                     │
│  • Full @evidence / discipline routing works                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ user hits daily quota OR repeated 429
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Soft notification (Slice 11)                                │
│  “You've reached the anonymous limit for OpenAlex today…”    │
│  [Get free API key] [Open Settings] [Continue without]       │
└───────────────────────────┬─────────────────────────────────┘
                            │ user adds free key in Settings
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Authenticated mode                                            │
│  • Higher quota / rate (provider-specific)                     │
│  • Status panel shows Connected + remaining budget           │
│  • Scheduler switches to keyed policy tier                   │
└─────────────────────────────────────────────────────────────┘
```

**Behavior rules:**

- Limit notifications are **informational**, not modal blockers — retrieval may still return cached results or other adapters.
- Never imply payment is required when a **free** key exists — use explicit “free API key” language.
- “Continue tomorrow” is valid — respect provider reset times (OpenAlex: midnight UTC).

### 8.3 Provider access matrix (user-facing truth table)

Use this in Settings copy, tooltips, and optional help drawer. Values reflect current public policies (verify before release):

| Provider | Anonymous access | Free API key | Paid tier | Key benefits (concise UI copy) |
|----------|------------------|--------------|-----------|--------------------------------|
| **OpenAlex** | ✓ (~$0.10/day) | ✓ (~$1/day) | Optional higher limits | “Free key increases daily search budget ~10×.” |
| **NCBI** (PubMed, PubChem) | ✓ (3 req/sec) | ✓ (10 req/sec) | No | “Free NCBI key raises rate limit for PubMed and PubChem.” |
| **arXiv** | ✓ | N/A (no keys) | N/A | “No API key available. Qube respects arXiv’s 3-second request interval.” |
| **Crossref** | ✓ | N/A (polite pool / mailto) | Optional | “No key required. Uses public REST API.” |
| **INSPIRE-HEP** | ✓ | N/A | N/A | “Open API; rate limits apply per IP.” |
| **DBLP** | ✓ | N/A | N/A | “Open API.” |
| **RePEc** (EconBiz) | ✓ (may be limited) | Optional key | — | “Optional key for EconBiz API access.” |
| **CourtListener** | Limited without account | ✓ (account token) | Membership tiers | “Free account token unlocks higher API limits.” |
| **Semantic Scholar** | ✗ / very limited | ✓ (required for API) | — | “API key required when this source is enabled.” |
| **NASA ADS** | ✗ | ✓ (required) | — | “API key required (not yet live in Qube).” |

**UI badge vocabulary** (consistent labels):

- `Anonymous` — no key configured; anonymous limits apply
- `Connected` — user key validated
- `Not configured` — key required for this source but absent (retrieval disabled or stub)
- `Env override` — dev/CI env var active (show in status panel for transparency, hide key value)

### 8.4 Settings UI structure

Extend **Settings → Knowledge** with two related subsections:

#### A. Preferred sources (existing)

Keep discipline-group checkboxes — controls **which adapters may run**, unchanged in behavior.

#### B. Provider credentials (new)

Grouped by domain, **one credential block per provider id**:

**Scientific sources**

| Row | Checkbox | Key field | Footer copy |
|-----|----------|-----------|-------------|
| OpenAlex | Uses existing enable semantics or read-only “used when enabled” | Optional | Anonymous mode + key benefits |
| NCBI | Shared for PubMed + PubChem | Optional | Rate limit explanation |
| Semantic Scholar | Disabled until implemented | Required when live | Key required |
| Crossref | — | — | “No API key required” |
| arXiv | — | — | “No API key available” |

**Legal sources**

| CourtListener | Optional token | Free account benefits |

**Finance sources**

| SEC EDGAR | — | “No API key required” (User-Agent only) |

Each key row includes:

- **Test connection** — minimal probe (OpenAlex: `GET /rate-limit`; NCBI: `einfo` with key; CourtListener: profile endpoint)
- **Get free key ↗** — opens provider signup/docs URL from catalog
- **Clear key** — revert to anonymous

### 8.5 Source status panel (“competitive advantage”)

A read-only **Source status** subsection (or collapsible panel) showing live aggregated state:

```
OpenAlex
  Status:  Connected anonymously
  Quota:   Anonymous (~$0.10/day remaining — estimated)
  Health:  Good
  Last used: 2 minutes ago

PubMed (NCBI)
  Status:  Anonymous
  Quota:   3 requests/sec policy
  Health:  Good

CourtListener
  Status:  Not configured
  Quota:   —
  Health:  —
```

**Data sources:**

| Field | Source |
|-------|--------|
| `connection_mode` | `credential_resolver` |
| `quota_label` | Last OpenAlex `X-RateLimit-*` headers; static copy for anonymous NCBI |
| `health` | Circuit breaker + recent 429/503 rate from `http_metrics` |
| `last_used` | Last HTTP request timestamp per provider |
| `last_test_result` | Settings “Test connection” outcome |

Refresh: on Settings open, after Test, periodically (e.g. 60s) while Settings visible — not on every chat turn (avoid noise).

### 8.6 Catalog extension: `ProviderCredentialSpec`

Proposed registry (`core/knowledge/provider_credentials.py`) — separate from duplicate `ADAPTER_CATALOG` rows:

```python
@dataclass(frozen=True)
class ProviderCredentialSpec:
    provider_id: str           # "openalex", "ncbi", "courtlistener"
    label: str                 # "OpenAlex", "NCBI (PubMed & PubChem)"
    adapter_ids: tuple[str, ...]
    supports_anonymous: bool
    supports_free_api_key: bool
    paid_tier_available: bool
    key_required: bool         # True → retrieval blocked without key
    signup_url: str
    docs_url: str
    anonymous_summary: str     # "Current mode: Anonymous access"
    key_benefits: str          # Shown under key field
    test_probe: str            # "openalex_rate_limit" | "ncbi_einfo" | ...
```

Refine `AdapterCatalogEntry.requires_api_key`:

- **`key_required`** — cannot use adapter without key (Semantic Scholar, NASA ADS)
- **`key_optional`** — works anonymously; key improves quota (OpenAlex, NCBI) — new flag or enum `credential_mode: none | optional | required`

### 8.7 Security & storage

- Store user keys in `SettingsStore` under `qube.knowledge.provider_credentials` as `{ "openalex": { "api_key": "..." }, "ncbi": { "api_key": "..." } }`.
- Consider OS keychain integration as **future enhancement** (Slice 12+); v1 may use existing settings file encryption patterns if present, otherwise document local-disk risk like other desktop apps.
- Settings UI: password-echo fields; show last 4 chars only in status panel when connected.
- Env vars **override** user keys for dev — status panel shows “Env override” without revealing value.

### 8.8 Integration with HTTP layer

```
credential_resolver.resolve("openalex")
        │
        ▼
knowledge_http.get(..., credential=cred)
        │
        ├── inject api_key query param
        ├── select scheduler tier (anonymous vs keyed)
        └── record credential_mode in http_metrics (never the secret)
```

On `BudgetExhausted` from Slice 4:

→ emit `ProviderLimitEvent(provider_id, kind=daily_quota|rate_limit, resets_at)`
→ Slice 11 notification handler consumes event

---

## 9. Testing strategy

| Level | Scope |
|-------|-------|
| **Unit** | Scheduler token bucket, arXiv serialization, retry policy with mocked responses, metrics aggregation |
| **Adapter** | Each adapter uses `http_client`; fixture mode unchanged (`QUBE_KNOWLEDGE_FIXTURES=1`) |
| **Pipeline** | Scientific pipeline diag includes `http_summary`; cache hit → zero HTTP |
| **Eval** | Live eval with `--http-report`; gate on discipline primary rate **and** max 429 rate per run |
| **Manual QA** | Short checklist: keys configured, repeat `@evidence` query shows cache, chemistry query no NCBI storm |
| **Credentials UI** | Test connection mocks; persistence; anonymous default; limit notification copy |
| **Security** | Grep audit logs / traces for key patterns; resolver never logs secrets |

No live HTTP in default CI; live eval remains opt-in.

---

## 10. Rollout & risk

| Risk | Mitigation |
|------|------------|
| Slower retrieval from throttling | Conservative rates; cross-host parallelism preserved |
| Behavior change in tiered fan-out (Slice 8) | Behind feature flag; last priority |
| Key management / user confusion | Optional keys; clear “free key” copy; works without keys; §8.3 truth table |
| Users assume API key = paid | Explicit “Free API key” labels; provider matrix in Settings |
| Scheduler deadlock | Timeouts on `acquire`; max wait logging |
| Over-caching stale metadata | TTL caps; DOI cache opt-in |
| Duplicate OpenAlex rows in Settings | Credential UI keyed by `provider_id`, not catalog UI group |

**Rollout:** Slices 1–4 behind no user-visible flags (internal improvement). Slice 10–11 user-visible. Slice 8 flag-gated. Document env vars in release notes when Slice 2 ships; Settings UI when Slice 10 ships.

---

## 11. Manual QA checklist (post Slice 4 + 10)

1. Fresh install: all credential fields empty — `@evidence` queries succeed (anonymous).
2. Settings → Knowledge: OpenAlex row shows “Current mode: Anonymous access” and free-key benefits text.
3. Paste valid OpenAlex key → Test connection → Status shows Connected; `/rate-limit` budget reflects keyed tier.
4. Paste NCBI key → PubMed and PubChem retrieval both use keyed rate (metrics).
5. Clear key → returns to anonymous mode without restart.
6. Repeat `@evidence` query within 1h — cache hit, zero HTTP.
7. Simulate quota exhaustion — notification appears with Settings link; no modal blocker.
8. Run live eval once; confirm HTTP report totals and discipline gates.

---

## 12. Out of scope (this plan)

- Bulk download / OAI-PMH harvesting
- Qube-operated API keys or proxy (users bring their own keys only)
- Paid tier billing or in-app purchases for provider quotas
- Multi-user rate-limit proxy / gateway
- Replacing `scientific_query_planner` or discipline pack registry
- Citation enrichment pipelines that fan out to many DOI lookups (future work; measure first)
- OS keychain storage (future; §8.7)

---

## 13. Success metrics

After Slices 1–4 shipped:

| Metric | Target |
|--------|--------|
| Live eval 429 rate | <5% of HTTP requests per 12-query run (with keys) |
| Live eval 503 retry recovery | ≥90% of 503s succeed within 3 retries |
| Repeated query cache hit | 100% HTTP avoidance within TTL |
| p95 scientific retrieval latency | ≤+500ms vs pre-scheduler baseline (acceptable tradeoff) |
| Eval discipline primary rate | ≥70% stable across 3 same-day runs |

**Credentials UX (after Slice 10–11):**

| Metric | Target |
|--------|--------|
| Anonymous `@evidence` success rate | ≥95% for typical single-user daily usage |
| Settings Test connection success | ≥99% with valid user key |
| Limit notification → key added conversion | Track optionally; no target — informational only |
| User reports “thought key was paid” | Zero in manual QA feedback |

---

## 14. References

- [OpenAlex — Authentication & rate limits](https://developers.openalex.org/api-reference/authentication)
- [OpenAlex — Error handling](https://developers.openalex.org/api-reference/errors)
- [NCBI E-utilities — Usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [arXiv API — Terms of use (rate limits)](https://info.arxiv.org/help/api/tou.html)
- [INSPIRE-HEP REST API — Rate limiting](https://github.com/inspirehep/rest-api-doc)
- [PubChem PUG REST — Usage policies](https://academic.oup.com/nar/article/46/W1/W563/4990016)
- [Wikimedia APIs — Rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits)
- [CourtListener API — Overview & rate limits](https://wiki.free.law/c/courtlistener/help/api/rest/v4/overview)

---

## 15. Document history

| Date | Change |
|------|--------|
| 2026-06-25 | Initial plan from throttling analysis + external review |
| 2026-06-25 | §8 user-managed credentials, hybrid model, source status UI; Slices 9–11 |
