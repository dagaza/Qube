# Web Content Fetch — Design & Implementation Plan

**Status:** Draft v4 — MVP scope, source profiles & fetch provenance  
**Date:** 2026-07-14  
**Parent:** [External Knowledge Platform Plan](./external_knowledge_platform_plan.md)  
**Related:** [Knowledge Adapter HTTP Resilience Plan](./knowledge_adapter_http_resilience_plan.md), [ADR 003 — Evidence convergence](./adr/003-evidence-convergence.md), [ADR 001 — Skills orthogonal to routing](./adr/001-skills-orthogonal-to-routing.md), [Cognitive Router](./cognitive_router.md), [Knowledge Platform Evolution Review](./knowledge_platform_evolution_review.md), [Sidecar tasks](./sidecar_tasks.md)

This document specifies how Qube should extend the external knowledge platform to **fetch and extract content from web pages** that lack structured APIs — using deterministic HTTP scraping, a **canonical `Document` IR**, capability-based extractors (v1 MVP: Trafilatura + Recipe only), optional Playwright fallback (v1.1+), user-defined **source profiles** (`@[tool:user:…]`) for domain-specific site lists, fetch provenance for debugging and trust, and section-level ranking so small local LLMs (4B–9B) receive concise, relevant evidence rather than raw HTML dumps.

It is intended as a **review artifact**: sit on it, refine scope, then implement in slices.

---

## 0. Executive summary

### Problem

Today, Qube’s live web path (`@internet` / `general_web`) returns **DuckDuckGo SERP snippets only**. Result pages are not fetched. The audit trail explicitly notes: *“SERP snippets only; result pages are not fetched.”* For recipes, gaming walkthroughs, product reviews, and long-form articles, snippets are often insufficient or misleading.

The external knowledge platform plan already envisioned **selective fetch** after SERP discovery, but it was not implemented for general web. API adapters (PubMed, CourtListener, etc.) are mature; **non-API web content** is the remaining gap.

### Proposed solution

Add a **fetch subsystem** inside the existing `general_web` knowledge service, gated by **retrieval profiles** and **composer pins** — not a separate `web_content` service (deferred until the implementation stabilizes).

```
Composer @-tool → general_web → Pipeline
    Discovery → Fetch → Extract → Document → Chunk → Rank → [optional sidecar polish] → EvidenceBundle
```

Key design choices (v4):

1. **Do not replace `EvidenceBundle`.** Internal types (`CandidateUrl`, `FetchResult`, `Document`) stay below the ADR 003 boundary.
2. **Defer `web_content` as a separate KnowledgeService.** Extend `general_web` + `RetrievalProfileSpec` (`fetch_url_count`, `playwright_allowed`).
3. **Hard-separate discovery from extraction.** `DiscoveryProvider` returns URLs; fetch/extract pipeline unchanged for DDG, RSS, bookmarks, or user site lists.
4. **Canonical `Document` IR (central)** — mandatory layer between HTML and chunking:

   ```
   HTML → Extractor → Document → Chunker → EvidenceObject
   ```

   All page types converge on one `Document` before ranking.
5. **Capability-based extractors only** — `supports(url, html) → confidence` + `priority` for tie-breaks. No routing table, no content-type ontology. Extractors advertise observable page signals (JSON-LD, ordered lists, code blocks).
6. **Minimal built-in composer tools** — only `@internet`, `@fetch`, `@recipe`. Everything else (DIY, gaming, docs) → **source profiles** (`@[tool:user:…]`).
7. **No discovery ontology in v1 MVP** — DDG scoped by `site_bias` (preset or `@recipe` default list) and query keywords only. Extractor selection is always post-fetch via `supports()`.
8. **Deterministic fetch ladder:** HTTP → blocker detection → extractors → Playwright (v1.1+, opt-in).
9. **Structured failures, not garbage:** Blocker HTML never becomes `EvidenceObject` content.
10. **Fetch provenance by default** — every turn records discovery → selection → extractor → confidence (Inspector Explain).
11. **Section-level evidence:** Rank `Document.sections`; emit small `EvidenceObject`s within char budget.
12. **Three built-ins only:** `@internet` (SERP), `@fetch` (generic fetch), `@recipe` (recipe site list + JSON-LD). Procedural/docs/gaming → source profiles.

### Delivery bands

**v1 MVP (ship first)** — prove the deterministic pipeline on real pages:

| Slice | Deliverable |
|-------|-------------|
| **M1** | `DiscoveryProvider` + DDG (`CandidateUrl`) |
| **M2** | `FetchResult` + fetch engine + blocker detection |
| **M3** | `Document` model + extractor plugin registry (`priority`, `supports()`) + Trafilatura |
| **M4** | Section chunker + embedding ranker → `EvidenceObject`s |
| **M5** | Profile-gated `general_web` integration + `@fetch` pin |
| **M6** | `RecipeExtractor` + `@recipe` pin + default recipe `site_bias` |
| **M7** | Fetch provenance trail + Inspector Explain (minimal) |
| **M8** | Source profiles (minimal): `site_bias`, `fetch_url_count` on `general_web` presets |

**v1.1+ (after MVP works on real usage)** — defer until justified by failures:

| Slice | Deliverable |
|-------|-------------|
| **P1** | Pagination crawl policy |
| **P2** | `ProceduralExtractor`, `DocumentationExtractor` (plugin additions) |
| **P3** | Playwright worker |
| **P4** | Rich source profiles (`preferred_extractors`, ranking weights) |
| **P5** | Optional sidecar section polish |
| **P6** | RSS / bookmark discovery providers |

### Non-goals (v1 MVP)

- LLM-driven browser agents as foreground chat fallback
- Sidecar/LLM for intent classification or URL selection
- Built-in composer tools beyond `@internet`, `@fetch`, `@recipe` (no `@howto`, `@guides`)
- Content-type ontology for routing (`procedural`, `documentation`, `product_review`, …)
- `ProceduralExtractor` / `DocumentationExtractor` (v1.1+ plugins)
- Playwright (v1.1+)
- A separate `web_content` KnowledgeService (deferred)
- Paywalled full-text scraping or credential vaults for news sites
- Scrapy-scale sitewide crawling
- Replacing cognitive router domain logic
- Per-site entries in `SEARCH_FUNCTIONS` for hundreds of hosts

---

## 1. Problem statement

### 1.1 User scenarios poorly served today

| Scenario | Why SERP snippets fail |
|----------|------------------------|
| **Recipes** | Snippet shows blog intro, not ingredients/steps |
| **Procedural how-to** (gaming, DIY, assembly) | Multi-page step lists; snippet is one paragraph |
| **DIY / furniture** | IKEA-style assembly steps not in snippet |
| **Product comparisons** | Spec tables not in snippet |
| **Software / API docs** | Code blocks and nav-stripped content need extract |
| **How-to articles** | Steps truncated or absent |
| **Local business hours** | Sometimes in snippet; often requires page extract |

### 1.2 What works today (do not break)

| Path | Mechanism | Fetch depth |
|------|-----------|-------------|
| `@evidence` / `@science` | 58 API adapters | Abstracts, metadata |
| `@legal` | CourtListener REST | Opinion snippets |
| `@trusted` | Wikipedia API | Full intro extract |
| `@internet` | DDG SERP | `snippet_only` |
| `bailii` adapter | Light HTML scrape (regex) | Precedent for non-API fetch |

### 1.3 Platform gaps

| Gap | Location | Impact |
|-----|----------|--------|
| No page fetch after SERP | `general_web` / `pipeline.py` | Shallow evidence |
| Naive prompt truncation | `ui_adapter.bundle_to_prompt_context()` | `body[:char_budget]` — bad for long pages |
| `target_site` not wired | `mcp/internet_tool.py` | Site-scoped search unused |
| Selective fetch planned but unbuilt | `external_knowledge_platform_plan.md` §7 | Design debt |
| No blocker taxonomy | — | Risk of feeding Cloudflare HTML to LLM |
| No canonical document IR | — | Each extractor would emit incompatible shapes |

### 1.4 Constraints (Qube-specific)

- **Local-first desktop app** — Playwright adds binary size, RAM, latency; must be opt-in.
- **Small local LLMs** — default skills char budget ~1200; evidence must be selective.
- **ADR 003** — all v2 paths must produce `EvidenceBundle`.
- **ADR 001** — cognitive router picks MEMORY/RAG/WEB; sidecar never picks routes or URLs.
- **Sidecar constraints** — Qwen3 1.7B, CPU-only, 1500ms foreground timeout (`sidecar_tasks.md`).
- **Retrieval profiles** — `RetrievalProfileSpec` already controls orchestration (Fast / Balanced / Thorough); fetch depth belongs here.
- **Existing HTTP stack** — `knowledge_get()`, egress policy, host scheduler, negative cache must be reused.

---

## 2. Goals & success criteria

### 2.1 Goals

1. **Discover** candidate URLs via a pluggable provider protocol (DDG first).
2. **Fetch** top URLs and extract readable main content deterministically.
3. **Normalize** all extractions into a canonical `Document` before chunking.
4. Select extractors by **`supports(url, html)` confidence**, not LLM classification.
5. Route fetch depth via **retrieval profiles**, **composer pins**, and **source profiles** (not cognitive router).
6. Record **fetch provenance** on every turn (discovery → selection → extractor → confidence).
7. Detect anti-bot / JS / paywall blockers; return structured failures.
8. Emit section-ranked `EvidenceObject`s within retrieval and prompt budgets.
9. Integrate with RetrievalRecord, Inspector, Replay, and stage traces (trace schema v3).
10. Optional Playwright (v1.1+ P3); optional sidecar polish (v1.1+ P5).

### 2.2 Success criteria (measurable)

| Criterion | Target |
|-----------|--------|
| Recipe query with `@recipe` | ≥1 `EvidenceObject` with structured ingredients/steps in `raw_metadata` |
| User source profile `@[tool:user:serious-eats]` | Fetches only `site_bias` domains |
| Fetch provenance on recipe turn | Inspector shows URL, extractor, confidence chain |
| General article fetch (Balanced+) | `fetch_status: full_extract`; excerpt ≤ 800 chars per object |
| Cloudflare-blocked URL | `coverage: none` or explicit warning; **zero** challenge HTML in prompt |
| 4B model turn | Total web evidence ≤ `total_prompt_char_budget` without head truncation |
| Foreground latency (HTTP-only) | p95 ≤ 4s for single-URL fetch on Balanced profile |
| Foreground latency (Playwright) | p95 ≤ 15s; max 1 Playwright fetch per turn |
| Empty fetch failure | LLMWorker empty-source downgrade — no hallucinated `[W]` citations |
| Extractor replay | RetrievalRecord shows `extractor_name`, `extractor_version`, `confidence` |

---

## 3. Architectural fit

### 3.1 Existing spine (unchanged contract)

```
Composer @-mention
    → parse_attachments() / resolve_attachment_routing()
    → LLMWorker forces WEB + sets _composer_knowledge_tool
    → resolve_turn_knowledge_service()          # stays general_web for @internet/@fetch/@recipe
    → KnowledgeService.retrieve(RetrievalContext)
    → Pipeline (extended EvidencePipeline)
    → EvidenceBundle
    → run_v2_web_retrieval() → prompt [W] blocks
```

This plan **extends `general_web` pipeline modules** — not a parallel retrieval harness.

### 3.2 Pipeline stages (separation of concerns)

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌───────────┐     ┌─────────────┐
│  Discovery  │ ──► │    Fetch    │ ──► │  Extraction  │ ──► │ Document  │ ──► │ Chunk+Rank  │
│  providers  │     │   engine    │     │  extractors  │     │ normalize │     │  → Evidence │
└─────────────┘     └─────────────┘     └──────────────┘     └───────────┘     └─────────────┘
     URLs only         raw HTML           capability-based      canonical IR      EvidenceObject
```

Discovery and extraction are **independently extensible**. Future discovery sources (RSS, bookmarks, browser history with permission, user site lists) plug in without changing extractors.

### 3.3 New components (target modules)

```
core/knowledge/discovery/
    types.py              # CandidateUrl
    registry.py           # DiscoveryProvider protocol + get_discovery_provider()
    duckduckgo.py         # wraps existing DDG adapter

core/knowledge/fetch/
    types.py              # FetchResult, BlockerReason, CrawlPolicy
    engine.py             # Fetch ladder orchestration
    blockers.py           # Cloudflare, paywall, JS-shell heuristics
    pagination.py         # rel=next, max pages, byte budget

core/knowledge/document/
    types.py              # Document, DocumentSection, DocumentList, DocumentTable
    normalize.py          # HTML → Document helpers

core/knowledge/extractors/
    base.py               # Extractor protocol: supports(), extract(), metadata
    registry.py           # register + select_best_extractor()
    trafilatura_extractor.py
    recipe_extractor.py   # recipe-scrapers; JSON-LD supports()
    bs4_patterns.py
    # v1.1+: procedural_extractor.py, documentation_extractor.py

core/knowledge/fetch/
    section_chunker.py
    section_ranker.py

core/knowledge/site_bias.py        # site: hints for DDG from preset / @recipe defaults

core/knowledge/fetch_provenance.py # structured provenance trail per turn

core/knowledge/pipeline_general_web.py   # extended fetch stages (or extend pipeline.py)

workers/browser_fetch_worker.py    # Optional Playwright (v1.1+ P3)
```

**Deferred (revisit post-stabilization):**

```
core/knowledge/services/web_content.py
core/knowledge/pipeline_web_content.py
SERVICE_WEB_CONTENT constant
```

### 3.4 Type layering

| Type | Layer | Crosses prompt boundary? | Purpose |
|------|-------|--------------------------|---------|
| `CandidateUrl` | Discovery | No | URL + SERP metadata; not evidence |
| `FetchResult` | Fetch engine | No | Transport outcome, blockers, bytes |
| `Document` | Extraction | No | Canonical parsed content |
| `DocumentSection` | Chunking input | No | Heading-bounded text block |
| Adapter row `dict` | Pipeline internal | No | Legacy interop |
| `EvidenceObject` | Normalized source | Yes (via bundle) | Citation-facing |
| `EvidenceBundle` | Turn contract | Yes | ADR 003 boundary |

**Do not** introduce a user-facing `RetrievalResult` parallel to `EvidenceBundle`.

**`FetchResult` vs `Document`:** A failed fetch has no `Document`. A successful fetch with empty extract has `FetchResult.success=True` but `Document.sections=[]` → treated as `empty_extract`.

---

## 4. Routing, profiles & composer pins

### 4.0 Composer token format (built-in vs user-defined)

All composer tools use **one token syntax** via `format_token()` in `core/composer_attachments.py`:

```
@[tool:{tool_id}]
```

There is no shorter alternate format. The **tool id** distinguishes built-in Qube tools from user-created presets.

| Kind | Token (as inserted in composer) | `tool_id` | Registered in |
|------|----------------------------------|-----------|---------------|
| Built-in | `@[tool:internet]` | `internet` | `COMPOSER_TOOLS` |
| Built-in (planned) | `@[tool:fetch]` | `fetch` | `COMPOSER_TOOLS` |
| Built-in advanced | `@[tool:recipe]` | `recipe` | `COMPOSER_TOOLS` (`advanced: True`) |
| User preset | `@[tool:user:serious-eats]` | `user:serious-eats` | `composer_preset_tools()` |
| Source pin | `@[tool:source:pubmed]` | `source:pubmed` | dynamic |

User preset ids are parsed by `parse_user_preset_tool()` — e.g. `user:biology` → preset id `biology`. Preset ids must match `PRESET_ID_RE` and **cannot** collide with `RESERVED_PRESET_IDS` (must include planned built-ins: `fetch`, `recipe` — not `howto`).

**Built-in web fetch tools (v4):** only three — `internet`, `fetch`, `recipe`. All domain-specific tools are **source profiles** in Settings → My knowledge.

**Display vs token:** Settings UI may show “My recipes” as the label; the token remains `@[tool:user:my-recipes]`, not `@[tool:recipe]` (reserved for the built-in).

### 4.1 Four routing layers (do not collapse)

```
Layer 1 — Cognitive router (mcp/cognitive_router.py)
    Decides: MEMORY | RAG | WEB | HYBRID | NONE
    Does NOT decide: recipe vs procedural vs article; does NOT pick URLs

Layer 2 — Composer pin → RetrievalContext overrides (NOT a new service)
    @internet       → general_web; fetch_url_count=0 (unless profile overrides)
    @fetch          → general_web; fetch_url_count forced ≥ 1
    @recipe         → general_web; fetch_url_count ≥ 1; default recipe site_bias
    @[tool:user:X]  → general_web source profile (site_bias, fetch_url_count, …)

Layer 3 — Retrieval profile → orchestration policy (retrieval_profiles.py)
    Fast      → SERP only (fetch_url_count=0)
    Balanced  → SERP + fetch top 1
    Thorough  → SERP + fetch top 3; Playwright allowed if setting enabled

Layer 4 — Extractor selection (capability-based, per URL)
    All registered extractors: supports(url, html) → confidence
    Pick highest above threshold; fallback Trafilatura (confidence 0.3)
```

### 4.2 Why defer `web_content` as a separate service

`RetrievalProfileSpec` already materializes orchestration into `RetrievalBudget` via `RetrievalContext.retrieval_profile`. The evolution review states: *“Retrieval Profiles answer how hard/fast/local to search.”* Fetch depth is orchestration — it belongs in profiles.

| Approach | Verdict |
|----------|---------|
| Separate `web_content` service (v1 plan) | **Deferred** — premature boundary; duplicates DDG; another bundle builder path |
| Extend `general_web` + profiles (v2) | **Adopt** — one pipeline; Settings already expose profiles |
| Composer pins without new service | **Adopt** — `@recipe` sets `RetrievalContext` fields, like `@pubmed` sets adapter filter |

**Revisit `web_content` service when:** bundle warnings, ranking policy, or observability diverge enough that `general_web` accumulates unmaintainable `if fetch_enabled` branches.

### 4.3 Retrieval profile extensions

Add fields to `RetrievalProfileSpec` in `retrieval_profiles.py`:

```python
@dataclass(frozen=True)
class RetrievalProfileSpec:
    ...
    fetch_url_count: int = 0           # 0 = SERP only; 1/2/3 = fetch top-K
    playwright_allowed: bool = False   # permission only; still requires Settings toggle
    pagination_allowed: bool = False
```

| Profile | SERP | `fetch_url_count` | Playwright | Pagination |
|---------|------|-------------------|------------|------------|
| **Fast** | yes | 0 | no | no |
| **Balanced** | yes | 1 | no | no |
| **Thorough** | yes | 3 | if setting enabled | rel=next max 3 |

Composer `@fetch` / `@recipe` **override** `fetch_url_count` to at least 1 regardless of profile.

`@internet` with Fast profile = SERP only. `@internet` with Balanced profile = SERP + fetch top 1 (if user opts in globally).

### 4.4 Cognitive router integration (minimal)

**No new router lanes.** No `discovery_bias_hint` ontology in v1 MVP. Unpinned WEB turns: DDG with no site filter → fetch (if profile allows) → `supports()` picks extractor.

Composer `@recipe` applies built-in recipe `site_bias`. Source profiles apply their `site_bias`. **No sidecar classifier for intent in v1.**

### 4.5 Built-in composer tools — three only

Built-ins stay **extremely small**. Domain labels (“My Gaming”, “My DIY”, “My Linux Docs”) belong in **source profiles** (§4.6).

| Token | Service | `RetrievalContext` overrides | Rationale |
|-------|---------|------------------------------|-----------|
| `@internet` | `general_web` | `fetch_url_count` from profile (0 on Fast) | SERP; unchanged default |
| `@fetch` | `general_web` | `fetch_url_count ≥ 1` | Generic discover + fetch + best extractor |
| `@recipe` | `general_web` | `fetch_url_count ≥ 1`; built-in recipe `site_bias` | JSON-LD Recipe — structured signal, like `@pubmed` |

**Dropped: `@guides`** — too ambiguous (travel, buying, style, game, repair guides).

**Dropped: `@howto`** — still a semantic domain label. Procedural content (gaming, DIY, docs) → source profiles only, e.g. `@[tool:user:ikea-diy]`, `@[tool:user:my-gaming]`, `@[tool:user:linux-docs]`.

Advanced/hidden flags for `@recipe` mirror `@pubmed` / `@science` pattern.

### 4.6 Source profiles — primary domain mechanism

**Source profile** is the user-facing name for a **knowledge preset** with `base_service: general_web`. It is richer than `site_bias` alone and evolves toward: preferred domains → preferred extractors → ranking weights → output shape (v1.1+).

Source profiles are the **main way** to define `@finance`-style tools for specific websites.

**Today (codebase gap):** `KnowledgePreset` only allows `base_service` in `{scientific_evidence, finance_knowledge, legal_knowledge}`. Slice M8 must add `general_web` to `ALLOWED_BASE_SERVICES` and extend the preset schema.

**Analogy:**

| Built-in `@finance` | Source profile `@[tool:user:serious-eats]` |
|---------------------|--------------------------------------------|
| Routes to `finance_knowledge` | Routes to `general_web` |
| Pins SEC EDGAR etc. via adapter list | Pins domains via `site_bias` |
| API adapters return rows | Discovery + fetch + extractors |

**v1 MVP source profile fields:**

```json
{
  "id": "serious-eats",
  "label": "My Recipes",
  "description": "Recipes from seriouseats.com only",
  "base_service": "general_web",
  "site_bias": ["seriouseats.com"],
  "fetch_url_count": 2,
  "ranking_profile": "generic",
  "query_planner": "keyword_extract"
}
```

Token: `@[tool:user:serious-eats]` — **not** `@[tool:recipe]` (built-in uses Qube’s default recipe site list).

**Example — “My Gaming” source profile:**

```json
{
  "id": "my-gaming",
  "label": "My Gaming",
  "base_service": "general_web",
  "site_bias": ["fandom.com", "ign.com"],
  "fetch_url_count": 2
}
```

**Example — “My DIY” / “My Linux Docs”:**

```json
{
  "id": "linux-docs",
  "label": "My Linux Docs",
  "base_service": "general_web",
  "site_bias": ["wiki.archlinux.org", "manpages.debian.org"],
  "fetch_url_count": 2
}
```

**v1.1+ source profile fields (defer):** `preferred_extractors`, `pagination_allowed`, `section_ranking_weights` (e.g. boost code blocks for docs).

**Built-in `@recipe` vs source profile:** Both use `general_web`. Built-in applies default recipe `site_bias`; source profile overrides sites. Extractor selection is always `supports()` on the fetched page.

**Routing:** `parse_user_preset_tool("user:serious-eats")` → load preset → apply `RetrievalContext` fields.

Presets remain **source/discovery bundles**, not pipelines (per evolution review).

### 4.7 Observable page signals (not a routing ontology)

Do **not** maintain a semantic taxonomy (`recipe`, `procedural`, `documentation`, `product_review`) for routing. Those labels drift and overlap (a product review is often an article with tables).

Instead, think in **observable page properties** that extractors detect via `supports()`:

| Observable signal | Example extractor (v1 / v1.1+) |
|-------------------|-------------------------------|
| JSON-LD `@type: Recipe` | `RecipeExtractor` (v1 MVP) |
| Dense `<ol>` / “Step N” headings | `ProceduralExtractor` (v1.1+ plugin) |
| Code-block density, `<article>` docs layout | `DocumentationExtractor` (v1.1+ plugin) |
| Main prose + metadata | `TrafilaturaExtractor` (v1 MVP fallback) |
| Tables + ratings | Handled by Trafilatura + section ranker (no dedicated extractor v1) |

Discovery (DDG) uses **`site_bias` only** — from source profile or `@recipe` defaults — not content-type enums.

### 4.8 Acknowledged use-cases (by path, not built-in tools)

| Use case | v1 MVP path | v1.1+ |
|----------|-------------|-------|
| Recipes | `@recipe` or `@[tool:user:…]` + recipe `site_bias` | — |
| Gaming walkthroughs | `@[tool:user:my-gaming]` | Pagination P1; ProceduralExtractor P2 |
| DIY / furniture | `@[tool:user:ikea-diy]` | ProceduralExtractor P2 |
| Product reviews | `@fetch` or source profile | Trafilatura + table sections |
| Software / API docs | `@[tool:user:linux-docs]` | DocumentationExtractor P2 |
| News / current events | `@fetch` | Paywall failures common |
| Coding tutorials | Source profile with doc sites | Code-block ranking weights P4 |
| Health/fitness (non-clinical) | `@fetch` or source profile | **Not** `@pubmed` |
| Forums (Reddit, etc.) | Defer | Poor extract signal |

**Design rule:** Three built-ins only. All domain branding via source profile **labels** (“My Recipes”, “My Gaming”, “My Linux Docs”).

## 5. Discovery subsystem

### 5.1 DiscoveryProvider protocol

Parallel to `SearchFn` in `adapters/registry.py`:

```python
class DiscoveryProvider(Protocol):
    id: str

    def discover(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> list[CandidateUrl]: ...


@dataclass(frozen=True)
class CandidateUrl:
    url: str
    title: str | None
    snippet: str | None
    source: str              # "duckduckgo", "rss", "site_list", ...
    rank: int = 0
```

**Discovery returns URLs, not evidence.** Evidence exists only after `Document` → `EvidenceObject`.

### 5.2 Discovery sources (phased)

| Phase | Provider | Notes |
|-------|----------|-------|
| v1 | `DuckDuckGoDiscovery` | Wraps existing `duckduckgo.py` |
| v2 | `SiteListDiscovery` | User curated domains from presets |
| v3 | `RssDiscovery` | Reuse `rss_atom` connector patterns |
| v4 | `BookmarkDiscovery` | User permission required |

### 5.3 Post-discovery gating

Before fetch:

1. `filter_web_results()` — lexical/embedding relevance gate (existing).
2. Optional `site:` prefix from source profile `site_bias` or `@recipe` default site list.
3. Take top **K** URLs where K = `fetch_url_count` from profile or composer override.

Wire `target_site` from `mcp/internet_tool.py` into `RetrievalContext` (currently unused).

### 5.4 Site bias (`site_bias.py`)

**Discovery-only** — scopes DDG queries. **Does not** select extractors (that is `supports()` on fetched HTML).

| Source of `site_bias` | When applied |
|-----------------------|--------------|
| Source profile `site_bias` field | `@[tool:user:…]` attached |
| Built-in `@recipe` default list | `@[tool:recipe]` attached |
| None | `@fetch` or unpinned WEB — no `site:` filter |

Implementation: append `site:domain1.com OR site:domain2.com` to DDG query when `site_bias` is non-empty. No `discovery_bias` enum in v1 MVP.

---

## 6. Fetch subsystem

### 6.1 FetchResult (transport layer)

```python
@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str | None
    success: bool
    failure_reason: str | None     # BlockerReason enum
    status_code: int | None
    content_type_header: str | None
    html: str | None               # raw HTML for extractors; never passed to LLM
    fetch_tier: str                # "http" | "playwright"
    page_count: int = 1
    total_bytes: int = 0
    latency_ms: float = 0.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)
```

Note: `FetchResult` no longer carries `extracted_text` or `structured` — those live on `Document` after extraction.

### 6.2 Blocker taxonomy

| `failure_reason` | Detection signal | User-facing warning | Retry |
|------------------|------------------|---------------------|-------|
| `cloudflare` | “just a moment”, “checking your browser”, CF challenge DOM | Site blocked automated access | Playwright if enabled |
| `js_rendered` | Empty text, React root shell, `<noscript>` redirect | Page requires JavaScript | Playwright |
| `cookie_wall` | Consent overlay dominates; little main content | Could not bypass consent dialog | Playwright + dismiss script |
| `paywall` | Subscribe/login CTAs; metered wall patterns | Content is paywalled | None |
| `robots_disallowed` | robots.txt disallow for URL | Site disallows fetching | None |
| `timeout` | HTTP / Playwright timeout | Site did not respond in time | Negative-cache host |
| `oversized` | Exceeds `max_fetch_bytes` | Page too large | Section-only extract |
| `empty_extract` | HTTP 200 but `Document.sections` empty | No readable content found | Try next URL |
| `egress_blocked` | `EgressPolicyError` | URL blocked by policy | None |

**Invariant (ADR 004):** Blocker HTML must never appear in `EvidenceObject.excerpt` or `full_text`.

### 6.3 Deterministic fetch ladder

```
For each CandidateUrl (relevance-ordered, up to fetch_url_count):

  Tier 1 — HTTP GET via knowledge_get()
      • EgressPolicy (max bytes, SSRF)
      • Blocker detection on response body
      • If blocked → FetchResult(success=False) → maybe escalate

  Tier 2 — Playwright (opt-in setting + Thorough permission + budget)
      • Only if failure_reason ∈ {cloudflare, js_rendered, cookie_wall}
      • Separate worker process; max 1 per turn
      • Re-fetch HTML → re-run blocker detection

  Tier 3 — LLM browser agent
      • NOT in v1 foreground path
      • Future: @research async job only
```

Extraction runs **after** clean HTML is obtained — not inside the fetch tier loop.

### 6.4 Pagination & multi-page guides

```python
@dataclass(frozen=True)
class CrawlPolicy:
    max_pages: int = 3
    max_total_bytes: int = 524_288
    max_latency_ms: int = 8000
    follow_rel_next: bool = True
    follow_selectors: tuple[str, ...] = (
        'a[rel="next"]',
        '.pagination a.next',
        'a[aria-label="Next"]',
    )
```

Algorithm:

1. Fetch page 1 → extract → `Document`.
2. Score sections against query.
3. If coverage weak **and** `rel=next` exists **and** budget remains → fetch page 2.
4. Merge into single `Document`; dedupe sections by heading hash.
5. Stop at `max_pages` or `max_total_bytes`.

Enabled when `pagination_allowed=True` on source profile or Thorough profile (v1.1+ P1). Disabled on Fast/Balanced in v1 MVP.

### 6.5 HTTP reuse (mandatory)

All fetches **must** use `knowledge_get()` from `http_client.py`:

- Egress validation (`egress_policy.py`)
- Per-host scheduling (`host_scheduler.py`)
- Negative cache for persistently blocked hosts
- HTTP metrics (`http_metrics.py`)

Playwright navigations must pass `validate_url()`.

---

## 7. Canonical Document model

### 7.1 Why a Document IR

The v1 plan skipped a stable intermediate representation, jumping from HTML to sections. A canonical `Document` simplifies:

- Multiple extractors converging on one shape
- Section chunker operating on consistent structure
- Replay/Inspector showing parsed content before ranking
- Optional sidecar polish on `Document`, not raw HTML
- Versioned extractor metadata attached at document level

### 7.2 Document schema

```python
@dataclass
class DocumentSection:
    heading: str | None
    level: int = 0           # h1=1, h2=2, ...
    text: str
    list_items: tuple[str, ...] = ()
    char_offset: int = 0


@dataclass
class DocumentTable:
    caption: str | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass
class Document:
    url: str
    title: str | None
    author: str | None = None
    date: str | None = None
    sections: list[DocumentSection]
    tables: list[DocumentTable] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)   # alt, src — not fetched v1
    structured_data: dict[str, Any] = field(default_factory=dict)  # JSON-LD, recipe schema
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


@dataclass(frozen=True)
class DocumentMetadata:
    extractor_name: str
    extractor_version: str
    extractor_confidence: float
    fetch_tier: str
    page_count: int = 1
    language: str | None = None
```

All content types (recipes, procedural how-to, articles, documentation, Wikipedia-style pages) produce a `Document`. Differences live in `structured_data` and section structure.

### 7.3 Pipeline position

```
HTML (from FetchResult)
    → select_best_extractor(url, html)
    → extractor.extract(html, url) → Document
    → section_chunker(Document)
    → section_ranker(sections, query)
    → [optional sidecar polish on top sections]
    → EvidenceObject(s)
```

---

## 8. Capability-based extractors

### 8.1 Extractor protocol

Replace category-first `EXTRACTOR_REGISTRY` with capability-based selection:

```python
@dataclass(frozen=True)
class ExtractorMetadata:
    name: str
    version: str
    priority: int = 50   # higher wins on equal confidence


class Extractor(Protocol):
    metadata: ExtractorMetadata

    def supports(
        self,
        url: str,
        html: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> float:
        """Return confidence 0.0–1.0 that this extractor can handle the page."""

    def extract(self, html: str, url: str) -> Document: ...
```

### 8.2 Selection algorithm (plugin loop — no routing table)

```python
def select_best_extractor(url: str, html: str) -> tuple[Extractor, float]:
    scores = [(ext, ext.supports(url, html)) for ext in registered_extractors()]
    scores = [(e, c) for e, c in scores if c > 0]
    if not scores:
        return fallback_trafilatura, 0.3
    best = max(scores, key=lambda x: (x[1], x[0].metadata.priority))
    if best[1] < 0.5:
        return fallback_trafilatura, 0.3
    return best
```

No `if domain == gaming` branches. No content-type routing table. New extractors register as plugins with `supports()`, `extract()`, `priority`.

Composer `@recipe` applies built-in recipe `site_bias` for DDG only — it does **not** force `RecipeExtractor` on a non-recipe page. Optional: lower `supports()` threshold to 0.4 when `@recipe` is pinned.

### 8.3 Built-in extractors

**v1 MVP (ship in M3 + M6):**

| Extractor | `priority` | `supports()` signals | Output |
|-----------|------------|---------------------|--------|
| `RecipeExtractor` v1.0 | 90 | JSON-LD `@type: Recipe`; schema.org Recipe microdata; recipe-scrapers hosts | `Document` + `structured_data` (ingredients, steps) |
| `TrafilaturaExtractor` v1.0 | 10 | Always (fallback, confidence 0.3) | General article `Document` |

Pass pre-fetched HTML to recipe-scrapers — **do not double-fetch**.

**v1.1+ plugins (register when MVP proves need):**

| Extractor | `supports()` signals | Notes |
|-----------|---------------------|-------|
| `ProceduralExtractor` | Dense `<ol>`, “Step N” headings | Gaming, DIY — often Trafilatura suffices until proven otherwise |
| `DocumentationExtractor` | Code-block density, docs DOM | Linux docs, API reference |

A product review page is typically **Trafilatura + section ranker** on tables — no dedicated extractor required.

### 8.4 Extractor versioning & observability

Every `Document.metadata` records:

```
extractor_name: "RecipeExtractor"
extractor_version: "1.0.0"
extractor_confidence: 0.98
```

Propagated to:

- `PipelineStageTrace.outputs`
- `EvidenceObject.raw_metadata` (audit)
- `RetrievalRecord` / Replay comparisons

Follows existing patterns: `PROFILE_VERSION`, `DISCIPLINE_PACK_VERSION`, `adapter_readiness.py`.

---

## 9. Section ranking & prompt budgets

### 9.1 Problem

`bundle_to_prompt_context()` truncates with `body[:char_budget]` — unacceptable for fetched content.

### 9.2 Section pipeline

```
Document
    → section_chunker (heading-aware; 200–400 token targets)
    → section_ranker (embedding similarity to semantic_query)
    → MMR diversity (reuse ranking/diversity.py)
    → select top-N sections (N from RetrievalBudget.max_results)
    → [optional sidecar polish — v1.1+ P5]
    → one EvidenceObject per section
```

### 9.3 EvidenceObject mapping (per section)

```python
EvidenceObject(
    title=f"{doc.title} — {section.heading}" if section.heading else doc.title,
    excerpt=section_text[:800],
    full_text=section_text if len(section_text) <= 4000 else None,
    url=doc.url,
    document_type="web_section",  # or "recipe", "procedural_section"
    fetch_status="full_extract",
    relevance_score=section_score,
    authority_score=authority_score_for_url(doc.url),
    raw_metadata={
        "section_index": i,
        "extractor_name": doc.metadata.extractor_name,
        "extractor_version": doc.metadata.extractor_version,
        "extractor_confidence": doc.metadata.extractor_confidence,
        "fetch_tier": doc.metadata.fetch_tier,
        "structured_data": doc.structured_data if i == 0 else None,
    },
)
```

### 9.4 Budget knobs

| Knob | Source | Default |
|------|--------|---------|
| `fetch_url_count` | `RetrievalProfileSpec` / composer pin | 0 (Fast) |
| `max_results` | `RetrievalBudget` | 3 sections |
| `max_fetch_bytes` | `RetrievalBudget` / `EgressPolicy` | 512 KiB |
| `total_prompt_char_budget` | Skills settings | 1200 |
| `max_section_chars` | Profile / constant | 800 |

### 9.5 Bundle warnings & coverage

| Situation | `coverage` | `warnings` |
|-----------|------------|------------|
| All fetches blocked | `none` | `fetch_blocked:cloudflare` |
| Partial success | `adequate` | `partial_fetch` |
| Recipe missing steps | `poor` | `incomplete_recipe` |
| Profile fetch_url_count=0 | varies | `snippet_fallback` |

---

## 10. Sidecar role (v1.1+ P5 — defer)

Not on the v1 MVP critical path. Evaluate after Trafilatura + Recipe + section ranking prove insufficient.

### 10.1 What the sidecar must NOT do

Per ADR 001 and `sidecar_tasks.md`:

- Browse or choose links
- Recover from broken HTML heuristically
- Plan multi-step navigation
- Classify content intent

### 10.2 What the sidecar MAY do (v1.1+ P5)

After deterministic extract + rank, on **top 1–2 sections only**:

- Compress verbose section to concise procedural JSON
- Clean markdown artifacts
- Normalize recipe step numbering
- Generate a short section title

This extends the existing `source_digest` pattern — constrained post-extract transform, not a fetch planner.

### 10.3 Integration approach

```
section_ranker → top sections
    → if sidecar_enabled and section_chars > threshold:
         SidecarTask.source_digest (reuse) OR new SidecarTask.section_polish
    → on timeout: use deterministic excerpt (same fallback as source_digest)
    → EvidenceObject
```

**Not on critical path for Slices 1–8.** Evaluate after recipe-scrapers structured output proves insufficient.

Constraints: 1500ms foreground timeout; CPU-only; assistive only.

---

## 11. Playwright fallback (v1.1+ P3 — defer)

### 11.1 Two-gate permission model

| Gate | Controls |
|------|----------|
| **Retrieval profile** | `playwright_allowed=True` (Thorough) — permission |
| **Settings toggle** | “Browser-based fetching” — capability (default off) |

Both must be true for Playwright escalation.

### 11.2 Design principles

- **Separate process** — `workers/browser_fetch_worker.py`; never block `LLMWorker`.
- **Hard caps** — 1 Playwright fetch per turn; 2 per host per session.
- **Package optional** — user-installed Chromium first (smaller install).

### 11.3 Cookie banner handling

Deterministic only (v1): click common selectors; 2s timeout; else `failure_reason=cookie_wall`. No LLM-driven clicking.

---

## 12. `general_web` service extension

### 12.1 Service definition (unchanged id)

| Field | Value |
|-------|-------|
| `id` | `general_web` |
| `composer tokens` | `@internet`, `@fetch`, `@recipe` (+ `@[tool:user:…]` source profiles) |
| `pipeline` | Extended `EvidencePipeline` / `pipeline_general_web.py` |
| `strategies` | `ddg_serp_only` (fetch_url_count=0); `ddg_serp_selective_fetch` (fetch_url_count≥1) |

### 12.2 Pipeline stages (trace schema v3)

| Stage | Output |
|-------|--------|
| `discover` | `CandidateUrl[]` |
| `gate` | Relevance-filtered URLs |
| `fetch` | `FetchResult[]` |
| `extract` | `Document[]` |
| `chunk` | `DocumentSection[]` |
| `rank` | Scored sections |
| `polish` | Optional sidecar (v1.1+ P5) |
| `bundle` | `EvidenceBundle` |

---

## 13. Settings & UI

### 13.1 Settings → Knowledge (additive)

| Control | Type | Default |
|---------|------|---------|
| Enable page fetching | toggle | on |
| Browser-based fetching (Playwright) | toggle | off |
| Max fetch size | enum (256K / 512K / 1M) | 512K |
| Respect robots.txt | toggle | on |
| Sidecar section polish | toggle | off |
| Fetch failure notifications | toggle | on |

Retrieval profile (Fast / Balanced / Thorough) already in Settings — document fetch behavior per profile.

### 13.2 Retrieval Inspector

Per URL: `fetch_tier`, `failure_reason`, `extractor_name`, `extractor_version`, `confidence`, `page_count`, `section_count`. Blocker badge on UI source rows.

### 13.3 FAQ — custom composer tools vs connectors

**Q: Can a user create a custom tool like `@game_guides` for walkthroughs from gaming sites?**

Yes — via **Settings → Knowledge → My knowledge** (source profile), not via Custom sources. Preset id `game_guides` becomes composer token `@[tool:user:game-guides]` (ids are normalized to lowercase with hyphens). The user picks it in the `@` picker; the app inserts the full token.

Requires slice **M8** (`general_web` in `ALLOWED_BASE_SERVICES`, `site_bias` on presets). Until then, presets only wire API adapters (`pubmed`, custom `rest_json` sources, etc.) — not HTML scraping.

**Q: Which Connector type should the user pick for scraping (Settings → Knowledge → Custom sources)?**

**None.** Connectors (`rest_json`, `rss_atom`, `graphql`, `mcp`, `sqlite`, `filesystem`, `postgresql`) are for **configured sources**: user-defined endpoints that return structured rows via `get_search_function()` fallback. They are not the web-fetch path.

Scraping specialty websites (Fandom, IGN, etc.) uses the **`general_web` fetch pipeline** (DDG discovery with `site_bias` → HTTP fetch → extractor plugins → `Document` → `EvidenceBundle`). No connector dropdown applies.

| User goal | Settings section | Mechanism |
|-----------|------------------|-----------|
| Game walkthroughs from curated sites | **My knowledge** (source profile) | `base_service: general_web`, `site_bias: ["fandom.com", "ign.com"]` |
| Site exposes a JSON search API | **Custom sources** | `rest_json` + base URL / search path with `{query}` |
| Site exposes RSS/Atom | **Custom sources** | `rss_atom` |
| Generic web, no domain list | Built-in `@[tool:fetch]` or `@[tool:internet]` | Profile + optional pin |

**Q: What is the difference between a source profile and a custom source?**

- **Source profile** (My knowledge): bundles behavior for chat — `@[tool:user:…]`, `base_service`, `site_bias`, `fetch_url_count`. Drives discovery bias and fetch depth. **No connector field.**
- **Custom source** (Custom sources): registers an adapter id (e.g. `gamer_api`) referenced inside a preset’s source list. **Requires a connector** that matches how data is accessed (API, feed, DB).

A preset can combine both: e.g. `@[tool:user:game-guides]` with `site_bias` for scraping **and** a `rest_json` custom source id if a companion API exists. Scraping domains themselves do not need a custom source row.

**Q: Why not add a `html_scrape` or `web_fetch` connector?**

Connectors assume a **declarative, repeatable query contract** (`execute(query)` → list of dicts). Arbitrary HTML pages need discovery, blocker handling, extractor selection, section ranking, and provenance — the shared fetch engine in §5–§9. Duplicating that per connector would fork the pipeline. Source profiles configure *where* to look; extractors configure *how* to read the page.

**Q: What works today vs after MVP?**

| Today (codebase) | After M1–M8 |
|------------------|-------------|
| `@[tool:user:…]` presets for API services only | Presets with `base_service: general_web` + `site_bias` |
| `@internet` = SERP snippets only | `@fetch` / `@recipe` + profile-gated fetch |
| No scrape connector (by design) | Scraping via fetch pipeline; still no scrape connector |

**Implementer note (M8 UI):** My knowledge should expose a **mode** or `base_service` choice — API adapters (current) vs **Web fetch (source profile)** with `site_bias` and `fetch_url_count`. Do not require a Connector selection for the web-fetch mode; that dropdown belongs only on Custom sources.

---

## 14. Observability & audit

### 14.1 RetrievalRecord

- `knowledge_service: general_web`
- `adapter_calls: ("duckduckgo", "fetch_engine")`
- Fetch diag in `relevance_diag.fetch`

### 14.2 Stage traces

```json
{
  "urls_discovered": 5,
  "urls_attempted": 2,
  "urls_success": 1,
  "blockers": {"cloudflare": 1},
  "extractors": [{"name": "RecipeExtractor", "version": "1.0.0", "confidence": 0.98}],
  "sections_emitted": 3
}
```

### 14.3 Fetch provenance trail (required in v1 MVP — M7)

Every fetch turn must produce a **human-readable provenance chain** for debugging, user trust, regression tests, and Inspector Explain. Reuses `PipelineStageTrace` + `RetrievalRecord`; add structured `fetch_provenance` in `relevance_diag`.

**Example provenance (persisted + Inspector Explain tab):**

```
Query: "carbonara recipe"
Composer: @recipe

Discovery:
  provider: duckduckgo
  site_bias: [seriouseats.com, bbcgoodfood.com]
  candidates: 5 URLs (ranked)

Selected for fetch:
  1. https://www.bbcgoodfood.com/recipes/...

Fetch:
  tier: http
  success: true
  bytes: 48231

Extractor:
  name: RecipeExtractor
  version: 1.0.0
  confidence: 0.98
  priority: 90

Output:
  document_sections: 1
  structured_data: recipe (ingredients + steps)
  evidence_objects: 1
```

**Schema (`fetch_provenance.py`):**

```python
@dataclass
class FetchProvenance:
    query: str
    composer_tool: str | None
    site_bias: tuple[str, ...]
    discovery_provider: str
    candidates: tuple[dict, ...]      # url, rank, title
    selected_urls: tuple[str, ...]
    fetch_attempts: tuple[dict, ...]  # url, tier, success, failure_reason
    extractor_name: str | None
    extractor_version: str | None
    extractor_confidence: float | None
    sections_emitted: int
```

**User questions answered:** “Why this URL?” “Why RecipeExtractor?” “Why didn’t it fetch another page?” — without re-running the turn.

### 14.4 Evidence cache

Cache key: `sha256(url + extractor_version + fetch_tier_config)`. TTL: 1h default.

---

## 15. Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `trafilatura` | General article extraction | Yes (M3) |
| `recipe-scrapers` | Structured recipes | M6 |
| `beautifulsoup4` | Already in requirements | Yes |
| `lxml` | Trafilatura / BS4 | Likely yes |
| `playwright` | Browser fallback | Optional extra |

---

## 16. Security & compliance

- All URLs through `validate_url()` and `EgressPolicy`.
- robots.txt simple disallow check when enabled.
- `host_scheduler` per domain; max 2 parallel fetches.
- Settings copy: personal use; user responsible for site ToS.

---

## 17. Testing strategy

### 17.1 Fixtures

```
eval/fixtures/fetch/
    recipe_jsonld.html
    walkthrough_paginated_p1.html
    cloudflare_challenge.html
    js_shell.html
    article_long.html
```

### 17.2 Unit tests

| Module | Tests |
|--------|-------|
| `blockers.py` | Cloudflare, paywall, JS shell |
| `document/types.py` | Schema validation |
| `extractors/*` | `supports()` confidence scoring |
| `section_chunker.py` | Heading splits from `Document` |
| `section_ranker.py` | Top-K, MMR |
| `site_bias.py` | `site:` query augmentation |
| `fetch_provenance.py` | Provenance schema + serialization |
| `pagination.py` | `rel=next` (v1.1+ P1) |

### 17.3 Integration tests

- `general_web` extended pipeline with fixtures → `EvidenceBundle`
- Empty failure → `coverage: none` → worker downgrade
- `@recipe` pin → built-in recipe `site_bias`; `RecipeExtractor` when JSON-LD present
- `@[tool:user:serious-eats]` preset → fetches only `site_bias` domains
- Fast profile → SERP only, no fetch HTTP calls

---

## 18. Implementation slices (detailed)

Slices use **M** = v1 MVP (ship first), **P** = v1.1+ (defer until MVP validated on real usage).

### M1 — Discovery protocol

**Goal:** URL discovery decoupled from fetch.

| Task | Files |
|------|-------|
| `CandidateUrl`, `DiscoveryProvider` | `core/knowledge/discovery/types.py`, `registry.py` |
| `DuckDuckGoDiscovery` | `core/knowledge/discovery/duckduckgo.py` |
| Tests | `tests/test_discovery_provider.py` |

**Exit criteria:** `discover("pasta recipe", max_results=5)` → `list[CandidateUrl]` with URLs only.

---

### M2 — Fetch engine

**Goal:** HTTP fetch with blocker detection.

| Task | Files |
|------|-------|
| `FetchResult`, `BlockerReason` | `core/knowledge/fetch/types.py` |
| Blocker heuristics | `core/knowledge/fetch/blockers.py` |
| Fetch engine | `core/knowledge/fetch/engine.py` |
| Fixtures | `eval/fixtures/fetch/` |
| Tests | `tests/test_fetch_blockers.py` |

**Exit criteria:** Cloudflare fixture → `failure_reason=cloudflare`; clean HTML → `success=True`.

---

### M3 — Document model + extractor plugins (Trafilatura)

**Goal:** HTML → canonical `Document` via plugin registry.

| Task | Files |
|------|-------|
| `Document`, `DocumentSection`, `DocumentMetadata` | `core/knowledge/document/types.py` |
| `Extractor` protocol (`supports`, `extract`, `priority`) | `core/knowledge/extractors/base.py`, `registry.py` |
| `TrafilaturaExtractor` v1.0 | `core/knowledge/extractors/trafilatura_extractor.py` |
| Tests | `tests/test_document_model.py`, `tests/test_trafilatura_extractor.py` |

**Exit criteria:** Article fixture → `Document` with titled sections and extractor metadata.

---

### M4 — Section chunking & ranking

**Goal:** `Document` → ranked `EvidenceObject`s within char budget.

| Task | Files |
|------|-------|
| Section chunker | `core/knowledge/fetch/section_chunker.py` |
| Section ranker | `core/knowledge/fetch/section_ranker.py` |
| Update `bundle_to_prompt_context` | `core/knowledge/ui_adapter.py` |
| Tests | `tests/test_section_ranker.py` |

**Exit criteria:** Long article → 3 ranked sections; prompt ≤ budget without mid-word chop.

---

### M5 — Profile-gated `general_web` integration + `@fetch`

**Goal:** End-to-end wired into existing service.

| Task | Files |
|------|-------|
| Extend `EvidencePipeline` with fetch stages | `core/knowledge/pipeline.py` or `pipeline_general_web.py` |
| `fetch_url_count` on `RetrievalProfileSpec` | `core/knowledge/retrieval_profiles.py` |
| `RetrievalContext` overrides | `core/knowledge/types.py` |
| `@fetch` composer pin | `core/composer_attachments.py`, `registry.py` |
| Worker wiring | `workers/llm_worker.py` |
| Settings copy | `ui/views/settings/sections/knowledge.py` |

**Exit criteria:** Balanced profile + WEB → SERP + fetch top 1 → `EvidenceBundle`. Fast profile → SERP only.

---

### M6 — Recipe extractor + `@recipe`

**Goal:** JSON-LD-driven recipe extraction.

| Task | Files |
|------|-------|
| `RecipeExtractor` v1.0 (`priority=90`) | `core/knowledge/extractors/recipe_extractor.py` |
| Built-in recipe `site_bias` defaults | `core/knowledge/site_bias.py` |
| `@recipe` composer pin | `core/composer_attachments.py` |
| Tests | `tests/test_recipe_extractor.py` |

**Exit criteria:** JSON-LD recipe fixture → `Document.structured_data` with ingredients; `@recipe` scopes DDG to default recipe sites.

---

### M7 — Fetch provenance + Inspector Explain

**Goal:** Every fetch turn explainable.

| Task | Files |
|------|-------|
| `FetchProvenance` schema | `core/knowledge/fetch_provenance.py` |
| Wire into `relevance_diag` + `PipelineStageTrace` | `pipeline.py`, `observability.py` |
| Inspector Explain tab (fetch chain) | retrieval inspector UI |
| Tests | `tests/test_fetch_provenance.py` |

**Exit criteria:** Recipe turn shows full discovery → extractor → output chain in Inspector.

---

### M8 — Source profiles (minimal)

**Goal:** User-defined domain tools (“My Recipes”, “My Gaming”, “My Linux Docs”).

| Task | Files |
|------|-------|
| Add `general_web` to `ALLOWED_BASE_SERVICES` | `core/knowledge/presets.py` |
| Preset fields: `site_bias`, `fetch_url_count` | `presets.py` |
| Add `fetch`, `recipe` to `RESERVED_PRESET_IDS` | `presets.py` |
| Settings UI (label: Source profiles / My knowledge) | `knowledge_presets.py` |
| Preset → `RetrievalContext` wiring | `registry.py`, `llm_worker.py` |
| Manual QA doc | `docs/manual_qa_web_content_fetch.md` |

**Exit criteria:** User creates “My Recipes” → `@[tool:user:serious-eats]` fetches only `site_bias` domains.

---

### v1.1+ deferred slices

#### P1 — Pagination crawl policy

Multi-page content. `pagination.py`; `pagination_allowed` on source profiles.

#### P2 — Additional extractor plugins

`ProceduralExtractor`, `DocumentationExtractor` — register when Trafilatura fails systematically on source-profile use-cases.

#### P3 — Playwright worker

Opt-in Cloudflare/JS fallback. `workers/browser_fetch_worker.py`.+

#### P4 — Rich source profiles

`preferred_extractors`, `section_ranking_weights`.

#### P5 — Optional sidecar section polish

Extend `source_digest` for ranked sections.

#### P6 — Additional discovery providers

RSS, bookmarks, extended site lists.

---

## 19. File map (target state)

**v1 MVP:**

```
core/knowledge/
  discovery/
    types.py
    registry.py
    duckduckgo.py
  fetch/
    types.py
    engine.py
    blockers.py
    section_chunker.py
    section_ranker.py
  document/
    types.py
    normalize.py
  extractors/
    base.py
    registry.py
    trafilatura_extractor.py
    recipe_extractor.py
    bs4_patterns.py
  site_bias.py
  fetch_provenance.py
  pipeline.py                         # extended

core/composer_attachments.py          # @internet, @fetch, @recipe pins
core/knowledge/registry.py
core/knowledge/presets.py             # general_web source profiles
core/knowledge/types.py
core/knowledge/ui_adapter.py
core/knowledge/retrieval_profiles.py

workers/llm_worker.py
ui/views/settings/sections/knowledge.py
ui/views/settings/sections/knowledge_presets.py
```

**v1.1+ additions:** `pagination.py`, `procedural_extractor.py`, `documentation_extractor.py`, `browser_fetch_worker.py`

**Not in target state:** `services/web_content.py`, `pipeline_web_content.py`, `discovery_bias.py` (replaced by `site_bias.py`).

---

## 20. Open questions

| # | Question | Recommendation |
|---|----------|----------------|
| 1 | Default for plain WEB (no `@` tool) on Balanced profile | **fetch_url_count=1** on Balanced; 0 on Fast |
| 2 | Playwright packaging | User-installed Chromium first |
| 3 | recipe-scrapers input | Pass pre-fetched HTML from `knowledge_get` |
| 4 | Recipe evidence shape | One `EvidenceObject` with `structured_data` in metadata |
| 5 | robots.txt parser | Simple disallow (v1) |
| 6 | Hybrid route | Fetch WEB leg only; RAG unchanged |
| 7 | Sidecar polish vs `source_digest` reuse | **Defer to P5**; reuse `source_digest` first |
| 8 | When to split `web_content` service | When `general_web` if-branches become unmaintainable |
| 9 | Built-in tools beyond three | **No** — only `@internet`, `@fetch`, `@recipe` |
| 10 | Preset id vs built-in `@recipe` | `RESERVED_PRESET_IDS`; user uses `my-recipes` → `@[tool:user:my-recipes]` |
| 11 | Source profile vs site_bias naming | **Source profile** in UI; `site_bias` is v1 MVP field |

---

## 21. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Playwright install friction | Optional extra; two-gate permission |
| Site layout changes | `supports()` confidence drops → Trafilatura fallback; fixtures in CI |
| Legal/ToS | robots.txt; rate limits; user acknowledgment |
| Latency on `@internet` | Fast profile default; fetch_url_count=0 |
| Small LLM overflow | Section ranker + char caps |
| Anti-bot arms race | Structured failure; no garbage in prompt |
| Extractor regression | Version metadata in RetrievalRecord / Replay |
| Over-engineering v1 | Strict MVP band M1–M8; defer P1–P6 until real usage |
| Ambiguous built-in naming | Three built-ins only; source profiles for all domains |

---

## 22. Relationship to other docs

| Document | Relationship |
|----------|--------------|
| [External Knowledge Platform Plan](./external_knowledge_platform_plan.md) | Implements §7 “Selective fetch” |
| [HTTP Resilience Plan](./knowledge_adapter_http_resilience_plan.md) | Fetch reuses `knowledge_get` |
| [ADR 003](./adr/003-evidence-convergence.md) | `Document` → `EvidenceObject` → `EvidenceBundle` |
| [ADR 001](./adr/001-skills-orthogonal-to-routing.md) | No sidecar routing; polish only post-rank |
| [Sidecar tasks](./sidecar_tasks.md) | `source_digest` pattern for optional polish |
| [Cognitive Router](./cognitive_router.md) | WEB lane only |
| [Knowledge Platform Evolution Review](./knowledge_platform_evolution_review.md) | Profiles = orchestration; defer service split |

---

## 23. Revision history

| Date | Change |
|------|--------|
| 2026-07-14 | v1 — Initial draft |
| 2026-07-14 | **v2 — Revised after external architecture review:** defer `web_content` service; add `Document` IR; capability-based extractors; discovery/extraction separation; profile-gated fetch; deterministic discovery bias only; extractor versioning; sidecar scoped to optional post-rank polish; reordered slices |
| 2026-07-14 | **v3 — Composer tools & presets round** |
| 2026-07-14 | **v4 — Final check-up:** drop `@howto`; three built-ins only; collapse discovery ontology to `site_bias`; extractor `priority` + plugin loop; v1 MVP M1–M8 vs v1.1+ P1–P6; source profiles; fetch provenance trail (M7); Trafilatura + Recipe only in MVP |
| 2026-07-14 | **v4.1 — §13.3 FAQ:** custom composer tools vs Custom sources connectors; scraping uses source profiles, not connector types |

---

## Appendix A — Example turn flows

### A.1 `@recipe` carbonara (Balanced profile)

1. User: `@[tool:recipe] authentic carbonara recipe`
2. `resolve_turn_knowledge_service` → `general_web`
3. `RetrievalContext`: `fetch_url_count=1`, built-in recipe `site_bias`
4. `DuckDuckGoDiscovery` with site bias → 5 URLs → relevance gate → top 1
5. Fetch → HTTP OK → `RecipeExtractor.supports()` → 0.98 (JSON-LD Recipe)
6. `Document` with `structured_data` → 1 compact `EvidenceObject`
7. Provenance recorded (M7): discovery → bbcgoodfood.com → RecipeExtractor 0.98
8. Bundle `coverage: excellent`, `fetch_status: full_extract`

### A.2 Balanced profile, no composer pin

1. Router → WEB
2. Balanced profile → `fetch_url_count=1`
3. DDG (no site filter) → fetch → `supports()` picks extractor
4. If fetch fails → try next URL or SERP fallback with `snippet_fallback` warning

### A.3 Fast profile + `@internet`

1. User: `@[tool:internet] weather in Copenhagen`
2. Fast profile → `fetch_url_count=0`
3. DDG SERP only → `snippet_only` evidence (unchanged behavior)

### A.4 Cloudflare blocked (Thorough + Playwright enabled)

1. Thorough profile → `fetch_url_count=3`, `playwright_allowed=True`
2. Settings → browser fetching ON
3. URL 1 → HTTP → `cloudflare` → Playwright → extract OK
4. Bundle `warnings: ("partial_fetch:1_cloudflare_escalated",)`

### A.5 User preset `@[tool:user:serious-eats]`

1. User: `@[tool:user:serious-eats] weeknight pasta recipe`
2. `parse_user_preset_tool` → preset `serious-eats`
3. `resolve_turn_knowledge_service` → `general_web` (preset `base_service`)
4. `RetrievalContext` from source profile: `site_bias=["seriouseats.com"]`, `fetch_url_count=2`
5. DDG query scoped to site → fetch → `RecipeExtractor.supports()` on JSON-LD page
6. Provenance shows `composer: user:serious-eats`, selected URL, extractor confidence
7. Bundle cites Serious Eats URL only

### A.6 `@[tool:user:ikea-diy]` IKEA assembly (v1 MVP — Trafilatura)

1. User: `@[tool:user:ikea-diy] how to assemble BILLY bookcase`
2. Source profile `site_bias=["ikea.com"]`
3. Fetch IKEA instructions page → no JSON-LD Recipe → `TrafilaturaExtractor` wins
4. Section ranker selects step-heavy sections
5. (v1.1+ P2: add `ProceduralExtractor` if Trafilatura quality insufficient)

---

## Appendix B — Proposed ADRs (draft)

### ADR 004 — Fetch failures never become evidence

**Decision:** Blocker pages, challenge HTML, and empty extracts must not be placed in `EvidenceObject.excerpt` or `full_text`. Failed fetches are `FetchResult` diagnostics, bundle `warnings`, and `coverage: none` — triggering empty-source downgrade when all fetches fail.

### ADR 005 — Discovery ≠ evidence; Document before EvidenceObject

**Decision:** URLs from discovery are not evidence. Extraction must produce a canonical `Document` before chunking. Only ranked sections become `EvidenceObject`s inside an `EvidenceBundle`.

**Consequences:** New discovery sources and extractors can be added without changing the prompt contract.

---

*End of document.*
