# Memory system (RAG + long-term atomic memory)

**Audience:** Contributors and advanced users.  
**Extracted from:** [archived README](../archive/readme-pre-launch-rewrite.md) (pre–launch rewrite).  
**User-facing:** In-app **Memory Manager** help · [How to use — Memory Manager](../user/how-to-use.md#memory-manager)  
**QA:** [memory_manual_qa.md](../memory_manual_qa.md)

For routing that selects memory vs Library vs chat, see [cognitive_router.md](../cognitive_router.md).

---

## 🏗️ Deep Dive: Architecture & Features

### 🧠 Dual Memory System (RAG + Long-Term Atomic Memory)

Qube uses two complementary memory layers:

#### 1. Document RAG Memory

- Built on **LanceDB vector storage**.
    
- Ingests PDFs, EPUBs, TXT, and Markdown files.
    
- Retrieves semantic chunks for grounding responses.
    
- Injects retrieved context directly into LLM prompts using sequential numeric citations (e.g., `[1]`).
    

#### 2. Long-Term Atomic Memory (v6)

- Extracts durable “facts” from conversations asynchronously.

- Runs in a background **QThread Enrichment Worker** that yields to the main LLM to prevent local server deadlocks.

- Stores structured atomic memory in LanceDB using a dedicated `qube_memory::%` namespace.

**Key properties (Phase A → C hardening):**

- **Typed extraction schema** — every memory carries `subject` (user / third_party / system), `source_role` (user / assistant / derived), `durability`, `category`, `content`, `provenance_quote`, and `confidence`. The LLM is given explicit NEGATIVE examples for the classic regressions ("Tell me about Alice" → `[]`, "I don't have internet access" → `[]`).

- **Role-aware preprocessing** — assistant refusal / limitation messages are matched against a regex blacklist and replaced with `[failure message omitted]` *before* the extraction prompt is built, so a one-off "I can't access the internet" turn cannot become a permanent "the agent has no internet" memory.

- **Tool-aware turn fences (T3.3)** — the LLM worker now tells the enrichment worker when a turn should not be mined at all. Stream-repetition guard trips, web-search failure sentinels on WEB/INTERNET/HYBRID routes, pipeline errors, and assistant-failure final text all set `enrichment_mode = "skip"`. Explicit-remember turns ("please remember that…") set `enrichment_mode = "explicit_only"` so the user-requested fact is still seeded while the extractor LLM call is skipped on the acknowledgement. Cadence-driven maintenance (usage drain, decay sweep) keeps running independently.

- **Episodic session summaries (T3.2)** — alongside the atomic-fact pipeline, the enrichment worker now writes a single-paragraph `episode` row per active session. After every extraction flush, `_maybe_summarise_session` bumps a per-session turn counter and fires `_summarise_session_now` when it hits the cadence (8 turns) or when the session has been idle for more than 15 minutes. The summariser LLM returns `SUMMARY: <paragraph>` + `TOPICS: <tags>` (or `SUMMARY: SKIP` for trivial chitchat), the result is validated against the usual thin-content / assistant-failure / negative-list filters, capped at 800 chars, and written in-place to `qube_memory::episode::<session_id>` — replacing any prior episode for that session. Narrative recap queries ("what have we been working on?", "where did we leave off?", "recap my session", "summarize this conversation") are detected up-front by `detect_narrative_intent`, routed to `MEMORY` with `prefer_episode=True`, and the retrieval scorer boosts `category=="episode"` rows by `+0.35` so they outrank atomic facts; the returned sources are inline-labelled `[EPISODE]` and a narrative-recap system-prompt suffix tells the LLM to prefer them. The reflection worker skips episode rows (they regenerate on cadence and are not the kind of durable user fact the judge rates), and the Memory Manager surfaces episodes under a dedicated **Episode** category with a topics line so you can inspect / flag / delete them like any other memory.

- **Structured preference / knowledge tiers (T3.4)** — every atomic fact is now stored under a structural tier derived from its validated payload: `preference` (user-subject user_stated/user_confirmed facts), `knowledge` (third-party subject, document-derived, or explicit-remember), `episode` (T3.2 session summaries), or `context` (legacy fallback). The tier lives in the LanceDB `source` column as `qube_memory::<tier>::<category>` so retrieval is tier-scoped with a cheap `LIKE` filter — no new columns, no migration needed on fresh installs. Plain chat turns now run a MemGPT-style "core memory" lookup that queries preferences + context only (`top_k=3`); recall / hybrid turns additionally surface the knowledge tier; narrative recap turns surface every tier with episodes on top. The Memory Manager grows a two-level **Tier × Category** filter, each row gets a colour-coded `PREF` / `KNOW` / `EP` / `CTX` pill next to the category badge, and the reflection worker learned two new structural labels — `tier_mismatch` (preference-tier row whose subject is not the user) and `orphan_knowledge` (knowledge-tier row that has lost every piece of evidence it was stored for) — both raised deterministically before the LLM judge runs.

- **RAG relevance gate + empty-retrieval downgrade (T4.1)** — RAG vector hits now pass through a hard semantic-relevance floor (`MIN_RAG_SEMANTIC_SCORE=0.30` on L2-normalised Nomic v1.5 embeddings, mirroring the memory tool's gate). Below-floor chunks are dropped before ranking, and if the vector channel produced candidates but the gate killed all of them, the FTS fallback is also suppressed — lexical matches without semantic corroboration are almost always brittle (FTS matching the word "blue" in a Blue Jay migration study when you asked about Rayleigh scattering). If every retrieval channel comes back empty on a `MEMORY` / `RAG` / `HYBRID` turn, the route is downgraded to `NONE` after telemetry is logged, so the LLM answers the general-knowledge question from its own parameters on the base system prompt instead of being steered by a citation-discipline suffix into a "I couldn't find anything in my sources" reply. This closes the regression where "Why is the sky blue?" against a single-document library returned a bare `[1]` pointing to the unrelated document.

- **WEB-route empty-source downgrade + proactive tool-disabled veto** — the cognitive router internally promotes `route` to `"web"` as soon as `_score_web_intent` clears its threshold on keywords like `weather` / `today`, and that value previously flowed straight through to the prompt build. When the user had the internet tool turned off (or when `search_internet` returned the "Internet search failed" sentinel and the guard cleared `web_results`), the WEB system-prompt branch still asserted *"You have just been provided with real-time, live web search results. Cite the web sources inline using a plain [W] token…"* against an empty source block — a small LLM duly hallucinated both an answer and a `[W]` citation, and the UI correctly warned `Citation id 'W' not found on this message (0 sources)`. Two complementary guards now close this: (a) a **proactive veto** that reverts `execution_route` from `WEB` to `NONE` before tool execution when the router picked WEB but none of `force_web` / manual-trigger / auto-trigger fired *and* `mcp_internet_enabled` is False (stamping `decision["web_vetoed_tool_disabled"]=True` and emitting a distinctive INFO log), and (b) an **extended T4.1 empty-source downgrade** that now includes `WEB` / `INTERNET` in its route tuple, so even when the tool is enabled but the search returns nothing usable the route still flips to `NONE` before the prompt build — landing the turn on the base *"You are Qube, be concise"* system prompt with no `[W]` citation instruction. The WEB-downgrade path also marks `skip_enrichment("web_route_no_sources")` so the thin "I can't check live data right now" reply is not mined for user facts by the enrichment worker, mirroring the existing `web_tool_failure` sentinel behaviour.

- **Server-side validation** — drops candidates that are `subject=system`, `source_role=assistant` (without an explicit `remember that…` from the user), bare third-party stubs, non-`long_term`, thin (`< 3 words` / single proper-noun / all stop-words), match an assistant-failure pattern, or are missing a `provenance_quote`.

- **Per-turn provenance** — each memory records its `source_session_id`, `source_message_ids`, `origin` (user_stated / user_confirmed / document_derived / system_derived), and `links_to_document_ids` for the RAG chunks that were in context when it was formed. On retrieval, a thin memory **auto-expands to its originating document chunk** so "Who is Alice?" answers from the actual document, not the bare name.

- **Embedding-based clustering** — replaces the old keyword-length cluster key with a nearest-neighbor join (`L2 < 0.30`) on the memory table, so related-but-distinct facts ("I prefer dark roast" / "my favorite is arabica") share a cluster and can trigger the contradiction judge.

- **Two-stage contradiction judge** — Jaccard fast-path detects literal duplicates; otherwise a short LLM micro-call labels the pair `duplicate` (reinforce strength), `contradiction` (replace old with new), or `complement` (insert alongside).

- **Persistent negative-pattern list** — every memory you delete in the Memory Manager is appended (content + vector) to `~/.qube/memory_negatives.json` so the next extraction pass rejects any candidate within `L2 < 0.20` of a deleted memory. The same memory cannot be recreated by a similar conversation tomorrow.

- **Usage-driven decay** — payloads carry `times_retrieved`, `times_cited_positively`, `last_used_at`. A 24 h sweep recomputes `usefulness` and `decay`, purges rows below `decay < 0.15`, and the retrieval scorer re-weights to include the decay term so memories that earn their keep float to the top.

- **Self-reflection worker** — every 6 hours, batches 10 least-recently-reflected memories and asks the **CPU sidecar** (Qwen3 1.7B) to label each as `durable_user_fact` / `third_party_stub` / `system_claim` / `transient` / `unclear`. Anything other than `durable_user_fact` is marked `flagged_for_review` and surfaced in the Memory Manager's Flagged section. **Never auto-deletes** — final say belongs to you.

- **Memory v7 hardening** — hybrid vector + FTS retrieval (0.7 / 0.3 rank fusion) with action-boundary filtering (`expires_at`, `safe_to_act_after`, `authority` in JSON payloads); CHAT-route core-memory relevance gate suppresses weak preference/context injection; pre-window **salvage** re-extracts facts from turns dropped by session history windowing; query-fingerprint tracking feeds deferred **promotion** of context/knowledge rows to preference (optional in Settings); MMR diversity + tier-specific temporal decay on recall routes; calendar daily episode rollup (`qube_memory::episode::YYYYMMDD`); Memory Manager **Promotion candidates** section + Markdown export of visible rows.

#### 3. Memory v7.1 reliability (cross-day trust + explainability)

v7.1 strengthens the existing LanceDB-centric stack without markdown dream files or auto-promotion by default. All new fields live **inside the JSON `text` payload**—no LanceDB schema migration.

**Exposure telemetry (feeds promotion + consolidation scoring):**

- **`retrieval_days`** — FIFO list of ISO calendar dates (`YYYY-MM-DD`, cap 16) deduped per retrieve event; analog to multi-day recall signals.
- **`retrieval_score_sum` / `retrieval_score_count`** — running average of final `memory_search` scores, recorded via `MemoryUsageRecorder` on each kept hit.
- **`times_salvage_considered` / `times_episode_overlap`** — bounded weak-exposure counters when salvage or daily/session episode rollups touch related atomic rows.

**Background workers (all QThread, never on the UI thread):**

| Worker | Cadence | Purpose |
| :--- | :--- | :--- |
| `MemoryConsolidationWorker` | 6 h | Deterministic cross-day staging: writes `consolidation_score`, `consolidation_hints`, `consolidation_staged_at` on context/knowledge rows. **Never auto-promotes or auto-deletes.** Default **on** in Settings. |
| `MemoryPromotionWorker` | 6 h | Optional context/knowledge → preference tier upgrade when gates pass. **Off by default.** Pre-promote hardening: live re-read, near-duplicate block (`L2 < 0.22` vs existing preference rows), reflection veto on `flagged_for_review`. |

**Promotion scoring (Settings → preset: Conservative / Standard / Aggressive):**

- Log-scaled **frequency** from composite exposure (retrievals + capped citations + salvage touches).
- **Relevance** blends citation rate with average retrieval score when scores exist.
- **Diversity** uses `max(unique_query_count, len(retrieval_days))`.
- **Consolidation** from multi-day `retrieval_days` span (not binary first-seen age alone).
- `passes_promotion_gates_with_reason()` returns `(ok, reason, components)` for Memory Manager tooltips and logs.

**Retrieval polish:**

- MMR normalizes candidate scores to `[0, 1]` before diversity rerank (near-duplicate skip at similarity ≥ 0.85 unchanged).
- When FTS exposes `_score` / rank metadata, hybrid merge uses `bm25_rank_to_score` + weighted vector/text fusion; otherwise canonical rank fusion is unchanged.

**Settings toggles (Performance section):**

- **Enable Memory Enrichment & Reflection** — master switch for extraction + reflection (existing).
- **Enable Memory Promotion (v7)** — opt-in promotion worker (default off).
- **Promotion preset** — Conservative / Standard / Aggressive gate thresholds (only when promotion is enabled).
- **Enable Memory Consolidation (v7.1)** — cross-day staging worker (default on).

---

### 🗂️ Memory Manager

A dedicated nav screen (between Library and Telemetry) that makes the long-term memory store a first-class, user-editable surface.

- **Promotion candidates** — rows that pass v7.1 promotion gates (when promotion is enabled in Settings); hover for a weighted signal breakdown.

- **Almost promoted** — high-scoring context/knowledge rows that fail one gate, or rows staged by the consolidation worker (`consolidation_staged_at`); tooltips show `passes_promotion_gates_with_reason` gate failure + signal components (capped at 12 rows).

- **Recurring themes** — deterministic rollup card (categories, episode topics, frequent query fingerprints) over currently visible rows—no LLM.

- **Consolidation badge (`STAGED`)** — on row cards when `consolidation_hints` is non-empty (e.g. `multi_day_retrieval`, `high_citation`, `episode_overlap`).

- **Top "Flagged for review" section** shows entries the self-reflection worker has surfaced as suspect, so you can confirm or delete them in one pass.

- **Tier × category filters** — structural tier (preference / knowledge / episode / context) plus category dropdown; each row shows a colour-coded `PREF` / `KNOW` / `EP` / `CTX` pill.

- **Category-grouped sections** for everything else, with subject, origin, confidence, decay, and usage counters visible at a glance.

- **Per-row actions:** Edit content (PrestigeDialog input), Flag / Unflag for review, Delete (PrestigeDialog confirm). Bulk **Delete all visible** and **Export visible** (Markdown under `~/.qube/exports/`) for cleanup passes.

- **Filters:** SelectorButton tier + category dropdowns, **Flagged only** toggle, free-text search across memory content.

- **Negative-list integration:** every delete also records the entry into `~/.qube/memory_negatives.json`, so the enrichment pipeline cannot recreate it from a similar conversation later.

- **Off-thread DB work:** all LanceDB read / delete / re-add goes through a `MemoryManagerWorker` QThread; the UI stays fluid even on large stores.

**QA:** See [memory_manual_qa.md](../memory_manual_qa.md) for the full manual test plan (v6–v7.1). Run `pytest tests/test_memory_qa_smoke.py` before release for settings/export/negative-list smoke.
