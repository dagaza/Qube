# Manual QA — Phase 6 Slices 2–4

**Purpose:** In-app validation of internal corpus (`@library`), entity resolution, and the session research map after enabling Phase 6 knowledge features.

**Related:** [External knowledge platform plan](./external_knowledge_platform_plan.md) (Phase 6), [Retrieval eval README](../eval/retrieval_corpus/README.md), [Manual QA Slice 5a Finance](./manual_qa_phase6_slice5_finance.md), [Manual QA Slice 6 Discipline routing](./manual_qa_phase6_slice6_discipline_routing.md)

---

## Prerequisites

### Settings (Settings → Knowledge)

| Toggle | Required for |
|--------|----------------|
| **External knowledge pipeline (v2)** | All slices — must be **ON** |
| **Internal corpus (@library evidence service)** | Slice 2 — **ON** for positive tests |
| **Research map (session knowledge graph)** | Slice 4 — **ON** for positive tests |
| **Deep research (@research)** | Slice 3D only — **ON** if testing merge dedupe |

Also ensure **internet / web retrieval** is enabled for `@evidence` / `@science` and `@research` turns.

**Terminology:** `@evidence` is a composer alias for the **Scientific literature** knowledge service (`scientific_evidence`). It retrieves scholarly sources across disciplines — not only biomedicine. The platform **Evidence** model (`EvidenceBundle`) is domain-agnostic and applies to all knowledge services including `@finance`.

### Library index (Slice 2)

1. Copy a fixture into your Library folder, e.g. `eval/fixtures/library/eval_kubernetes_notes.md`.
2. Run **Reindex** so LanceDB has searchable chunks.

Suggested fixtures:

| File | Good for |
|------|----------|
| `eval_kubernetes_notes.md` | Ingress / NGINX / cert-manager (QA-2A) |
| `eval_research_physics_paper.md` | Rayleigh scattering (optional extra library turn) |
| `eval_quarterly_revenue_report.md` | Empty-hit negative test (QA-2B) if not indexed |

### Entity resolution (Slice 3)

Enabled by default (`qube.knowledge.entity_resolution_enabled`). No UI toggle. Leave on unless testing rollback.

Optional RxNorm lookup (`qube.knowledge.rxnorm_entity_lookup_enabled`) is **off** by default; QA-3C notes expected behavior for both states.

---

## Slice 2 — Internal corpus (`@library`)

Routes `@library` through the evidence pipeline over your LanceDB index (`knowledge_service: internal_corpus`). Implicit RAG on non-`@library` turns is unchanged.

### QA-2A — Happy path (corpus hit)

**Prompt:**

```text
@library How is ingress configured on our staging cluster?
```

**Pass if:**

- Answer reflects indexed content (NGINX Ingress Controller v1.11, cert-manager, `/api` → `api-gateway`, etc.).
- **Sources** dialog shows library provenance (filename such as `eval_kubernetes_notes.md`).
- Transparency / “why” summary reflects internal corpus retrieval — not general web search.
- **No duplicate injection:** the answer does not appear to combine the same library chunks twice (evidence bundle + parallel RAG dump).

**Optional (audit):** Retrieval trace includes `knowledge_service: internal_corpus` and `adapter_calls` containing `lancedb_library`.

---

### QA-2B — Weak / empty hit

**Prompt:**

```text
@library What is the quarterly revenue forecast for Q4?
```

Use a topic your indexed library does not cover (or index only `eval_kubernetes_notes.md`).

**Pass if:**

- Empty or low-coverage bundle is handled cleanly — no fabricated library citations.
- App remains stable; response tone reflects missing sources appropriately.

---

### QA-2C — Negative control (internal corpus off)

1. Turn **Internal corpus** **OFF** (keep External knowledge v2 **ON**).
2. Repeat QA-2A prompt.

**Pass if:**

- `@library` uses **legacy RAG** routing (not the evidence pipeline).
- Sources / trace behavior differs from QA-2A (no `internal_corpus` service).

---

## Slice 3 — Entity resolution

Post-retrieval enrichment attaches canonical `entity_ids` to sources. Transparency shows **`Entities detected:`** when entities are found.

Use **Scientific literature (`@evidence`)** (requires network). Open **Sources** on each answer and inspect the transparency / “why” block.

### QA-3A — Drug class + condition

**Prompt:**

```text
@evidence What is the evidence for ACE inhibitors in heart failure?
```

**Pass if:**

- Transparency includes **`Entities detected:`** with labels such as **ace inhibitors (drug class)** and **heart failure (condition)**.
- Source rows show enriched metadata where available (DOI, authors, scores).

---

### QA-3B — Trial acronym

**Prompt:**

```text
@evidence Summarize key outcomes from the EMPEROR-Reduced trial for heart failure.
```

**Pass if:**

- **`Entities detected:`** includes **emperor reduced (trial)** (or equivalent slug label).
- Sources are plausibly related to the trial (not random unrelated hits only).

**Sign-off (2026-06-29):** **Good enough pass.** PubMed is engaged via keyword planning (`scientific_keyword_query`: `EMPEROR-Reduced heart failure`); trial grounding re-ranks away from real-world SGLT2 class papers. Top hits may still be EMPEROR-program sub-studies or secondary analyses rather than the primary NEJM outcomes paper — acceptable for Phase 6; improve via deeper fetch / RetrievalPlan (Stage 2). Example session: `d3ef82ce-aad2-404c-a43d-3075f9ab9765`.

---

### QA-3C — Named drug

**Prompt:**

```text
@evidence What does the literature say about dapagliflozin in HFrEF?
```

**Pass if:**

- Entities include **dapagliflozin (drug)** and **hfref (condition)**.
- With RxNorm **off** (default): no requirement for `rxnorm`-kind ids.
- With RxNorm **on**: optional authority ids may appear in transparency payload.

**Sign-off (2026-06-29):** **Good enough pass.** Keyword planning (`dapagliflozin HFrEF`) restores PubMed hits; answers cite dapagliflozin-specific literature. OpenAlex prescribing-barrier sources may still appear alongside PubMed — acceptable for Phase 6. Example session: `d3ef82ce-aad2-404c-a43d-3075f9ab9765` (same session as QA-3B).

---

### QA-3D — Deep research merge (entity-aware dedupe)

**Prompt:**

```text
@research SGLT2 inhibitors in heart failure: trials and meta-analyses
```

Wait for async deep-research completion; open **Sources** on the finished report.

**Pass if:**

- Report completes with a merged bibliography.
- Sources list does not obviously duplicate the same trial/DOI (e.g. two near-identical EMPEROR-Reduced entries).
- Transparency still shows entity labels when biomedical terms appear in retrieved sources.

---

### QA-3E — Non-medical scientific literature (multi-disciplinary parity)

**Prompt:**

```text
@evidence transformer attention mechanism neural machine translation
```

**Pass if:**

- Retrieval trace shows `knowledge_service: scientific_evidence` with `scientific_adapters_selected` containing `openalex` and/or `arxiv` (PubMed may be absent for non-medical queries).
- Sources are plausibly related to ML/NLP literature — not random biomedical hits only.
- **`Entities detected:`** is absent or limited to bibliographic ids (no spurious drug/trial entities on a CS query).

---

## Slice 4 — Research map (knowledge graph)

Builds a **session-local** graph of queries, sources, and entities across evidence turns. View from **Sources → Research map** when the feature flag is on.

**Use one chat session** for QA-4A and QA-4B so the graph accumulates.

### QA-4A — Graph after two evidence turns

**Turn 1:**

```text
@evidence ACE inhibitors in heart failure
```

**Turn 2:**

```text
@evidence SGLT2 inhibitors and hospitalization in heart failure
```

**Pass if:**

- After turn 2, **Sources** on an evidence answer shows a **Research map** button.
- Research map lists **nodes**: queries, sources, entities (e.g. heart failure, drug classes).
- **Edges** link queries to sources and sources to entities (`mentions`, `about`, or similar).

---

### QA-4B — Graph includes `@library` turn

**Turn 3 (same session):**

```text
@library What ingress controller version do we use?
```

**Pass if:**

- Research map includes the library turn (additional query + source nodes).
- Graph remains scoped to **this session** (new session → empty or fresh graph).

---

### QA-4C — Negative control (research map off)

1. Start a **new session**.
2. Turn **Research map** **OFF**.
3. Run one `@evidence` turn.

**Pass if:**

- **Sources** dialog has **no** Research map button.
- Sources and transparency otherwise behave normally.

---

## Quick reference

| ID | Prompt | Slice | Key signal | Sign-off |
|----|--------|-------|------------|----------|
| QA-2A | `@library How is ingress configured on our staging cluster?` | 2 | Library hit, internal corpus, no double injection | **Pass** |
| QA-2B | `@library What is the quarterly revenue forecast for Q4?` | 2 | Graceful empty / weak coverage | **Pass** |
| QA-2C | QA-2A with internal corpus **off** | 2 | Legacy RAG fallback | **Pass** |
| QA-3A | `@evidence … ACE inhibitors … heart failure` | 3 | `Entities detected:` drug class + condition | **Pass** |
| QA-3B | `@evidence … EMPEROR-Reduced trial …` | 3 | Trial entity in transparency | **Good enough** |
| QA-3C | `@evidence … dapagliflozin … HFrEF` | 3 | Named drug + condition entities | **Good enough** |
| QA-3D | `@research SGLT2 inhibitors … heart failure` | 3 | Merged bib; fewer duplicate trials | **Pass** |
| QA-3E | `@evidence transformer attention …` | 3 | OpenAlex/arXiv; no spurious biomedical entities | Optional |
| QA-4A | Two `@evidence` HF prompts (same session) | 4 | Research map button; nodes + edges | **Pass** (retrieval; confirm UI in Sources) |
| QA-4B | `@library … ingress controller version` (same session) | 4 | Graph includes library turn | **Pass** (backend; confirm UI in Sources) |
| QA-4C | `@evidence` with research map **off** | 4 | No Research map button | **Pass** |

---

## Recording results

For each case, capture:

1. **Session ID** (for audit log correlation, if enabled)
2. **Pass / fail** against criteria above
3. Screenshot or copy of Sources transparency (`why_summary`, `Entities detected` line)
4. For Slice 4: whether Research map showed expected nodes after multi-turn session

### Phase 6 sign-off (2026-06-29)

Manual QA for Slices 2–4 completed in-app with External knowledge v2, Internal corpus, and Research map enabled where required. Log-backed validation used `~/.qube/logs/web_search.log` (`retrieval_trace`, `relevance_diag`).

| Area | Verdict | Notes |
|------|---------|-------|
| Slice 2 (2A–2C) | **Pass** | `internal_corpus` / `lancedb_library`; 2C legacy RAG confirmed |
| Slice 3 (3A, 3D) | **Pass** | Entity + deep-research paths validated |
| Slice 3 (3B, 3C) | **Good enough** | Stage 1 scientific query planner + trial grounding patch; see case notes above |
| Slice 4 (4A–4C) | **Pass** | Retrieval traces confirmed; UI (Research map button/nodes) verified in session |

**Retrieval fixes applied during QA cycle (not re-test prerequisites):**

- **Scientific query planner** — conversational `@evidence` prompts → PubMed keyword queries (`scientific_keyword_query` in trace).
- **Trial grounding** — when a trial entity is in the query, ranking boosts RCT/trial-titled PubMed rows (`scientific_trial_signals` in trace).

**Cache note:** After planner/ranking changes, use a new session or wait for evidence-cache TTL (~1 hour) before re-testing the same prompt.

**Follow-up (Stage 2, not blocking sign-off):** General `RetrievalPlan` layer; deeper PubMed fetch for trial-acronym queries to surface primary outcome papers in top 3.

**Suggested sign-off:** All **QA-2A**, **QA-3A**, **QA-3B** (good enough), **QA-3C** (good enough), **QA-4A** accepted for Phase 6; negative controls **QA-2C** and **QA-4C** pass.

---

## Automated regression (optional companion)

These manual cases complement — do not replace — automated checks:

```bash
python3 -m unittest tests.test_internal_corpus tests.test_entity_resolution tests.test_knowledge_graph tests.test_deep_research tests.test_scientific_query_planner tests.test_trial_grounding -q
python3 tools/evaluate_retrieval.py --live --service scientific_evidence --min-pass 5
python3 tools/evaluate_retrieval.py --live --service trusted_knowledge --min-pass 4
```
