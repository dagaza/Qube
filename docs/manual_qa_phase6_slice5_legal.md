# Manual QA — Phase 6 Slice 5b (Legal)

**Purpose:** In-app validation of `@legal` / CourtListener retrieval, legal disclaimers, and preferred-source settings.

**Related:** [External knowledge platform plan](./external_knowledge_platform_plan.md) (Phase 6 Slice 5), [Manual QA Slice 5a Finance](./manual_qa_phase6_slice5_finance.md), [Retrieval eval README](../eval/retrieval_corpus/README.md)

---

## Prerequisites

### Settings (Settings → Knowledge)

| Toggle | Required for |
|--------|----------------|
| **External knowledge pipeline (v2)** | All cases — must be **ON** |
| **Preferred sources → Legal → CourtListener** | QA-5L-A–5L-D — **checked** (default) |

Ensure **internet / web retrieval** is enabled.

**Optional:** Set `QUBE_COURTLISTENER_API_TOKEN` for higher CourtListener rate limits (not required for basic search).

### Automated baseline

```bash
python3 tools/evaluate_retrieval.py --live --service legal_knowledge --min-pass 3
python3 -m unittest tests.test_legal_knowledge tests.test_courtlistener_adapter -q
```

**Scientific regression:**

```bash
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py --live --service scientific_evidence --min-pass 5
```

---

## Slice 5b — Legal (`@legal`)

Routes `@legal` through `legal_knowledge` → CourtListener adapter. Bundle includes `not_legal_advice` warning.

### QA-5L-A — Landmark case (Miranda)

**Prompt:**

```text
@legal What did the Supreme Court hold in Miranda v Arizona about police interrogation?
```

**Pass if:**

- Answer references Miranda warnings / rights during custodial interrogation.
- **Sources** show CourtListener case law (`courtlistener` adapter).
- **`not_legal_advice`** appears in bundle warnings / transparency.

**Optional (audit):** Trace includes `knowledge_service: legal_knowledge`, `legal_search_query`, and `legal_adapters_selected: ["courtlistener"]`.

---

### QA-5L-B — Constitutional topic

**Prompt:**

```text
@legal First Amendment commercial speech Supreme Court precedent
```

**Pass if:**

- Sources include relevant Supreme Court or federal opinions.
- Legal disclaimer present.

---

### QA-5L-C — Fourth Amendment search

**Prompt:**

```text
@legal Fourth Amendment cell phone search warrant Supreme Court
```

**Pass if:**

- Sources reference cell-phone / search-warrant case law (e.g. *Riley*/*Carpenter* era cases acceptable).
- High authority scores for SCOTUS sources in trace.

---

### QA-5L-D — Preferred sources off (negative control)

1. Open **Settings → Knowledge → Preferred sources → Legal**.
2. Uncheck **CourtListener**.
3. Repeat QA-5L-A.

**Pass if:**

- Empty or gracefully degraded bundle — no fabricated case citations.
- Re-enable CourtListener and confirm QA-5L-A works again.

---

### QA-5L-E — Cross-domain regression

**Prompt:**

```text
@finance Apple Inc 10-K risk factors
```

**Pass if:**

- Finance service engaged; no CourtListener sources mixed in.

---

## Quick reference

| ID | Prompt | Key signal | Sign-off |
|----|--------|------------|----------|
| QA-5L-A | `@legal … Miranda … interrogation` | SCOTUS opinion, legal disclaimer | |
| QA-5L-B | `@legal First Amendment commercial speech` | Case law sources | |
| QA-5L-C | `@legal Fourth Amendment cell phone search` | Search/seizure opinions | |
| QA-5L-D | QA-5L-A with CourtListener **off** | Graceful empty | |
| QA-5L-E | `@finance Apple 10-K` | No legal leakage | |

---

## Automated regression

```bash
python3 -m unittest tests.test_legal_knowledge tests.test_courtlistener_adapter tests.test_knowledge_source_preferences -q
python3 tools/evaluate_retrieval.py --live --service legal_knowledge --min-pass 3
python3 tools/evaluate_retrieval.py --live --service finance_knowledge --min-pass 3
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py --live --service scientific_evidence --min-pass 5
```
