# Manual QA — Phase 6 Slice 5a (Finance)

**Purpose:** In-app validation of `@finance` / SEC EDGAR retrieval, finance disclaimers, and preferred-source settings after enabling Phase 6 knowledge features.

**Related:** [External knowledge platform plan](./external_knowledge_platform_plan.md) (Phase 6 Slice 5), [Manual QA Slices 2–4](./manual_qa_phase6.md), [Retrieval eval README](../eval/retrieval_corpus/README.md)

---

## Prerequisites

### Settings (Settings → Knowledge)

| Toggle | Required for |
|--------|----------------|
| **External knowledge pipeline (v2)** | All cases — must be **ON** |
| **Preferred sources → Finance → SEC EDGAR** | QA-5A–5D — **checked** (default) |

Ensure **internet / web retrieval** is enabled.

**Terminology:** `@finance` routes to the **Finance** knowledge service (`finance_knowledge`). It is separate from **Scientific literature** (`@evidence` / `@science`). Both produce the same **Evidence** output model (`EvidenceBundle`).

### Automated baseline (run before manual QA)

```bash
python3 tools/evaluate_retrieval.py --live --service finance_knowledge --min-pass 3
python3 -m unittest tests.test_finance_knowledge tests.test_sec_edgar_adapter -q
```

**Live eval target:** ≥ 3/4 queries `ok` on `eval/retrieval_corpus/v1_finance.json`.

**Scientific regression (required):**

```bash
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py --live --service scientific_evidence --min-pass 5
```

Use `QUBE_EVIDENCE_CACHE=0` when re-running after a prior partial run so stale cache does not mask adapter selection.

---

## Slice 5a — Finance (`@finance`)

Routes `@finance` through `finance_knowledge` → SEC EDGAR adapter. Bundle includes `not_financial_advice` warning.

### QA-5A — Company name + form type (10-K)

**Prompt:**

```text
@finance What are Apple Inc 10-K risk factors?
```

**Pass if:**

- Answer references Apple SEC filings (10-K, risk factors, or filing metadata).
- **Sources** dialog shows SEC EDGAR provenance (`sec_edgar` adapter).
- Transparency / “why” summary reflects finance knowledge service — not general web search.
- **`not_financial_advice`** appears in bundle warnings / transparency (wording may vary in UI).

**Optional (audit):** Retrieval trace includes `knowledge_service: finance_knowledge`, `adapter_calls: ["sec_edgar"]`, and `finance_search_query` in relevance diagnostics.

---

### QA-5B — Ticker symbol query

**Prompt:**

```text
@finance AMZN recent SEC filings
```

**Pass if:**

- Sources include Amazon (`AMAZON COM INC` or similar) SEC filings.
- Multiple form types may appear (10-K, 10-Q, 8-K) — acceptable.
- Finance disclaimer present.

---

### QA-5C — Quarterly filing (10-Q)

**Prompt:**

```text
@finance Microsoft quarterly 10-Q revenue
```

**Pass if:**

- At least one Microsoft 10-Q filing in sources.
- Authority scores reflect primary SEC source tier (high authority in trace).
- No scientific / PubMed adapters engaged.

---

### QA-5D — Preferred sources off (negative control)

1. Open **Settings → Knowledge → Preferred sources → Finance**.
2. Uncheck **SEC EDGAR**.
3. Run QA-5A prompt again.

**Pass if:**

- Empty or gracefully degraded bundle — no fabricated SEC citations.
- Re-enable SEC EDGAR and confirm QA-5A works again.

---

### QA-5E — Scientific regression (same session)

**Prompt:**

```text
@evidence transformer attention mechanism neural machine translation
```

**Pass if:**

- Scientific literature service engaged (`scientific_evidence`).
- OpenAlex and/or arXiv sources; **no** SEC EDGAR sources.
- No spurious biomedical entities in transparency (optional QA-3E overlap).

---

## Quick reference

| ID | Prompt | Key signal | Sign-off |
|----|--------|------------|----------|
| QA-5A | `@finance … Apple … 10-K …` | SEC EDGAR hit, finance disclaimer | |
| QA-5B | `@finance AMZN recent SEC filings` | Ticker resolution, multi-form filings | |
| QA-5C | `@finance Microsoft … 10-Q …` | 10-Q hit, high authority | |
| QA-5D | QA-5A with SEC EDGAR **off** | Graceful empty / no fake filings | |
| QA-5E | `@evidence transformer attention …` | No finance leakage; scholarly adapters | |

---

## Recording results

For each case, capture:

1. **Session ID** (for audit log correlation, if enabled)
2. **Pass / fail** against criteria above
3. Screenshot or copy of Sources transparency (`why_summary`, warnings)
4. For QA-5D: confirm checkbox state before and after

### Slice 5a sign-off template

| Area | Verdict | Notes |
|------|---------|-------|
| QA-5A–5C (happy path) | | |
| QA-5D (sources off) | | |
| QA-5E (scientific regression) | | |
| Live eval `v1_finance.json` | | ≥ 3/4 ok |

---

## Automated regression (companion)

```bash
python3 -m unittest tests.test_finance_knowledge tests.test_sec_edgar_adapter tests.test_knowledge_source_preferences -q
python3 tools/evaluate_retrieval.py --live --service finance_knowledge --min-pass 3
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py --live --service scientific_evidence --min-pass 5
python3 tools/evaluate_deep_research.py --live --require-relevance --min-relevance-ok 3
```
