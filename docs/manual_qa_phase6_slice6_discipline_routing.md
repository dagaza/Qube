# Manual QA — Phase 6 Slice 6 + Phase 6c (Discipline packs)

**Purpose:** In-app validation of **Slice 6a/6b** (heuristic discipline routing, DBLP/RePEc) and **Phase 6c** (life sciences, chemistry, social sciences, economics depth, physics/INSPIRE-HEP, eval harness).

**Related:** [Phase 6c discipline packs](./phase6c_scientific_discipline_packs.md), [External knowledge platform plan](./external_knowledge_platform_plan.md), [Retrieval eval README](../eval/retrieval_corpus/README.md)

---

## What changed (tester summary)

| Slice | Behavior |
|-------|----------|
| **6a** | `@evidence` / `@science` queries classified into a **discipline bucket**. Enabled adapters **re-ordered** per discipline pack. PubMed **gated off** non-biomedical queries. |
| **6b** | **DBLP** (CS) live; **RePEc** economics routing. |
| **6c-1** | Medicine vs **biology** split; **bioRxiv** adapter. |
| **6c-2** | **Chemistry** → PubChem. |
| **6c-3** | **Psychology**, **sociology**, **political science** routing. |
| **6c-4** | **RePEc** live via EconBiz (open economics index). |
| **6c-5** | **INSPIRE-HEP** live (open HEP/astrophysics); NASA ADS catalog stub (API key). |
| **6c-6** | **12-query** eval corpus; harness gates **≥70% primary rate per discipline group**. |

**Terminology:** `@evidence` and `@science` → **Scientific literature** (`scientific_evidence`). Separate from `@finance` / `@legal`. Composer overrides (`@pubmed`, `@arxiv`) still win for one turn.

---

## Prerequisites

### Settings (Settings → Knowledge)

| Toggle / control | Required for |
|------------------|--------------|
| **External knowledge pipeline (v2)** | All cases — must be **ON** |
| **Preferred sources → Scientific literature** | QA-6A / QA-6B / QA-6C — enable discipline adapters below |
| **Internet / web retrieval** | All live adapter calls |

**Implemented sources by UI group (enable for full 6c coverage):**

| UI group | Adapters |
|----------|----------|
| **Science** | PubMed, OpenAlex, arXiv, INSPIRE-HEP |
| **Biology** | PubMed, bioRxiv, OpenAlex |
| **Chemistry** | PubChem, OpenAlex, PubMed |
| **Computer Science** | arXiv, OpenAlex, DBLP |
| **Economics** | RePEc, OpenAlex |
| **Psychology** | PubMed, OpenAlex |
| **Social Science** | OpenAlex |

Stubs (disabled): SSRN, SocArXiv, PsycINFO, NASA ADS (API key), Semantic Scholar.

### Automated baseline (run before manual QA)

```bash
python3 -m unittest \
  tests.test_scientific_discipline \
  tests.test_scientific_discipline_packs \
  tests.test_biorxiv_adapter \
  tests.test_pubchem_adapter \
  tests.test_inspire_hep_adapter \
  tests.test_repec_adapter \
  tests.test_ssrn_adapter \
  tests.test_dblp_adapter \
  tests.test_knowledge_source_preferences \
  tests.test_scientific_query_planner \
  tests.test_evaluate_retrieval -q

# Live scientific eval (12/12 ok + per-discipline primary ≥ 70%)
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py \
  --live --service scientific_evidence --min-pass 12
```

**Note:** The eval harness uses **catalog default adapters**, not your saved Settings, unless you pass `--user-prefs`.

### Audit logging (recommended)

Enable web search audit logging so you can confirm discipline routing without guessing from UI alone:

- Set env `QUBE_WEB_SEARCH_AUDIT_LOG=1` before launch, **or** use your existing audit toggle if configured.
- Inspect `~/.qube/logs/web_search.log` for `retrieval_trace` / `relevance_diag` on each `@evidence` turn.

**Key trace fields (Slice 6a):**

| Field | Example values | Meaning |
|-------|----------------|---------|
| `scientific_discipline` | `biomedical`, `computer_science`, `economics`, `physics`, `general_science` | Detected bucket |
| `scientific_discipline_ui_group` | `Science`, `Computer Science`, `Economics` | Catalog group used for ordering |
| `scientific_adapters_selected` | e.g. `["arxiv", "openalex", "dblp"]` | Final ordered adapter list for the turn |
| `scientific_discipline_scores` | heuristic counts | Optional tie-break detail |

**Cache tip:** If you re-run the same prompt and results look stale, restart with `QUBE_EVIDENCE_CACHE=0` or use a slightly rephrased query.

**Domain regression (no leakage into finance / legal):**

```bash
python3 tools/evaluate_retrieval.py --live --service finance_knowledge --min-pass 3
python3 tools/evaluate_retrieval.py --live --service legal_knowledge --min-pass 3
```

---

## Slice 6a — Discipline detection and adapter ordering

All prompts use `@evidence` unless noted. Pass criteria assume default preferred sources (above).

### QA-6A-A — Biomedical (PubMed primary)

**Prompt:**

```text
@evidence Ozempic semaglutide cardiovascular outcomes randomized trial
```

**Pass if:**

- Answer discusses semaglutide / cardiovascular trial evidence.
- **Sources** include **PubMed** (`pubmed` adapter) with journal-style abstracts.
- Trace: `scientific_discipline: biomedical`, `scientific_discipline_ui_group: Science`.
- `scientific_adapters_selected` lists **pubmed** before openalex/arxiv.

---

### QA-6A-B — Computer science (arXiv emphasis)

**Prompt:**

```text
@evidence transformer attention mechanism neural machine translation
```

**Pass if:**

- Sources include **arXiv** and/or **OpenAlex** ML/NLP papers (e.g. “Attention Is All You Need” class results acceptable).
- **PubMed is not** the only adapter and ideally **absent** from sources on this non-medical CS query.
- Trace: `scientific_discipline: computer_science`, `scientific_discipline_ui_group: Computer Science`.
- `scientific_adapters_selected` starts with **`arxiv`** (then `openalex`, optionally `dblp`).

---

### QA-6A-C — Physics (arXiv + INSPIRE-HEP)

**Prompt:**

```text
@evidence gravitational wave detection LIGO binary black hole
```

**Pass if:**

- Sources are astrophysics / gravitational-wave literature (arXiv preprints acceptable).
- Trace: `scientific_discipline: physics`; `scientific_adapters_selected` includes **`arxiv`** and optionally **`inspire_hep`**.
- **INSPIRE-HEP** may enrich metadata (citations, curated HEP index) alongside arXiv.
- No spurious clinical PubMed-only results.

---

### QA-6A-D — Economics (OpenAlex primary on live RePEc)

**Prompt:**

```text
@evidence monetary policy inflation econometric VAR model central bank
```

**Pass if:**

- Answer references macro / monetary policy econometrics.
- **Sources** include **OpenAlex** scholarly works.
- Trace: `scientific_discipline: economics`, `scientific_discipline_ui_group: Economics`.
- `scientific_adapters_selected` lists **`repec`** before **`openalex`** even if RePEc returns no live rows (expected today).

**Note:** Live RePEc search is intentionally empty without an approved API key. OpenAlex fallback is correct behavior for Slice 6b.

---

### QA-6A-E — General science (OpenAlex default)

**Prompt:**

```text
@evidence climate change Arctic sea ice extent satellite observations
```

**Pass if:**

- Sources include climate / remote-sensing literature via **OpenAlex** (arXiv may also appear).
- Trace: `scientific_discipline: general_science`.
- PubMed not required; biomedical gating should not force PubMed-only retrieval.

---

### QA-6A-F — `@science` alias parity

Repeat **QA-6A-B** using:

```text
@science transformer attention mechanism neural machine translation
```

**Pass if:** Same discipline routing and adapter signals as QA-6A-B (`scientific_evidence` service, CS discipline).

---

### QA-6A-G — PubMed gating (non-medical negative control)

**Prompt:**

```text
@evidence quantum computing error correction surface codes 2024
```

**Pass if:**

- Sources emphasize **arXiv** / **OpenAlex** physics or CS literature.
- Trace shows **PubMed absent** from `scientific_adapters_selected` (or at least no PubMed-only bundle).
- `scientific_discipline` is **not** `biomedical` unless query explicitly triggers medical heuristics.

---

### QA-6A-H — Preferred sources reorder (disable arXiv)

1. Open **Settings → Knowledge → Preferred sources → Scientific literature → Computer Science**.
2. Uncheck **arXiv** (leave OpenAlex and DBLP on).
3. Run QA-6A-B prompt again.

**Pass if:**

- Trace `scientific_adapters_selected` **excludes arxiv**; OpenAlex and/or DBLP still retrieve sources.
- App remains stable; answer still grounded in scholarly sources.
4. Re-enable **arXiv** and confirm QA-6A-B passes again.

---

## Slice 6b — Specialist adapters (RePEc, DBLP) and eval harness

### QA-6B-A — Settings visibility (RePEc + DBLP)

1. Open **Settings → Knowledge → Preferred sources → Scientific literature**.
2. Expand **Computer Science** and **Economics** groups.

**Pass if:**

- **DBLP** checkbox is **enabled** (not “coming soon”) under Computer Science.
- **RePEc** checkbox is **enabled** under Economics.
- **SSRN** still shows **coming soon** (disabled).

---

### QA-6B-B — DBLP participates in CS adapter chain

**Prompt:**

```text
@evidence BERT pre-training deep bidirectional transformers language understanding
```

**Pass if:**

- Trace `scientific_adapters_selected` includes **`dblp`** for a computer-science-classified query.
- Final cited sources may still be dominated by arXiv/OpenAlex (ranking merges adapters) — that is acceptable.
- At least one scholarly source appears; no empty bundle.

**Optional:** Run with audit log and confirm a DBLP API call occurred (adapter attempted) even if ranked rows come from arXiv.

---

### QA-6B-C — Economics RePEc live + OpenAlex fallback

**Prompt:**

```text
@evidence central bank inflation targeting Taylor rule empirical estimates
```

**Pass if:**

- Trace shows `scientific_discipline: economics` and **`repec`** in `scientific_adapters_selected`.
- **RePEc** sources appear in the bundle (live via EconBiz / IDEAS-indexed metadata) when the adapter is enabled.
- **OpenAlex** may appear as fallback when RePEc is disabled or returns sparse hits.
- No fabricated RePEc handles or IDEAS URLs unless actually retrieved.

### QA-6C-G — Economics eval query (monetary policy / VAR)

**Prompt:**

```text
@evidence monetary policy inflation econometric VAR model central bank
```

**Pass if:**

- Trace: `scientific_discipline: economics`.
- **RePEc** in adapters (primary); OpenAlex optional secondary.
- Not routed to finance_knowledge (SEC/FRED) or legal services.

---

### QA-6B-D — Composer `@arxiv` override

**Prompt:**

```text
@arxiv gravitational wave detection LIGO
```

**Pass if:**

- Only **arXiv** adapter used for the turn (composer override beats discipline policy).
- Sources are arXiv preprints; trace adapter list is arxiv-only.

---

### QA-6B-E — Composer `@pubmed` override (medical)

**Prompt:**

```text
@pubmed SGLT2 inhibitors heart failure hospitalization trials
```

**Pass if:**

- PubMed-sourced abstracts dominate the bundle regardless of other enabled adapters.
- Trace reflects pubmed-first retrieval for the turn.

---

### QA-6B-F — Cross-domain regression (finance / legal unchanged)

**Turn 1:**

```text
@finance Apple Inc 10-K risk factors
```

**Turn 2:**

```text
@legal Miranda v Arizona police interrogation rights
```

**Pass if:**

- Finance turn uses **SEC EDGAR** only (`finance_knowledge`); no RePEc/DBLP/PubMed leakage.
- Legal turn uses **CourtListener** only (`legal_knowledge`); discipline fields absent or irrelevant.
- Finance / legal disclaimers still present where applicable.

---

### QA-6B-G — Deep research / multi-step regression (optional)

**Prompt:**

```text
@research ACE inhibitors heart failure mortality evidence
```

**Pass if:**

- Deep research completes without error.
- Merged bibliography still includes biomedical literature; discipline routing on sub-queries does not break merge.
- No regression vs pre–Slice 6 behavior (coverage adequate+, cited Findings).
- Merged source count ≥ 2 (not collapsed by pre-rank title gates — see [ADR 002](adr/002-merge-ranker-v2-deep-research.md)).

**Note:** A pre–Merge Ranker v2 failure mode here was `merged_sources_post_filter: 1` with `merged_title_anchor_dropped: 7` in `web_search.log`; v2 uses weighted ranking instead of a title-first anchor gate.

---

## Phase 6c-1 — Life sciences (Medicine vs Biology)

### QA-6C-A — Biology (molecular / ecology, not clinical)

**Prompt:**

```text
@evidence microbiome diversity soil ecology metagenomics
```

**Pass if:**

- Trace: `scientific_discipline: biology` (not `biomedical`).
- Adapters include **PubMed**; **bioRxiv** attempted when enabled in settings.
- No finance/legal adapters.

### QA-6C-B — Medicine unchanged (clinical query)

**Prompt:**

```text
@evidence Ozempic semaglutide cardiovascular outcomes randomized trial
```

**Pass if:**

- Trace: `scientific_discipline: biomedical` (medicine alias).
- **PubMed** primary; discipline routing unchanged from QA-6A-A.

### QA-6C-C — Chemistry (PubChem compound query)

**Prompt:**

```text
@evidence aspirin acetylsalicylic acid binding COX-2 cyclooxygenase
```

**Pass if:**

- Trace: `scientific_discipline: chemistry`.
- **PubChem** in adapters; compound record includes CID / molecular metadata when available.
- Not routed to finance/legal services.

### QA-6C-D — Psychology (cognitive / experimental query)

**Prompt:**

```text
@evidence working memory cognitive load dual-task experiment psychology
```

**Pass if:**

- Trace: `scientific_discipline: psychology`.
- **PubMed** in adapters (primary when enabled); OpenAlex as fallback.
- Not routed to finance/legal services.

### QA-6C-E — Sociology (stratification / inequality query)

**Prompt:**

```text
@evidence social stratification income inequality sociology survey methods
```

**Pass if:**

- Trace: `scientific_discipline: sociology`.
- **OpenAlex** in adapters; discipline UI group **Social Science**.

### QA-6C-F — Political science (electoral / comparative politics query)

**Prompt:**

```text
@evidence voter turnout electoral reform democracy comparative political science
```

**Pass if:**

- Trace: `scientific_discipline: political_science`.
- **OpenAlex** in adapters; discipline UI group **Social Science**.

### QA-6C-H — Physics depth (INSPIRE-HEP trace)

**Prompt:**

```text
@evidence gravitational wave detection LIGO binary black hole
```

**Pass if:**

- Trace: `scientific_discipline: physics`.
- **arXiv** primary in bundle; **inspire_hep** in adapters or adapter trace when enabled.
- NASA ADS not required (catalog stub — API key only).

---

## Quick reference

| ID | Prompt (abbrev.) | Discipline (expected) | Primary adapter signal | Sign-off |
|----|------------------|----------------------|-------------------------|----------|
| QA-6A-A | `@evidence … Ozempic semaglutide … trial` | biomedical | pubmed | |
| QA-6A-B | `@evidence … transformer attention … NMT` | computer_science | arxiv | |
| QA-6A-C | `@evidence … LIGO … black hole` | physics | arxiv | |
| QA-6A-D | `@evidence … monetary policy … VAR` | economics | repec | |
| QA-6A-E | `@evidence … Arctic sea ice …` | general_science | openalex | |
| QA-6A-F | `@science … transformer attention …` | computer_science | same as 6A-B | |
| QA-6A-G | `@evidence … quantum error correction …` | not biomedical | no pubmed-only | |
| QA-6A-H | 6A-B with arXiv **off** in settings | computer_science | openalex/dblp | |
| QA-6B-A | Settings: DBLP + RePEc enabled | — | UI checkboxes | |
| QA-6B-B | `@evidence … BERT … transformers` | computer_science | dblp in trace | |
| QA-6B-C | `@evidence … Taylor rule …` | economics | repec selected, openalex sources | |
| QA-6B-D | `@arxiv … LIGO` | override | arxiv only | |
| QA-6B-E | `@pubmed … SGLT2 …` | override | pubmed only | |
| QA-6B-F | `@finance …` then `@legal …` | — | no scientific leakage | |
| QA-6B-G | `@research … ACE inhibitors …` | optional | merge ok | |
| QA-6C-A | `@evidence microbiome … metagenomics` | biology | pubmed (+ biorxiv) | |
| QA-6C-B | `@evidence Ozempic … trial` | biomedical | pubmed | |
| QA-6C-C | `@evidence aspirin … COX-2` | chemistry | pubchem | |
| QA-6C-D | `@evidence working memory … dual-task` | psychology | pubmed | |
| QA-6C-E | `@evidence social stratification … sociology` | sociology | openalex | |
| QA-6C-F | `@evidence voter turnout … political science` | political_science | openalex | |
| QA-6C-G | `@evidence monetary policy … VAR` | economics | repec | |
| QA-6C-H | `@evidence … LIGO … black hole` | physics | arxiv (+ inspire_hep) | |

---

## Sign-off block

Record for each session:

1. **Qube version / branch / commit**
2. **Session ID** (from `~/.qube/logs/qube.log` if audit enabled)
3. **Settings snapshot:** External knowledge v2 ON; which scientific adapters enabled
4. **Automated eval result:** `12/12 ok`, overall primary rate ___%, per-discipline groups all ≥70%
5. **Manual cases passed:** QA-6A-A … QA-6C-H (checklist above)

**Phase 6c manual QA completed:** ☐ Yes ☐ No — **Tester:** __________ **Date:** __________

---

## Automated regression (post–manual QA)

```bash
python3 -m unittest \
  tests.test_scientific_discipline \
  tests.test_scientific_discipline_packs \
  tests.test_biorxiv_adapter \
  tests.test_pubchem_adapter \
  tests.test_inspire_hep_adapter \
  tests.test_repec_adapter \
  tests.test_ssrn_adapter \
  tests.test_dblp_adapter \
  tests.test_knowledge_source_preferences \
  tests.test_scientific_query_planner \
  tests.test_evaluate_retrieval -q

QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py \
  --live --service scientific_evidence --min-pass 12

python3 tools/evaluate_retrieval.py --live --service finance_knowledge --min-pass 3
python3 tools/evaluate_retrieval.py --live --service legal_knowledge --min-pass 3
```

**Developer-only RePEc fixture smoke test** (not required for product QA):

```bash
QUBE_KNOWLEDGE_FIXTURES=1 python3 -m unittest tests.test_repec_adapter -q
```
