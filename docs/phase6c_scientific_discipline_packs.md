# Phase 6c — Scientific Discipline Packs (mini-phase)

**Status:** Phase 6c implemented (6c-0 … 6c-6)  
**Date:** 2026-06-25  
**Parent:** [External Knowledge Platform Plan](./external_knowledge_platform_plan.md) — Phase 6  
**Related:** [ADR 001 — Skills orthogonal to routing](./adr/001-skills-orthogonal-to-routing.md), [§21 Compositional entity resolution](./external_knowledge_platform_plan.md#21-adr-002-compositional-entity-resolution-registry), Slice 6a/6b manual QA ([discipline routing](./manual_qa_phase6_slice6_discipline_routing.md))

---

## 0. Executive summary

Slice 6a/6b introduced **heuristic discipline routing** and the first specialist adapters (DBLP, RePEc stub) inside the **`scientific_evidence` Knowledge Service**. Slice **6c expands this into a mini-phase**: a durable **discipline-pack registry** that maps scholarly domains to rigorous bibliographic/data sources — without promoting Finance, Legal, or other non-scientific verticals into the scientific tree.

**Core principle:** One `EvidenceBundle` contract, one Scientific service shell, many **discipline packs** underneath for adapter order, detection heuristics, entity hints, and eval tags.

```
Evidence
        ▲
        │
Knowledge Services
        │
 ├── Scientific (scientific_evidence)
 │       ├── Medicine        ← clinical / therapeutic queries
 │       ├── Biology         ← molecular, ecological, preprint-heavy life science
 │       ├── Chemistry
 │       ├── Physics
 │       ├── Computer Science
 │       ├── Economics       ← scholarly economics (not market filings)
 │       ├── Psychology
 │       ├── Sociology
 │       ├── Political Science
 │       ├── Earth & Environment  (general_science successor for geo/climate)
 │       └── general_science     (fallback)
 │
 ├── Finance (finance_knowledge)     ← SEC EDGAR, FRED — NOT a scientific discipline
 ├── Legal (legal_knowledge)           ← CourtListener — NOT a scientific discipline
 ├── Trusted Web (trusted_knowledge)
 └── Internal (internal_corpus)
```

---

## 1. Service boundaries (non-negotiable)

| Top-level service | Examples | Why separate |
|-------------------|----------|--------------|
| **`scientific_evidence`** | PubMed trials, arXiv preprints, RePEc working papers, PubChem compounds | Peer-reviewed / scholarly evidence; bibliographic rigor |
| **`finance_knowledge`** | 10-K filings, insider trades, macro series (FRED) | Regulatory/market primary sources; `@finance` composer token |
| **`legal_knowledge`** | Case law, dockets | Judicial/regulatory corpus; `@legal` token |
| **`trusted_knowledge`** | Wikipedia, gov health pages | General factual web with authority tiers |
| **`internal_corpus`** | User library LanceDB | Enterprise provenance |

**Economics, psychology, sociology, and political science are scientific disciplines** — they route through `scientific_evidence` with discipline-specific adapter emphasis. They are **not** sibling services to Finance or Legal.

**Anti-pattern:** Adding `@economics` as a top-level Knowledge Service duplicating RePEc/OpenAlex while `scientific_evidence` already covers scholarly economics.

---

## 2. Discipline pack vs entity pack

Two parallel registries, different jobs:

| Concept | Owns | Does not own |
|---------|------|--------------|
| **`ScientificDisciplinePack`** (6c) | Query classification, adapter priority, settings UI group, eval `discipline` tag, optional entity-pack **hints** | Retrieval HTTP calls, bundle assembly |
| **`EntityPackDefinition`** (§21) | Extractor/activator/linker grouping for `entity_ids` | Adapter selection |

A discipline pack **may reference** entity pack ids (e.g. `medicine` → `biomedical` entity pack) but must not embed extractors or import adapters.

---

## 3. Discipline taxonomy and source matrix

Sources are **API-first**, public, and suitable for structured metadata + abstracts/snippets. Paywalled indexes (Web of Science, PsycINFO, IEEE Xplore without key) are catalog **stubs** until keys or open alternatives exist.

| Discipline id | User label | Primary adapters (target order) | Secondary / fallback | Notes |
|---------------|------------|--------------------------------|----------------------|-------|
| `medicine` | Medicine | pubmed | openalex, europe_pmc | Clinical trials, therapeutics; maps from current `biomedical` |
| `biology` | Biology | pubmed, biorxiv | openalex, europe_pmc | Molecular/ecology/evolution; preprint emphasis |
| `chemistry` | Chemistry | pubchem, openalex | pubmed (chem journals) | Structures, compounds; PubChem PUG REST |
| `physics` | Physics | arxiv, inspire_hep, openalex | nasa_ads (stub) | 6c-5: INSPIRE-HEP live; NASA ADS key stub |
| `computer_science` | Computer Science | arxiv, dblp, openalex | acm_dl (keyed stub) | 6b: DBLP live |
| `economics` | Economics | repec, openalex | ssrn (stub) | 6c-4: RePEc live via EconBiz; SSRN fixture stub |
| `psychology` | Psychology | pubmed, openalex | psycinfo (stub) | Much overlap with medicine; PsycINFO paywalled |
| `sociology` | Sociology | openalex | socarxiv (stub) | SocArXiv open preprints |
| `political_science` | Political Science | openalex | ssrn (stub) | Working papers + journals via OpenAlex concepts |
| `earth_environment` | Earth & Environment | openalex | noaa, nasa_earthdata (stubs) | Climate, geoscience, remote sensing |
| `general_science` | General science | openalex, arxiv | pubmed | Current fallback; interdisciplinary |

**Universal fallback:** OpenAlex remains the cross-disciplinary backstop for every pack when primary adapters return empty or time out.

---

## 4. Mini-phase slices (recommended order)

| Slice | Focus | Deliverables | Exit criteria |
|-------|--------|--------------|---------------|
| **6c-0 Foundation** | Registry + docs | `scientific_discipline_packs.py`, this doc, plan cross-links | All disciplines defined with `status`; existing 6a behavior unchanged |
| **6c-1 Life sciences split** | Medicine vs biology | Detection patterns; `biorxiv` adapter stub + Europe PMC live; eval tags | **IMPLEMENTED (2026-06-25)** — 2 eval queries; `biomedical` trace alias preserved |
| **6c-2 Chemistry** | PubChem | `adapters/pubchem.py` + fixture; catalog Chemistry UI group | **IMPLEMENTED (2026-06-25)** — `chem_001` eval; live PUG REST |
| **6c-3 Social sciences** | Psych / soc / polisci | Heuristic patterns; OpenAlex primary for soc/polisci | **IMPLEMENTED (2026-06-25)** — `psych_001`, `soc_001`, `polisci_001` eval |
| **6c-4 Economics depth** | Live RePEc + SSRN stub | EconBiz live search for RePEc adapter; SSRN fixture stub | **IMPLEMENTED (2026-06-25)** — `econ_001` primary `repec` |
| **6c-5 Physics depth** | INSPIRE-HEP live + ADS stub | Open INSPIRE REST API; NASA ADS catalog stub (API key) | **IMPLEMENTED (2026-06-25)** — `phys_001` primary arxiv; inspire in trace |
| **6c-6 Eval + QA** | Corpus expansion | Harness per-group primary gates; 12-query corpus; manual QA sign-off | **IMPLEMENTED (2026-06-25)** — `--min-pass 12`, per-discipline ≥70% |

**Parallel track (optional):** Sidecar discipline **suggestions** (telemetry only, ADR 001) — must not override heuristic routing in v1.

---

## 5. Detection and routing rules

1. **Single primary discipline per turn** (v1) — same as 6a; ties break by configured priority list.
2. **Medicine before biology** — clinical/therapeutic signals (`BIOMEDICAL_ACTIVATOR`, FDA, trial language) → `medicine`.
3. **Biology** — gene/protein/species/ecology/biorxiv vocabulary without dominant clinical framing.
4. **Never route finance/legal tokens** — `@finance` / `@legal` composer attachments bypass scientific discipline routing entirely (existing service registry).
5. **User adapter prefs always win on membership** — discipline pack only **reorders** enabled adapters (existing `apply_scientific_adapter_policy`).
6. **Trace fields** (extend, do not rename): `scientific_discipline`, `scientific_discipline_label`, `scientific_discipline_pack_version`, `scientific_discipline_adapters_planned`.

Backward compatibility: trace value `biomedical` remains accepted as alias for `medicine` until eval corpus migrates.

---

## 6. Settings UI evolution

Today: Scientific literature grouped as **Science**, **Computer Science**, **Economics**.

Target (incremental):

- Settings → Knowledge → **Scientific literature** expands into discipline sub-groups matching the pack registry.
- Adapters appear under every discipline group they serve (same adapter id may appear in multiple groups — already true for arXiv/OpenAlex).
- Finance and Legal settings panels **unchanged** — separate services.

---

## 7. Eval strategy

Extend `eval/retrieval_corpus/v1_scientific.json`:

```json
{
  "id": "chem_001",
  "query": "aspirin acetylsalicylic acid binding COX-2",
  "discipline": "chemistry",
  "primary_adapter": "pubchem",
  "expect_adapters": ["pubchem", "openalex"],
  "medical": false
}
```

Harness checks (existing + new):

- `discipline` matches `detect_scientific_discipline().discipline`
- `primary_adapter` appears in retrieved bundle adapters
- `--min-discipline-primary-rate 0.70` across corpus (default for scientific live eval)
- `--min-discipline-group-primary-rate 0.70` — **each** discipline tag must meet threshold
- Catalog default adapters during eval (pass `--user-prefs` to use saved Settings)

Deep-research regression: `evaluate_deep_research.py --live --require-relevance --min-relevance-ok 3` unchanged.

---

## 8. Explicit non-goals (6c)

- New top-level Knowledge Services for Economics, Psychology, etc.
- Paywalled full-text scraping (IEEE, Elsevier, PsycINFO) without user API keys
- LLM/sidecar as **authoritative** discipline router
- Skills selecting adapters
- Merging Finance EDGAR into scientific economics routing
- Neo4j or external discipline ontologies in v1

---

## 9. Implementation map

| Artifact | Purpose |
|----------|---------|
| `core/knowledge/scientific_discipline_packs.py` | Canonical pack registry |
| `core/knowledge/scientific_discipline.py` | Detection (migrate patterns → packs over time) |
| `core/knowledge/adapters/catalog.py` | Per-discipline UI groups + adapter stubs |
| `core/knowledge/adapters/*.py` | One module per new primary source |
| `eval/retrieval_corpus/v1_scientific.json` | Tagged discipline eval queries |
| `eval/fixtures/knowledge/` | HTTP fixtures for offline tests |
| `tools/evaluate_retrieval.py` | Discipline + primary-adapter gates |
| `docs/manual_qa_phase6_slice6_discipline_routing.md` | Extended QA cases per new discipline |

---

## 10. Success metrics

| Metric | Target |
|--------|--------|
| Disciplines with ≥1 implemented primary adapter | ≥ 6 by end of 6c-3 |
| Scientific eval corpus size | ≥ 12 discipline-tagged queries |
| Primary-adapter hit rate (live eval) | ≥ 70% per discipline group |
| Deep-research relevance regression | 3/3 maintained |
| Finance/legal isolation | 0 scientific adapter leakage on `@finance` / `@legal` turns |

---

## 11. Next action (6c-0)

1. Land **`scientific_discipline_packs.py`** registry (all disciplines, `status: active | stub | planned`).
2. Wire **`preferred_adapters_for_discipline`** to read primary adapter order from packs when `status=active`.
3. Keep **`biomedical` → `medicine` alias** in detection for one release.
4. Open PR **6c-1** (life sciences split) after 6c-0 merges to `dev`.
