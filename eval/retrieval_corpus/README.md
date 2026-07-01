# Retrieval eval corpora

JSON query sets for live validation of Qube's external knowledge platform. Used by `tools/evaluate_retrieval.py` and `tools/evaluate_deep_research.py`.

## Corpora

| File | Service | Purpose |
|------|---------|---------|
| `v1_scientific.json` | `scientific_evidence` | **Phase 6c** — 12 discipline-tagged scholarly queries (`discipline` + `primary_adapter`) |
| `v1_trusted.json` | `trusted_knowledge` | Phase 6 Slice 1 `@trusted` — Wikipedia-first, authority tiers |
| `v1_finance.json` | `finance_knowledge` | Phase 6 Slice 5a `@finance` — SEC EDGAR filings |
| `v1_legal.json` | `legal_knowledge` | Phase 6 Slice 5b `@legal` — CourtListener case law |
| `v1_deep_research.json` | deep research merge | Phase 5 topical relevance on merged bundles |

## Scientific corpus (Phase 6c-6)

| Query ID | Discipline | Primary adapter |
|----------|------------|-----------------|
| `bio_001`, `bio_002` | biomedical | pubmed |
| `bio_003`, `bio_004` | biology | pubmed |
| `chem_001` | chemistry | pubchem |
| `cs_001` | computer_science | arxiv |
| `phys_001` | physics | arxiv |
| `econ_001` | economics | repec |
| `psych_001` | psychology | pubmed |
| `soc_001` | sociology | openalex |
| `polisci_001` | political_science | openalex |
| `cross_001` | general_science | openalex |

**Exit gates (live):**

- `12/12` queries `status=ok` (`--min-pass 12`)
- Overall primary-adapter hit rate ≥ **70%**
- **Each discipline group** primary rate ≥ **70%** (e.g. both biomedical queries must hit `pubmed`)

The harness uses **catalog default adapters** (not your saved Settings prefs) unless you pass `--user-prefs`. Inter-query delay (2s) and one automatic retry on `no_results` reduce OpenAlex rate-limit / 503 flakes.

## Commands

```bash
# Scientific (Phase 6c — expect 12/12 ok, per-discipline primary ≥ 70%)
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py \
  --live --service scientific_evidence --min-pass 12

# Trusted (Phase 6 Slice 1 — expect ≥ 4/5 ok)
python3 tools/evaluate_retrieval.py --live --service trusted_knowledge

# Finance (Phase 6 Slice 5a — expect ≥ 3/4 ok)
python3 tools/evaluate_retrieval.py --live --service finance_knowledge --min-pass 3

# Legal (Phase 6 Slice 5b — expect ≥ 3/4 ok)
python3 tools/evaluate_retrieval.py --live --service legal_knowledge --min-pass 3

# Deep research (Phase 5 — expect 3/3 relevance_ok)
python3 tools/evaluate_deep_research.py --live --require-relevance --min-relevance-ok 3
```

Dry-run (no network):

```bash
python3 tools/evaluate_retrieval.py --corpus eval/retrieval_corpus/v1_trusted.json
```

Evaluate with your saved adapter preferences (optional):

```bash
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py \
  --live --service scientific_evidence --min-pass 12 --user-prefs
```

## Manual QA playbook (Phase 6 Slice 1)

Enable **External knowledge v2** and web audit logging. Record `session_id` from `~/.qube/logs/qube.log` or routing debug.

| # | Composer | Query | Expected adapter chain | Coverage |
|---|----------|-------|------------------------|----------|
| 1 | `@internet` | "latest news about Mars rover" | `duckduckgo` | adequate+ SERP snippets |
| 2 | `@trusted` | "capital of Japan" | `wikipedia_api` | Wikipedia extract, authority ≥ 0.9 |
| 3 | `@evidence` | "SGLT2 inhibitors heart failure trials" | `pubmed`, `openalex` | ≥2 sources, abstracts |
| 4 | `@evidence` | "quantum computing error correction 2024" | `arxiv` and/or `openalex` | non-medical, no disclaimer-only empty |
| 5 | `@research` | "ACE inhibitors heart failure mortality evidence" | merged `deep_research_merged` | relevance filter, cited Findings |

**Trace audit:** Each `@trusted` / `@evidence` turn should emit one `retrieval_trace` line in `~/.qube/logs/web_search.log` when audit is enabled (`schema_version: 2`, matching `knowledge_service`).

**Empty-result check:** `@evidence` on nonsense string should downgrade gracefully (no hallucinated `[W]` cites).

See also: [Manual QA — Slice 6 discipline routing](../docs/manual_qa_phase6_slice6_discipline_routing.md).
