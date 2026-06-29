# Retrieval eval corpora

JSON query sets for live validation of Qube's external knowledge platform. Used by `tools/evaluate_retrieval.py` and `tools/evaluate_deep_research.py`.

## Corpora

| File | Service | Purpose |
|------|---------|---------|
| `v1_scientific.json` | `scientific_evidence` | Multi-disciplinary scholarly literature (`@evidence` / `@science`) — `discipline` + `primary_adapter` tags for Slice 6 routing eval |
| `v1_trusted.json` | `trusted_knowledge` | Phase 6 Slice 1 `@trusted` — Wikipedia-first, authority tiers |
| `v1_finance.json` | `finance_knowledge` | Phase 6 Slice 5a `@finance` — SEC EDGAR filings |
| `v1_legal.json` | `legal_knowledge` | Phase 6 Slice 5b `@legal` — CourtListener case law |
| `v1_deep_research.json` | deep research merge | Phase 5 topical relevance on merged bundles |

## Commands

```bash
# Scientific (Phase 2 sign-off — expect 6/6 ok, discipline primary ≥ 70%)
QUBE_EVIDENCE_CACHE=0 python3 tools/evaluate_retrieval.py --live --service scientific_evidence --min-pass 6

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
