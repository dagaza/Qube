# Live Knowledge Adapters — Inventory & Platform Status

**Last updated:** July 2026  
**Adapter source of truth:** `core/knowledge/adapters/registry.py` (`SEARCH_FUNCTIONS`) and `core/knowledge/adapters/catalog.py`  
**Platform source of truth:** [Knowledge Adapter HTTP Resilience Plan](./knowledge_adapter_http_resilience_plan.md)

This document lists every **live** external-knowledge adapter wired into Qube’s configurable retrieval pipelines, summarizes the **platform work shipped with them**, and tracks a **planned expansion backlog** from an external architecture review (July 2026).

It covers the three user-facing knowledge domains exposed in **Settings → Knowledge → Preferred Sources**:

| Domain | Knowledge service ID | Composer tokens |
|--------|----------------------|-----------------|
| **Scientific literature** | `scientific_evidence` | `@evidence`, `@science`, `@pubmed`, `@arxiv`, … |
| **Finance** | `finance_knowledge` | `@finance` |
| **Legal** | `legal_knowledge` | `@legal` |

**Total live adapters: 58** (46 scientific · 9 finance · 7 legal)

Finance and legal are **top-level domains**. Scientific literature is one domain with **multiple UI subdomains** (discipline-oriented groupings in Settings and discipline-pack routing). **New UI groups (Slice 12):** Engineering, Agriculture & Nutrition.

---

## Implementation status (since this inventory)

The adapter tables below reflect the **Tier 3–4 expansion** (Slices 7a–11). Alongside those adapters, Qube now routes outbound knowledge HTTP through a shared resilience stack and exposes credentials + health in Settings.

| Workstream | Scope | Status | Key modules |
|------------|-------|--------|-------------|
| **HTTP resilience Slice 1** | Request metrics, host-level counters | ✅ Shipped | `core/knowledge/http_metrics.py`, `http_throttle_report.py` |
| **Slice 2** | OpenAlex + NCBI keys in adapters | ✅ Shipped | `credential_resolver.py`, adapter modules |
| **Slice 3** | Per-host token bucket / interval scheduler | ✅ Shipped | `core/knowledge/host_scheduler.py` |
| **Slice 4** | Header-aware retries (`Retry-After`, OpenAlex budget) | ✅ Shipped | `core/knowledge/http_client.py` |
| **Slice 5** | Evidence cache TTL + negative cache for exhausted hosts | ✅ Shipped | `evidence_cache.py`, `negative_cache.py` |
| **Slice 6** | Per-host circuit breaker | ✅ Shipped | `host_scheduler.py` |
| **Slice 7 (eval)** | Inter-query pacing in live eval | ⚠️ Partial | `tools/evaluate_retrieval.py` (2s delay + retry on `no_results`) |
| **Slice 8 (tiered fan-out)** | Primary-first scientific adapter phases | 🔧 Opt-in | `tiered_scientific_retrieval.py` (`QUBE_TIERED_SCIENTIFIC_RETRIEVAL=1`) |
| **Slice 19 (query-type routing)** | Institutional-first routing for guideline/statistics/standard intents | ✅ Shipped | `scientific_query_type.py` (on by default; `QUBE_QUERY_TYPE_ROUTING=0` to disable) |
| **Slice 9** | Credential store + resolver (env → user settings → anonymous) | ✅ Shipped | `credentials.py`, `app_settings.KEY_KNOWLEDGE_PROVIDER_CREDENTIALS` |
| **Slice 10** | Settings → Provider credentials UI | ✅ Shipped | `ui/views/settings/sections/knowledge_provider_credentials.py` |
| **Slice 11** | Source status panel + quota notifications | ✅ Shipped | `provider_status.py`, `provider_limit_events.py`, `knowledge_provider_status.py` |
| **Adapter Slice 7a** | Crossref, Semantic Scholar, NASA ADS | ✅ Shipped | `crossref.py`, `semantic_scholar.py`, `nasa_ads.py` |
| **Adapter Slice 7b** | SocArXiv, Europe PMC, FRED | ✅ Shipped | `socarxiv.py`, `europe_pmc.py`, `fred.py` |
| **Adapter Slice 7c** | Companies House, Alpha Vantage | ✅ Shipped | `companies_house.py`, `alpha_vantage.py` |
| **Adapter Slice 8** | EUR-Lex, CanLII, BAILII | ✅ Shipped | `eur_lex.py`, `canlii.py`, `bailii.py` |
| **Adapter Slice 9–10** | SSRN, NOAA, PsyArXiv, NASA Earthdata + credential wiring | ✅ Shipped | `ssrn.py`, `noaa.py`, `psyarxiv.py`, `nasa_earthdata.py` |
| **Adapter Slice 11** | ACM DL, PsycINFO, Bloomberg Open API | ✅ Shipped | `acm_dl.py`, `psycinfo.py`, `bloomberg_api.py` |
| **Adapter Slice 12 (P0)** | ClinicalTrials.gov, openFDA, World Bank, Eurostat, USGS, USDA FDC, NIST NVD, IETF RFCs, BLS, U.S. Census, IEEE Xplore | ✅ Shipped | See [P0 institutional adapters](#p0-institutional-adapters-slice-12) |
| **Adapter Slice 13** | OECD, NICE, CDC, WHO GHO | ✅ Shipped | See [Health guidelines & OECD (Slice 13)](#health-guidelines--oecd-slice-13) |
| **Adapter Slice 14** | IPCC, FAO, USDA ERS, Copernicus CDS | ✅ Shipped | See [Geoscience & agriculture (Slice 14)](#geoscience--agriculture-slice-14) |
| **Adapter Slice 15** | OpenReview, ACL Anthology | ✅ Shipped | See [CS / AI venues (Slice 15)](#cs--ai-venues-slice-15) |
| **Adapter Slice 16** | ChEMBL, UniProt, PDB, ChemRxiv | ✅ Shipped | See [Molecular biology & chemistry (Slice 16)](#molecular-biology--chemistry-slice-16) |
| **Adapter Slice 17** | Congress.gov, GovInfo, legislation.gov.uk | ✅ Shipped | See [Law & legislation (Slice 17)](#law--legislation-slice-17) |
| **Adapter Slice 18** | USPTO PatentsView, EPO Espacenet | ✅ Shipped | See [Patents (Slice 18)](#patents-slice-18) |
| **Adapter Slice 19** | Query-type routing (platform) | ✅ Shipped | See [Query-type routing (Slice 19)](#query-type-routing-slice-19) |

All live adapters call through `core/knowledge/http_client.py` (instrumented `requests` wrapper) unless running in fixture mode.

---

## Settings → Knowledge (UI map)

The Knowledge settings page (`ui/views/settings/sections/knowledge.py`) is ordered as follows:

| Subsection | Anchor | Purpose |
|------------|--------|---------|
| Library search phrases | `triggers` | NLP trigger phrases for RAG / knowledge routing |
| Search quality | `embedding_mode` | Fast / Balanced / Power embedding presets |
| External knowledge toggles | — | External v2 pipeline, internal corpus, research map, deep research |
| **Provider credentials** | `knowledge_provider_credentials` | One card per provider id: masked key field, Test connection, Get free key, Clear saved key |
| **Source status** | `knowledge_provider_status` | Read-only table: Provider · Status · Quota · Health (refreshes every 60s while Settings is open) |
| **Preferred sources** | `sources_*` | Per-domain adapter enable/disable checkboxes |
| Prepare search models | — | Bootstrap ONNX search presets |
| Advanced embedding | `embedding_model` | Optional custom `.gguf` override (gated) |

**Deep links:** quota-limit notifications open Settings → Knowledge at anchor `knowledge_provider_credentials` (`MainWindow._open_settings_section`).

---

## How to read access modes

| Mode in this doc | Meaning in Qube |
|------------------|-----------------|
| **Anonymous** | Live retrieval works with no API key or account token configured. |
| **Optional key** | Anonymous access works; a **free** API key (Settings → Provider credentials or env var) raises rate limits or daily budgets. |
| **Key required** | Adapter skips live HTTP when no key is configured; configure in **Provider credentials**. |
| **HTML search** | No REST API; single-query HTML search with conservative rate limits (BAILII). |

**Default enabled** = included in out-of-box Preferred Sources for that domain. Adapters marked **Off by default** are opt-in.

**Provider credentials UI** only lists providers that have at least one live adapter (`list_active_provider_credential_specs()`).

---

## Readiness levels

Canonical metadata lives in `core/knowledge/adapter_readiness.py` (synced with `catalog.py`).

| Readiness | Meaning |
|-----------|---------|
| **stub** | Catalog placeholder only — not registered in `SEARCH_FUNCTIONS`. |
| **beta** | Live retrieval, but opt-in (off by default), key/enterprise required, indirect index (e.g. OpenAlex filter), or HTML scrape. Shown as **(beta)** in Preferred Sources. |
| **production** | Stable default-path adapter — on by default or anonymous primary API. |

Discipline packs are validated against this registry via `python3 tools/sync_discipline_packs.py --check` (`core/knowledge/discipline_pack_sync.py`).

---

## Scientific literature (`scientific_evidence`)

Scientific adapters are grouped in Settings by **UI subdomain**. The same adapter can appear in multiple groups (e.g. PubMed in Science, Biology, Chemistry, and Psychology). **Discipline packs** (`core/knowledge/scientific_discipline_packs.py`) pick primary/fallback adapters when `@evidence` routing detects a scholarly discipline.

### UI subdomains and typical adapters

| UI subdomain | Discipline pack(s) | Primary adapters (routing) | Other live adapters in group |
|--------------|-------------------|----------------------------|------------------------------|
| **Science** (general) | Medicine, Physics, General science | `pubmed`, `arxiv`, `openalex`, `inspire_hep` | `crossref`, `semantic_scholar`, `europe_pmc`, `nasa_ads` |
| **Earth & Environment** | Earth & Environment | `openalex` | `noaa`, `nasa_earthdata`, `arxiv` |
| **Biology** | Biology | `pubmed`, `biorxiv` | `uniprot`, `pdb`, `openalex`, `europe_pmc` |
| **Chemistry** | Chemistry | `pubchem` | `chembl`, `chemrxiv`, `uspto_patentsview`, `epo_espacenet`, `openalex`, `pubmed` |
| **Computer Science** | Computer science | `arxiv`, `dblp` | `openreview`, `acl_anthology`, `openalex`, `acm_dl` |
| **Economics** | Economics | `repec` | `openalex`, `ssrn` |
| **Psychology** | Psychology | `pubmed` | `openalex`, `psyarxiv`, `psycinfo` |
| **Social Science** | Sociology, Political science | `openalex` | `socarxiv`, `ssrn`, `world_bank`, `eurostat`, `us_census` |
| **Engineering** | Engineering | `ieee_xplore`, `nist`, `ietf_rfc` | `uspto_patentsview`, `epo_espacenet`, `arxiv`, `openalex` |
| **Agriculture & Nutrition** | — | `usda_fdc` | — |

### All live scientific adapters (46)

| Adapter | Readiness | Production strategy | Access | Default |
|---------|-----------|---------------------|--------|---------|
| `openalex` | production | OpenAlex REST works search (optional free API key). | Optional key | On |
| `pubmed` | production | NCBI E-utilities direct search (optional NCBI API key). | Optional key | On |
| `pubchem` | production | PubChem PUG REST compound search (optional NCBI key). | Optional key | On |
| `crossref` | production | Crossref REST works metadata (polite pool, anonymous). | Anonymous | On |
| `semantic_scholar` | beta | Semantic Scholar Graph API (free API key required). | Key required | On |
| `europe_pmc` | production | Europe PMC REST search (anonymous). | Anonymous | On |
| `arxiv` | production | arXiv Atom API (anonymous). | Anonymous | On |
| `biorxiv` | production | bioRxiv preprints via Europe PMC filter (anonymous). | Anonymous | On |
| `inspire_hep` | production | INSPIRE-HEP REST literature search (anonymous). | Anonymous | On |
| `nasa_ads` | beta | NASA ADS REST search (personal API token required). | Key required | Off |
| `socarxiv` | production | SocArXiv preprints via OSF API (anonymous). | Anonymous | On |
| `ssrn` | beta | SSRN works via OpenAlex source filter (optional OpenAlex key). | Anonymous | Off |
| `psyarxiv` | beta | PsyArXiv preprints via OSF API (anonymous, opt-in). | Anonymous | Off |
| `noaa` | beta | NOAA NCEI CDO datasets API (token required). | Key required | Off |
| `nasa_earthdata` | beta | NASA Earthdata CMR collections JSON search (anonymous, opt-in). | Anonymous | Off |
| `dblp` | production | DBLP publication search API (anonymous). | Anonymous | On |
| `acm_dl` | beta | ACM works via OpenAlex publisher filter (optional OpenAlex key). | Optional key | Off |
| `psycinfo` | beta | PsycINFO via institutional EBSCO EDS API (credentials required). | Key required | Off |
| `repec` | production | RePEc/IDEAS metadata via EconBiz API (anonymous). | Anonymous | On |
| `clinicaltrials_gov` | production | ClinicalTrials.gov REST API v2 (anonymous). | Anonymous | On |
| `openfda` | production | openFDA drug label search (anonymous). | Anonymous | On |
| `world_bank` | production | World Bank Open Data indicator catalog (anonymous). | Anonymous | On |
| `eurostat` | production | Eurostat discovery statistics search (anonymous). | Anonymous | On |
| `usgs` | production | USGS Publications Service search (anonymous). | Anonymous | On |
| `usda_fdc` | production | USDA FoodData Central REST search (optional free key). | Optional key | On |
| `nist` | production | NIST NVD keyword search (optional free API key). | Optional key | On |
| `ietf_rfc` | production | IETF Datatracker RFC search (anonymous). | Anonymous | On |
| `bls` | production | BLS series search (free registration key required). | Key required | On |
| `us_census` | production | U.S. Census data.json catalog search (optional free key). | Optional key | On |
| `ieee_xplore` | beta | IEEE Xplore Metadata API (free developer key required). | Key required | On |
| `oecd` | production | OECD SDMX dataflow catalog keyword search (anonymous). | Anonymous | On |
| `nice` | beta | NICE syndication guidance index (syndication API key required). | Key required | On |
| `cdc` | production | CDC Content Services media + Open Data catalog (anonymous). | Anonymous | On |
| `who` | production | WHO GHO OData indicator search (anonymous). | Anonymous | On |
| `ipcc` | production | IPCC-related Zenodo record discovery (anonymous). | Anonymous | On |
| `fao` | beta | FAOSTAT dataset catalog (FAOSTAT API bearer token required). | Key required | On |
| `usda` | production | USDA ERS ARMS variable search (optional api.data.gov key). | Optional key | On |
| `copernicus_cds` | production | Copernicus CDS STAC catalogue search (anonymous). | Optional key | On |
| `openreview` | production | OpenReview notes search API (anonymous). | Anonymous | On |
| `acl_anthology` | beta | ACL Anthology metadata via Verbatim search (anonymous). | Anonymous | On |
| `chembl` | production | ChEMBL molecule search REST API (anonymous). | Anonymous | On |
| `uniprot` | production | UniProtKB REST search API (anonymous). | Anonymous | On |
| `pdb` | production | RCSB PDB Search API + core entry metadata (anonymous). | Anonymous | On |
| `chemrxiv` | production | ChemRxiv preprints via Europe PMC DOI prefix filter (anonymous). | Anonymous | On |
| `uspto_patentsview` | beta | U.S. granted patent search via PatentsView PatentSearch API (API key required). | Key required | On |
| `epo_espacenet` | beta | Worldwide patent bibliographic search via EPO OPS (consumer credentials required). | Key required | On |

### P0 institutional adapters (Slice 12)

Shipped July 2026 as the first **institutional-source** expansion: health agencies, official statistics, geoscience, nutrition, and engineering standards. These adapters emit `document_type` values such as `clinical_trial`, `regulatory_label`, `statistical_indicator`, `government_publication`, `nutrition_dataset`, `standard_document`, and `standard_reference` (see `bundle_builder.py`).

> **Medicine routing** now prefers `nice`, `cdc`, `who`, `clinicaltrials_gov`, and `openfda` in discipline-pack fallbacks for biomedical queries.

> **Economics routing** adds `world_bank`, `eurostat`, `oecd`, and `bls` as institutional fallbacks after RePEc.

> **Engineering** is a new discipline pack (`engineering`) with primary adapters `ieee_xplore`, `nist`, `ietf_rfc`.

> **ACM DL** and **SSRN** reuse the optional OpenAlex key for higher search budgets when configured.

> **NCBI provider row** in Provider credentials covers both **PubMed** and **PubChem** (`provider_id: ncbi`).

### Health guidelines & OECD (Slice 13)

Shipped July 2026 as the **remaining P0 health-agency and OECD statistics** slice. These adapters emit `document_type` values such as `clinical_guideline`, `health_guidance`, `health_indicator`, and `statistical_release`.

| Adapter | API | Access |
|---------|-----|--------|
| `nice` | NICE syndication guidance index (`api.nice.org.uk`) | Syndication licence + `QUBE_NICE_API_KEY` |
| `cdc` | CDC Content Services v2 + data.cdc.gov catalog | Anonymous |
| `who` | WHO GHO OData (`ghoapi.azureedge.net`) | Anonymous |
| `oecd` | OECD SDMX dataflow catalog (`sdmx.oecd.org`) | Anonymous |

### Geoscience & agriculture (Slice 14)

Shipped July 2026 for **climate assessment archives** and **agriculture institutional data**. Document types include `assessment_report`, `climate_dataset`, `agricultural_dataset`, and `agricultural_indicator`.

| Adapter | API | Access |
|---------|-----|--------|
| `ipcc` | Zenodo IPCC-related record search | Anonymous |
| `copernicus_cds` | Copernicus CDS STAC catalogue (`cds.climate.copernicus.eu`) | Anonymous search; optional `QUBE_COPERNICUS_CDS_API_KEY` for downloads |
| `fao` | FAOSTAT dataset catalog (`faostatservices.fao.org`) | `QUBE_FAO_API_KEY` (JWT bearer) |
| `usda` | USDA ERS ARMS variable API (`api.ers.usda.gov`) | Optional `QUBE_USDA_API_KEY` (api.data.gov); `DEMO_KEY` fallback |

> **Earth & Environment routing** adds `ipcc` and `copernicus_cds` to discipline-pack fallbacks alongside `usgs`, `noaa`, and `nasa_earthdata`.

### CS / AI venues (Slice 15)

Shipped July 2026 for **machine-learning conference submissions** and **NLP proceedings**. Document types include `conference_paper` and `preprint`.

| Adapter | API | Access |
|---------|-----|--------|
| `openreview` | OpenReview notes search (`api2.openreview.net`) | Anonymous |
| `acl_anthology` | Verbatim ACL Anthology search (`verbatim.krlabs.eu`) | Anonymous |

> **Computer science routing** adds `openreview` and `acl_anthology` to discipline-pack fallbacks before OpenAlex and ACM DL.

> ACL Anthology has no official REST search API; Qube uses the third-party Verbatim metadata index that mirrors ACL proceedings metadata and PDF links.

### Molecular biology & chemistry (Slice 16)

Shipped July 2026 for **protein sequence/structure databases** and **medicinal chemistry**. Document types include `bioactive_compound`, `protein_record`, `protein_structure`, and `preprint`.

| Adapter | API | Access |
|---------|-----|--------|
| `chembl` | ChEMBL molecule search (`www.ebi.ac.uk/chembl`) | Anonymous |
| `uniprot` | UniProtKB REST search (`rest.uniprot.org`) | Anonymous |
| `pdb` | RCSB PDB Search API + core entry metadata | Anonymous |
| `chemrxiv` | Europe PMC with ChemRxiv DOI prefix (`10.26434*`) | Anonymous |

> **Biology routing** adds `uniprot` and `pdb` to discipline-pack fallbacks. **Chemistry routing** adds `chembl` and `chemrxiv`.

> ChemRxiv’s OSF preprints endpoint no longer exposes text search; Qube mirrors the bioRxiv pattern using Europe PMC’s ChemRxiv DOI prefix filter.

### Patents (Slice 18)

Shipped July 2026 for **U.S. and European patent bibliographic search**. Document type: `patent`.

| Adapter | API | Access |
|---------|-----|--------|
| `uspto_patentsview` | PatentsView PatentSearch (`search.patentsview.org`) | `QUBE_PATENTSVIEW_API_KEY` |
| `epo_espacenet` | EPO Open Patent Services (`ops.epo.org`) | `QUBE_EPO_OPS_CONSUMER_KEY` + `QUBE_EPO_OPS_CONSUMER_SECRET` (or `key:secret` in settings) |

> **Engineering routing** adds `uspto_patentsview` and `epo_espacenet` to discipline-pack fallbacks before arXiv and OpenAlex.

> **Chemistry routing** adds the same patent adapters after ChEMBL and ChemRxiv.

### Query-type routing (Slice 19)

Shipped July 2026 as **institutional-first adapter ordering** within `scientific_evidence`. After discipline-pack routing, queries are classified into intent buckets and enabled institutional adapters are moved ahead of bibliographic indexes.

| Query type | Example phrasing | Boosted adapters (when enabled) |
|------------|------------------|--------------------------------|
| `guideline` | “How is hypertension treated?” | `nice`, `cdc`, `who`, `openfda`, `clinicaltrials_gov` |
| `statistics` | “What is the current US unemployment rate?” | `bls`, `world_bank`, `eurostat`, `oecd`, `us_census` |
| `standard` | “What does RFC 8446 specify?” | `ietf_rfc`, `nist`, `ieee_xplore` |
| `clinical_trial` | “Phase 3 clinical trial recruiting patients” | `clinicaltrials_gov`, `openfda`, `pubmed` |
| `dataset` | “NOAA sea surface temperature dataset” | `ipcc`, `copernicus_cds`, `noaa`, `usgs`, … |
| `patent` | “lithium battery electrode patent search” | `uspto_patentsview`, `epo_espacenet` |
| `literature` | Default — research papers, meta-analyses | No reorder (discipline pack only) |

> Enabled by default. Set `QUBE_QUERY_TYPE_ROUTING=0` to restore discipline-only ordering. Retrieval traces expose `scientific_query_type_routing` in `relevance_diag`.

## Finance (`finance_knowledge`)

Finance is a single top-level domain (no subdomains). All finance bundles include the **`not_financial_advice`** warning.

| Adapter | Readiness | Production strategy | Access | Default |
|---------|-----------|---------------------|--------|---------|
| `sec_edgar` | production | SEC EDGAR submissions JSON (anonymous). | Anonymous | On |
| `fred` | production | FRED series search API (free API key required). | Key required | On |
| `world_bank` | production | World Bank Open Data indicator catalog (anonymous). | Anonymous | On |
| `eurostat` | production | Eurostat discovery statistics search (anonymous). | Anonymous | On |
| `bls` | production | BLS Public Data API series search (free registration key required). | Key required | On |
| `oecd` | production | OECD SDMX dataflow catalog keyword search (anonymous). | Anonymous | On |
| `companies_house` | beta | UK Companies House REST search (free API key required). | Key required | Off |
| `alpha_vantage` | beta | Alpha Vantage SYMBOL_SEARCH (free API key required). | Key required | Off |
| `bloomberg_api` | beta | Bloomberg Open API via local HTTP bridge (enterprise URL required). | Key required | Off |

---

## Legal (`legal_knowledge`)

Legal is a single top-level domain (no subdomains). All legal bundles include the **`not_legal_advice`** warning.

| Adapter | Readiness | Production strategy | Access | Default |
|---------|-----------|---------------------|--------|---------|
| `courtlistener` | production | CourtListener v4 REST search (optional free account token). | Optional key | On |
| `congress_gov` | beta | Congress.gov bill metadata search (api.data.gov key required). | Key required | On |
| `govinfo` | beta | GovInfo federal publication search (api.data.gov key required). | Key required | On |
| `legislation_uk` | production | UK legislation title search via Atom feed (anonymous). | Anonymous | On |
| `eur_lex` | beta | EUR-Lex CELLAR SPARQL legal-act search (anonymous, opt-in). | Anonymous | Off |
| `canlii` | beta | CanLII REST case search (free API key required). | Key required | Off |
| `bailii` | beta | BAILII HTML search (no official API; respectful scrape). | HTML search | Off |

### Law & legislation (Slice 17)

Shipped July 2026 for **primary U.S. and UK legislation**. Document types include `federal_bill`, `federal_statute`, `federal_publication`, and `uk_legislation`.

| Adapter | API | Access |
|---------|-----|--------|
| `congress_gov` | Congress.gov API v3 (`api.congress.gov`) | `QUBE_CONGRESS_GOV_API_KEY` (api.data.gov) |
| `govinfo` | GovInfo Search API (`api.govinfo.gov`) | `QUBE_GOVINFO_API_KEY` (api.data.gov) |
| `legislation_uk` | legislation.gov.uk Atom search feed | Anonymous |

> Congress.gov has no full-text search endpoint; Qube matches bill citations directly or filters recent bill titles client-side.

> GovInfo and Congress.gov share the same **api.data.gov** key family — one key may work for both providers.

---

## Planned expansion roadmap (external review, July 2026)

**Status:** 📋 Backlog — items marked ✅ **Shipped (Slice 12–14)** are live; remaining rows are planned work.

An external architecture review rated Qube’s current stack as strong on **primary literature indexes** (PubMed, arXiv, OpenAlex, CourtListener, SEC EDGAR, etc.) but identified a strategic gap: **authoritative institutional knowledge** rather than more bibliographic databases.

> **Design thesis:** Many factual questions (“How is hypertension treated?”, “What is the current US inflation rate?”, “What does USB-C specify?”) are better answered by **agency guidelines, official statistics, standards bodies, and curated institutional corpora** than by retrieving another batch of papers. The backlog below prioritizes those source classes while preserving literature adapters for research-oriented `@evidence` turns.

### Coverage assessment (current stack, external review)

Baseline ratings for the **live** adapter set — useful for prioritizing backlog work:

| Domain | Coverage (review) | Primary gap |
|--------|-------------------|-------------|
| Medicine | 9.5/10 | Clinical guidelines & agency consensus (WHO, CDC, NICE, …) |
| Biology | 9/10 | Molecular entity databases (UniProt, Ensembl, PDB) |
| Chemistry | 8.5/10 | Medicinal chemistry & preprints (ChEMBL, ChemRxiv) |
| Physics | 9.5/10 | — |
| Computer Science | 8.5/10 | ML venue preprints (OpenReview), NLP anthology |
| Economics | 8.5/10 | Official macro/stat releases (OECD, BLS, BEA, …) |
| Psychology | 8.5/10 | — |
| Social Sciences | 7/10 | Policy statistics & datasets (OECD, ICPSR, World Bank) |
| Earth Science | 8/10 | Institutional climate/geoscience (IPCC summaries, USGS, Copernicus) |
| Finance | 8/10 | Structured filings & macro identifiers (SEC XBRL, OpenFIGI) |
| Law | 7.5/10 | Primary legislation & IP (Congress.gov, govinfo, WIPO Lex) |
| **Engineering** | *(not scored)* | **Largest disciplinary gap** — IEEE, NIST, standards |

### Priority tiers

| Tier | Focus | Rationale |
|------|-------|-----------|
| **P0** | Government & health agencies; official statistics; engineering (IEEE, NIST); clinical trials | Highest user value for “current consensus” and factual questions |
| **P1** | Standards (IETF RFCs, W3C); legislation; patents; molecular biology; chemistry (ChEMBL); AI venues (OpenReview, ACL Anthology) | High authority, often API-accessible or metadata-friendly |
| **P2** | Licensed/commercial indexes; metadata-only integrations; niche geoscience/education | Valuable but blocked or degraded by licensing, access agreements, or scrape risk |

### Architectural work (cross-cutting, before many backlog items)

These platform changes are **prerequisites** for institutional sources, not individual adapters:

| Work item | Why |
|-----------|-----|
| **Source-type routing** | Distinguish *literature* vs *guideline* vs *statistics* vs *standard* query intent; route to institutional adapters first when appropriate — **✅ Shipped (Slice 19)** |
| **`trusted_knowledge` / new service boundaries** | Some backlog sources may fit `@trusted` or a future `@official` / `@statistics` composer path better than `@evidence` |
| **Document-type metadata in bundles** | Tag evidence as `guideline`, `statistics_release`, `standard`, `dataset`, `patent` — not only `journal_article` |
| **Metadata-only / abstract adapters** | JSTOR, Cochrane, MathSciNet, Engineering Village — retrieve citation + link when full text is licensed |
| **Geographic / jurisdiction routing** | NHS/NICE (UK), legislation.gov.uk, CanLII overlap — pick by user locale or explicit `@legal` jurisdiction hints |

---

### P0 — Government, health & institutional consensus

#### Health & clinical guidelines

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [WHO](https://www.who.int/) | `who` | GHO OData indicator search — **✅ Shipped (Slice 13)** | Medicine, `@trusted` |
| [CDC](https://www.cdc.gov/) | `cdc` | Content Services + Open Data catalog — **✅ Shipped (Slice 13)** | Medicine, infectious disease |
| [NIH](https://www.nih.gov/) | `nih` | Multiple APIs (e.g. RePORTER exists separately) | Medicine |
| [FDA](https://www.fda.gov/) | `fda` | openFDA API (drugs, devices, recalls) — partial coverage | Medicine, chemistry |
| [EMA](https://www.ema.europa.eu/) | `ema` | SPARQL / document search TBD | Medicine (EU) |
| [NHS](https://www.nhs.uk/) | `nhs` | No official bulk API; respectful HTML or syndication | Medicine (UK) |
| [NICE](https://www.nice.org.uk/) | `nice` | NICE syndication API (guidelines) — **✅ Shipped (Slice 13)** | Medicine (UK) |

**Example use case:** “How is hypertension treated?” → NICE/WHO guideline adapter preferred over PubMed paper flood.

#### Infectious disease surveillance

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [ECDC](https://www.ecdc.europa.eu/) | `ecdc` | Outbreak data / reports | Medicine |
| WHO disease outbreak reports | `who_outbreaks` | Sub-feed of WHO programmatic data | Medicine |

#### Climate, Earth & geoscience (institutional)

Complements live `noaa`, `nasa_earthdata` with **consensus-oriented** sources:

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| NASA Climate | `nasa_climate` | NASA GISS / climate portal APIs | Earth & Environment |
| [USGS](https://www.usgs.gov/) | `usgs` | USGS Publications API — **✅ Shipped (Slice 12)** | Earth & Environment, Geoscience |
| [IPCC](https://www.ipcc.ch/) | `ipcc` | Zenodo IPCC-related record search — **✅ Shipped (Slice 14)** | Earth & Environment |
| [ESA Earth Observation](https://www.esa.int/) | `esa_earth` | ESA catalogs / Copernicus links | Earth & Environment |
| [Copernicus Climate Data Store](https://climate.copernicus.eu/) | `copernicus_cds` | CDS STAC catalogue API — **✅ Shipped (Slice 14)** | Earth & Environment |
| [British Geological Survey](https://www.bgs.ac.uk/) | `bgs` | UK geoscience data | Geoscience |
| [OneGeology](https://www.onegeology.org/) | `onegeology` | Aggregated geological map metadata | Geoscience |

#### Agriculture & food / nutrition

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [USDA](https://www.usda.gov/) | `usda` | ERS ARMS variable API (optional api.data.gov key) — **✅ Shipped (Slice 14)** | Agriculture, nutrition |
| [USDA FoodData Central](https://fdc.nal.usda.gov/) | `usda_fdc` | Free REST API — **✅ Shipped (Slice 12)** | Nutrition |
| [FAO](https://www.fao.org/) | `fao` | FAOSTAT dataset catalog (JWT bearer) — **✅ Shipped (Slice 14)** | Agriculture |
| [EFSA](https://www.efsa.europa.eu/) | `efsa` | Scientific opinions & data | Food safety, nutrition |
| NIH Office of Dietary Supplements | `ods_nih` | Fact sheets (HTML / structured TBD) | Nutrition |

#### Astronomy & space (institutional, beyond literature)

Complements live `nasa_ads`, `inspire_hep`:

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [ESA](https://www.esa.int/) | `esa` | Mission catalogs, news, science summaries | Physics, astronomy |
| [NASA JPL](https://www.jpl.nasa.gov/) | `nasa_jpl` | Mission pages / Horizons (ephemerides) TBD | Astronomy |
| [Minor Planet Center](https://minorplanetcenter.net/) | `mpc` | Asteroid/comet catalogs | Astronomy |
| [IAU](https://www.iau.org/) | `iau` | Nomenclature, standards | Astronomy |

#### Official statistics (cross-domain, P0)

Many factual questions map directly to statistical releases:

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [OECD](https://www.oecd.org/) | `oecd` | OECD SDMX dataflow catalog — **✅ Shipped (Slice 13)** | Economics, social science, finance |
| [World Bank Open Data](https://data.worldbank.org/) | `world_bank` | Free API — **✅ Shipped (Slice 12)** | Economics, social science, finance |
| [UN Data](https://data.un.org/) | `un_data` | UN SDG / stat APIs | Social science, statistics |
| [Eurostat](https://ec.europa.eu/eurostat) | `eurostat` | REST discovery — **✅ Shipped (Slice 12)** | Statistics, economics |
| [US Census Bureau](https://www.census.gov/data.html) | `us_census` | Census API (optional key) — **✅ Shipped (Slice 12)** | Statistics, social science |
| [Statistics Canada](https://www.statcan.gc.ca/) | `statcan` | Web Data Service | Statistics |
| [UK ONS](https://www.ons.gov.uk/) | `ons` | ONS API | Statistics, economics (UK) |
| [BLS](https://www.bls.gov/) | `bls` | BLS Public Data API (key required) — **✅ Shipped (Slice 12)** | Economics (US) |

#### Biomedical — trials & evidence synthesis

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [ClinicalTrials.gov](https://clinicaltrials.gov/) | `clinicaltrials_gov` | Free REST API v2 — **✅ Shipped (Slice 12)** | Medicine, biology |
| [FDA openFDA](https://open.fda.gov/) | `openfda` | Drug label API — **✅ Shipped (Slice 12)** | Medicine, chemistry |
| [Cochrane Library](https://www.cochranelibrary.com/) | `cochrane` | **Metadata only** if licensing limits full text | Medicine |
| [GISAID](https://www.gisaid.org/) | `gisaid` | **Access agreement required** — likely P2 / opt-in | Infectious disease |

---

### P0 — Engineering & applied physical sciences

Largest **disciplinary gap** in the current catalog (arXiv + DBLP + ACM only):

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| **IEEE Xplore** | `ieee_xplore` | API key required — **✅ Shipped (Slice 12)** | Engineering (new UI group) |
| [NIST](https://www.nist.gov/) | `nist` | NVD REST API (optional key) — **✅ Shipped (Slice 12)** | Engineering, standards, chemistry |
| [IETF RFCs](https://www.rfc-editor.org/) | `ietf_rfc` | Datatracker API — **✅ Shipped (Slice 12)** | Engineering, CS |
| Engineering Village | `engineering_village` | **Commercial / Elsevier** — metadata-only or P2 | Engineering |

**New Settings UI group (proposed):** **Engineering** — discipline pack + primary `ieee_xplore` / `nist` / `arxiv` routing.

---

### P1 — Discipline-specific literature & metadata gaps

#### Mathematics

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [zbMATH Open](https://zbmath.org/) | `zbmath` | Free API | Mathematics (new UI group) |
| MathSciNet | `mathscinet` | **Licensed (AMS)** — metadata-only | Mathematics |

#### Computer science & AI

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [OpenReview](https://openreview.net/) | `openreview` | Public API — ICLR / NeurIPS / ICML submissions — **✅ Shipped (Slice 15)** | Computer science |
| [ACL Anthology](https://aclanthology.org/) | `acl_anthology` | Verbatim metadata search (no official REST API) — **✅ Shipped (Slice 15)** | Computer science, NLP |
| [Hugging Face Papers](https://huggingface.co/papers) | `hf_papers` | Metadata API / scrape policy TBD | Computer science, AI |
| [Papers with Code](https://paperswithcode.com/) | `papers_with_code` | Metadata (not primary literature) | Computer science |

#### Chemistry

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | `chembl` | Free REST API (EMBL-EBI) — **✅ Shipped (Slice 16)** | Chemistry, medicine |
| [ChemRxiv](https://chemrxiv.org/) | `chemrxiv` | Europe PMC ChemRxiv DOI filter — **✅ Shipped (Slice 16)** | Chemistry |

#### Biology (molecular)

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [UniProt](https://www.uniprot.org/) | `uniprot` | REST API — **✅ Shipped (Slice 16)** | Biology |
| NCBI Gene | `ncbi_gene` | E-utilities (shared NCBI key) | Biology |
| [Ensembl](https://www.ensembl.org/) | `ensembl` | REST API | Biology |
| [Protein Data Bank (PDB)](https://www.rcsb.org/) | `pdb` | RCSB PDB Search API — **✅ Shipped (Slice 16)** | Biology, chemistry |

#### Social sciences

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [JSTOR](https://www.jstor.org/) | `jstor` | **Metadata only** if licensing prevents full text | Social science |
| [ICPSR](https://www.icpsr.umich.edu/) | `icpsr` | Study metadata; download gated | Social science |

#### Economics (official releases, beyond RePEc papers)

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [IMF Data](https://www.imf.org/en/Data) | `imf` | IMF API | Economics, finance |
| [BIS](https://www.bis.org/) | `bis` | Statistics / research data | Economics, finance |
| [ECB](https://www.ecb.europa.eu/) | `ecb` | SDW / statistical data warehouse | Economics, finance |
| [BEA](https://www.bea.gov/) | `bea` | BEA API | Economics (US) |
| [BLS](https://www.bls.gov/) | `bls` | BLS Public Data API | Economics (US) |

*Note: OECD, World Bank listed under P0 statistics — shared adapters serve both economics and social-science routing.*

#### Finance (structured data & macro, beyond live stack)

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| SEC XBRL / structured filings | `sec_xbrl` | Extends `sec_edgar` — facts & taxonomy | Finance |
| [OpenFIGI](https://www.openfigi.org/) | `openfigi` | Free API (Bloomberg) — identifier resolution | Finance |
| Federal Reserve (beyond FRED) | `fed_h41`, `fed_speeches` | Multiple Fed APIs / RSS | Finance |
| IMF / World Bank / OECD | *(see P0)* | Macro & development finance series | Finance |

#### Law (primary legislation & specialized courts)

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [Congress.gov](https://www.congress.gov/) | `congress_gov` | Congress.gov API — **✅ Shipped (Slice 17)** | Legal (US) |
| [govinfo](https://www.govinfo.gov/) | `govinfo` | GovInfo Search API — **✅ Shipped (Slice 17)** | Legal (US) |
| [legislation.gov.uk](https://www.legislation.gov.uk/) | `legislation_uk` | Atom search feed — **✅ Shipped (Slice 17)** | Legal (UK) |
| [HUDOC](https://hudoc.echr.coe.int/) | `hudoc` | ECtHR case law API | Legal (EU human rights) |
| [WIPO Lex](https://wipolex.wipo.int/) | `wipo_lex` | IP laws & treaties database | Legal (IP) |

#### Standards & specifications

Surprisingly high value for “what is X?” factual questions:

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [IETF RFCs](https://www.rfc-editor.org/) | `ietf_rfc` | RFC index + datatracker API — **especially high ROI** | Engineering, CS |
| [W3C](https://www.w3.org/) | `w3c` | TR / standards catalog | Engineering, CS |
| ISO | `iso` | **Licensed** — metadata / purchase links only | Standards |
| IEC | `iec` | **Licensed** — metadata only | Standards |

*NIST standards cross-listed under Engineering.*

#### Patents

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [Google Patents](https://patents.google.com/) | `google_patents` | No official public API — scrape risk / P2 | Engineering, chemistry |
| [USPTO PatentsView](https://patentsview.org/) | `uspto_patentsview` | Free API — **✅ Shipped (Slice 18)** | Engineering, chemistry |
| [EPO Espacenet](https://worldwide.espacenet.com/) | `epo_espacenet` | Open Patent Services (OPS) API — **✅ Shipped (Slice 18)** | Engineering, chemistry |

#### Education

| Planned source | Suggested adapter id | Access notes | Target routing |
|----------------|---------------------|--------------|----------------|
| [ERIC](https://eric.ed.gov/) | `eric` | ERIC API (IES) | Education (new UI group) |
| [UNESCO](https://www.unesco.org/) | `unesco` | Data & reports | Education, social science |

---

### P2 — Licensed, commercial, or high-friction integrations

Defer until P0/P1 institutional and open-API sources ship:

| Planned source | Blocker | Fallback strategy |
|----------------|---------|-------------------|
| Engineering Village (Elsevier) | Commercial license | Metadata + link-out |
| MathSciNet (AMS) | Subscription | Metadata + link-out |
| JSTOR full text | Institutional license | Metadata + link-out |
| Cochrane full reviews | Licensing | Metadata + abstract |
| ISO / IEC full standards text | Paywalled | Metadata + purchase link |
| Google Patents bulk | ToS / no stable API | PatentsView + EPO OPS first |
| GISAID | Access agreement | Opt-in credential + legal review |
| Bloomberg (live) | ✅ Shipped (`bloomberg_api`) | Enterprise bridge only |

---

### Suggested implementation slices (backlog)

Proposed ordering for future adapter work — update this table as slices ship:

| Slice | Theme | Status | Representative adapters |
|-------|-------|--------|-------------------------|
| **12** | P0 institutional (health, stats, engineering, geo, nutrition) | ✅ Shipped | `clinicaltrials_gov`, `openfda`, `world_bank`, `eurostat`, `usgs`, `usda_fdc`, `nist`, `ietf_rfc`, `bls`, `us_census`, `ieee_xplore` |
| **13** | Remaining P0 statistics & health guidelines | ✅ Shipped | `oecd`, `nice`, `cdc`, `who` |
| **14** | Remaining P0 geoscience & agriculture | ✅ Shipped | `ipcc`, `fao`, `usda`, `copernicus_cds` |
| **15** | CS / AI venues (P1) | ✅ Shipped | `openreview`, `acl_anthology` |
| **16** | Molecular biology & chemistry (P1) | ✅ Shipped | `chembl`, `uniprot`, `pdb`, `chemrxiv` |
| **17** | Law & legislation (P1) | ✅ Shipped | `congress_gov`, `govinfo`, `legislation_uk` |
| **18** | Patents (P1) | ✅ Shipped | `uspto_patentsview`, `epo_espacenet` |
| **19** | Platform: query-type routing | ✅ Shipped | Institutional-first routing (`scientific_query_type.py`) |

Each slice should follow the [Maintenance](#maintenance) checklist (catalog, registry, host policy, fixtures, eval corpus, discipline pack sync).

---

## Provider credentials summary

Active rows in **Settings → Knowledge → Provider credentials** (**25 providers**):

| Provider | Adapters | Anonymous? | Free key? | Key required? |
|----------|----------|------------|-----------|---------------|
| OpenAlex | `openalex` | Yes | Yes | No |
| NCBI | `pubmed`, `pubchem` | Yes | Yes | No |
| CourtListener | `courtlistener` | Yes | Yes | No |
| Semantic Scholar | `semantic_scholar` | No | Yes | **Yes** |
| NASA ADS | `nasa_ads` | No | Yes | **Yes** |
| FRED | `fred` | No | Yes | **Yes** |
| Companies House | `companies_house` | No | Yes | **Yes** |
| Alpha Vantage | `alpha_vantage` | No | Yes | **Yes** |
| CanLII | `canlii` | No | Yes | **Yes** |
| NOAA NCEI | `noaa` | No | Yes | **Yes** |
| EBSCO Discovery | `psycinfo` | No | No | **Yes** |
| Bloomberg Open API | `bloomberg_api` | No | No | **Yes** |
| USDA FoodData Central | `usda_fdc` | Yes | Yes | No |
| BLS | `bls` | No | Yes | **Yes** |
| U.S. Census Bureau | `us_census` | Yes | Yes | No |
| NIST NVD | `nist` | Yes | Yes | No |
| IEEE Xplore | `ieee_xplore` | No | Yes | **Yes** |
| NICE Syndication | `nice` | No | Yes | **Yes** |
| FAOSTAT | `fao` | No | Yes | **Yes** |
| USDA (api.data.gov) | `usda` | Yes | Yes | No |
| Copernicus CDS | `copernicus_cds` | Yes | Yes | No |
| Congress.gov | `congress_gov` | No | Yes | **Yes** |
| GovInfo | `govinfo` | No | Yes | **Yes** |
| PatentsView (USPTO) | `uspto_patentsview` | No | Yes | **Yes** |
| EPO Open Patent Services | `epo_espacenet` | No | Yes | **Yes** |

Adapters with **no Provider credentials row**:  
`crossref`, `europe_pmc`, `arxiv`, `biorxiv`, `inspire_hep`, `socarxiv`, `ssrn`, `psyarxiv`, `nasa_earthdata`, `dblp`, `acm_dl`, `repec`, `sec_edgar`, `eur_lex`, `bailii`, `legislation_uk`, `clinicaltrials_gov`, `openfda`, `world_bank`, `eurostat`, `usgs`, `ietf_rfc`, `oecd`, `cdc`, `who`, `ipcc`, `openreview`, `acl_anthology`, `chembl`, `uniprot`, `pdb`, `chemrxiv`.

### Resolution order

`core/knowledge/credentials.py` → `resolve_credential(provider_id)`:

1. **Fixture mode** — `QUBE_KNOWLEDGE_FIXTURES=1` → `CredentialMode.FIXTURE` (no live HTTP).
2. **Environment override** — provider `env_var` (and documented aliases, e.g. `FRED_API_KEY`) wins over saved settings; UI fields show “Using environment variable override”.
3. **User settings** — `qube.knowledge.provider_credentials` in QSettings (`KEY_KNOWLEDGE_PROVIDER_CREDENTIALS`).
4. **Anonymous** — default when no key is configured and the provider supports anonymous access.

User keys are stored locally only; adapters receive a `CredentialBundle` and never log the secret. **Test connection** probes live via `core/knowledge/provider_credential_test.py` and updates both credential row status labels and the Source status table.

See `core/knowledge/provider_credentials.py` for signup URLs, benefit copy, and per-provider test probes.

---

## Source status panel

**Settings → Knowledge → Source status** (`ui/views/settings/sections/knowledge_provider_status.py`) shows one row per provider with a live implemented adapter.

| Column | Source |
|--------|--------|
| **Provider** | `ProviderCredentialSpec.label` |
| **Status** | Connection mode from `resolve_credential()` (Anonymous / Connected with key / Not configured / …) |
| **Quota** | OpenAlex daily budget headers when available; otherwise scheduler policy label from `http_metrics` |
| **Health** | `Good` · `Degraded` (circuit open or recent test failure) · `Unknown` · `—` |

Aggregated in `core/knowledge/provider_status.py` from:

- `http_metrics.global_http_summary()` — last request times, error counts
- `host_scheduler.host_health_snapshot()` — circuit breaker state
- `record_provider_credential_test()` — last Test connection result
- OpenAlex rate-limit / budget headers when present

The panel **refreshes on load**, every **60 seconds** while Settings is visible on the Knowledge section, after **Test connection**, and on **theme toggle** (health tint colors). Row tooltips include last used, last test, and last error when available.

---

## Quota limit notifications

When anonymous daily quota is exhausted (`BudgetExhaustedError` in `http_client.py`), `core/knowledge/provider_limit_events.py` emits a debounced in-app notification (at most **once per provider per UTC day** while still on anonymous mode).

- Handler registered in `MainWindow._setup_provider_limit_notifications`
- Notification copy built in `core/notification_types.provider_limit_notification_event`
- Primary action deep-links to **Provider credentials** for that provider

Keyed providers do not receive “upgrade to free key” nudges — exhaustion is handled as a retrieval failure for that turn.

---

## Rate-limit policies (Qube-side)

Qube applies per-host throttling in `core/knowledge/host_scheduler.py`. Highlights:

| Host / provider | Policy (approx.) |
|---------------|------------------|
| OpenAlex | 8 req/s token bucket |
| NCBI (PubMed/PubChem) | 2.5 req/s anonymous · 8 req/s with key |
| arXiv | 3.5 s minimum interval |
| CourtListener | ~4 req/min |
| Semantic Scholar | 1 req/s |
| NASA ADS | 2 req/s |
| FRED | 2 req/s (provider: 120 req/min) |
| Companies House | 2 req/s (provider: 600 req / 5 min) |
| Alpha Vantage | 12 s interval (~5 req/min free tier) |
| EUR-Lex SPARQL | 2 req/s |
| CanLII | 0.5 s interval (~2 req/s) |
| NOAA NCEI | 2 req/s |
| NASA CMR | 2 req/s |
| EBSCO EDS | 2 req/s |
| BAILII | 2 s interval |
| ClinicalTrials.gov | 3 req/s |
| openFDA | 2 req/s |
| World Bank | 2 req/s |
| Eurostat | 2 req/s |
| USGS Publications | 2 req/s |
| USDA FDC | 2 req/s |
| NIST NVD | 6 s interval (anonymous) |
| IETF Datatracker | 2 req/s |
| BLS | 2 req/s |
| U.S. Census | 2 req/s |
| IEEE Xplore | 1 req/s |
| OECD SDMX | 1 req/s |
| WHO GHO | 2 req/s |
| CDC Content Services | 2 req/s |
| CDC Open Data | 2 req/s |
| NICE Syndication | 1 req/s |
| Zenodo | 2 req/s |
| FAOSTAT | 1 req/s |
| USDA ERS | 2 req/s |
| Copernicus CDS | 1 req/s |

Circuit-open hosts are short-circuited via `negative_cache.py` (default TTL **300s**, toggle with `QUBE_NEGATIVE_CACHE`).

---

## Optional runtime controls

| Variable | Default | Effect |
|----------|---------|--------|
| `QUBE_KNOWLEDGE_FIXTURES` | off | Fixture-backed adapter responses (`eval/fixtures/knowledge/`) |
| `QUBE_EVIDENCE_CACHE` | on | Query-level evidence cache (`~/.qube/evidence_cache/`) |
| `QUBE_EVIDENCE_CACHE_TTL` | 3600 | Evidence cache entry lifetime (seconds) |
| `QUBE_NEGATIVE_CACHE` | on | Skip hosts recently marked budget-exhausted / circuit-open |
| `QUBE_NEGATIVE_CACHE_TTL` | 300 | Negative cache lifetime (seconds) |
| `QUBE_TIERED_SCIENTIFIC_RETRIEVAL` | off | Primary-first adapter fan-out (Slice 8) |
| `QUBE_TIERED_SCIENTIFIC_THRESHOLD` | auto | Min candidates before phase-2 fallbacks run |
| `QUBE_QUERY_TYPE_ROUTING` | on | Institutional-first adapter ordering for guideline/statistics/standard intents (Slice 19) |
| Provider `env_var` | unset | See `provider_credentials.py` per provider |

Per-provider env aliases (e.g. `FRED_API_KEY`, `QUBE_COURTLISTENER_TOKEN`) are documented in `core/knowledge/credentials.py`.

---

## Eval corpora (by slice)

| Corpus file | Domain | Adapters exercised |
|-------------|--------|-------------------|
| `eval/retrieval_corpus/v1_scientific.json` | Scientific | Discipline routing (Phase 6c) |
| `eval/retrieval_corpus/v1_slice7a.json` | Scientific | `crossref`, `semantic_scholar`, `nasa_ads` |
| `eval/retrieval_corpus/v1_slice7b.json` | Scientific / Finance | `socarxiv`, `europe_pmc`, `fred` |
| `eval/retrieval_corpus/v1_slice7c.json` | Finance | `companies_house`, `alpha_vantage` |
| `eval/retrieval_corpus/v1_finance.json` | Finance | `sec_edgar` |
| `eval/retrieval_corpus/v1_legal.json` | Legal | `courtlistener` |
| `eval/retrieval_corpus/v1_slice8.json` | Legal | `eur_lex`, `canlii`, `bailii` |
| `eval/retrieval_corpus/v1_slice9_10.json` | Scientific | `ssrn`, `noaa`, `psyarxiv`, `nasa_earthdata` |
| `eval/retrieval_corpus/v1_slice11.json` | Scientific / Finance | `acm_dl`, `psycinfo`, `bloomberg_api` |
| `eval/retrieval_corpus/v1_slice12_p0.json` | Scientific / Finance | P0 institutional adapters (Slice 12) |
| `eval/retrieval_corpus/v1_slice13.json` | Scientific / Finance | OECD, NICE, CDC, WHO (Slice 13) |
| `eval/retrieval_corpus/v1_slice14.json` | Scientific | IPCC, FAO, USDA, Copernicus CDS (Slice 14) |
| `eval/retrieval_corpus/v1_slice15.json` | Scientific | OpenReview, ACL Anthology (Slice 15) |
| `eval/retrieval_corpus/v1_slice16.json` | Scientific | ChEMBL, UniProt, PDB, ChemRxiv (Slice 16) |
| `eval/retrieval_corpus/v1_slice17.json` | Legal | Congress.gov, GovInfo, legislation.gov.uk (Slice 17) |
| `eval/retrieval_corpus/v1_slice18.json` | Scientific | USPTO PatentsView, EPO Espacenet (Slice 18) |
| `eval/retrieval_corpus/v1_slice19.json` | Scientific | Query-type routing — NICE guideline, BLS statistics (Slice 19) |

Fixture-backed offline runs: `QUBE_KNOWLEDGE_FIXTURES=1 python3 tools/evaluate_retrieval.py --corpus <file> …`

See `eval/retrieval_corpus/README.md` for live eval gates, inter-query pacing, and `--single-adapter` usage.

---

## Test coverage (platform + adapters)

| Area | Tests |
|------|-------|
| HTTP client / retries | `tests/test_http_client.py`, `test_http_metrics.py`, `test_http_throttle_report.py` |
| Scheduler / breaker | `tests/test_host_scheduler.py`, `test_circuit_breaker.py`, `test_negative_cache.py` |
| Credentials | `tests/test_credentials.py`, `test_credential_resolver.py`, `test_provider_credential_test.py` |
| Source status / limits | `tests/test_provider_status.py` |
| Evidence cache | `tests/test_evidence_cache.py` |
| Tiered fan-out | `tests/test_tiered_scientific_retrieval.py` |
| Adapter slices | `tests/test_slice7a_adapters.py` … `test_slice18_adapters.py`, `test_slice9_10_adapters.py` |
| Query-type routing | `tests/test_scientific_query_type.py` |

---

## Related knowledge services (outside this adapter registry)

These are live **Knowledge Services** but not counted in the 58 adapter `SEARCH_FUNCTIONS` entries above:

| Service | ID | Role |
|---------|-----|------|
| Trusted knowledge | `trusted_knowledge` | Wikipedia-first general reference (`@trusted`) |
| Wikipedia | `wikipedia` | Direct Wikipedia API (`@wikipedia`) |
| General web | `general_web` | DuckDuckGo SERP (not in Preferred Sources catalog) |
| Internal corpus | `internal_corpus` | Local LanceDB library hybrid search |

Implementation files: `wikipedia_api.py`, `duckduckgo.py`, `lancedb_library.py`.

---

## Quick reference — access at a glance

### Works fully anonymously (no key ever required)

`crossref`, `europe_pmc`, `arxiv`, `biorxiv`, `inspire_hep`, `socarxiv`, `psyarxiv`, `nasa_earthdata`, `dblp`, `repec`, `sec_edgar`, `eur_lex`, `bailii`, `legislation_uk`, `clinicaltrials_gov`, `openfda`, `world_bank`, `eurostat`, `usgs`, `ietf_rfc`, `oecd`, `cdc`, `who`, `ipcc`, `openreview`, `acl_anthology`, `chembl`, `uniprot`, `pdb`, `chemrxiv`

### Anonymous OK, optional free key improves limits

`openalex`, `pubmed`, `pubchem`, `courtlistener`, `ssrn`, `acm_dl`, `usda_fdc`, `nist`, `us_census`, `usda`, `copernicus_cds`

### Key required for live retrieval

`semantic_scholar`, `nasa_ads`, `fred`, `companies_house`, `alpha_vantage`, `canlii`, `noaa`, `psycinfo`, `bloomberg_api`, `bls`, `ieee_xplore`, `nice`, `fao`, `congress_gov`, `govinfo`, `uspto_patentsview`, `epo_espacenet`

### Off by default (opt-in in Preferred Sources)

`nasa_ads`, `companies_house`, `alpha_vantage`, `bloomberg_api`, `eur_lex`, `canlii`, `bailii`, `ssrn`, `psyarxiv`, `noaa`, `nasa_earthdata`, `acm_dl`, `psycinfo`

---

## Maintenance

When shipping a new adapter:

1. Add implementation module under `core/knowledge/adapters/`.
2. Register in `registry.py` and `catalog.py` (`implemented=True`).
3. Add readiness + production strategy in `adapter_readiness.py`.
4. Route HTTP through `http_client.py` (or document why not).
5. Add provider credential spec if keyed; wire `resolve_credential()` in the adapter.
6. Add host policy in `host_scheduler.py`, bundle mapping, fixtures, and eval corpus entry.
7. If keyed, add a Source status metrics host mapping in `provider_status.py` (`_PROVIDER_METRICS_HOSTS`).
8. Run `python3 tools/sync_discipline_packs.py --check` after updating discipline packs.
9. Add or extend slice tests under `tests/test_slice*_adapters.py`.
10. Update this document: move shipped items from [Planned expansion roadmap](#planned-expansion-roadmap-external-review-july-2026) into the live inventory tables; adjust priority tiers.
11. Update the [HTTP resilience plan](./knowledge_adapter_http_resilience_plan.md) slice status as needed.
