<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->
# Live sources overview

## Common questions

- What Live Sources are available in Knowledge settings?
- Which scientific literature adapters does Qube support?
- Do any sources require an API key?

## What Live Sources are

**Live Sources** (Settings → Knowledge) connect Qube to online catalogs — scientific papers, finance filings, legal opinions, and more. This page summarizes adapter metadata shipped with the app.

Library document search and Live Sources are different: Library searches files you ingested; Live Sources query external services when `@internet`, `@evidence`, or related tools route to them.

## Adapter catalog summary

## Finance (`finance_knowledge`)

9 catalog entries (9 implemented). Configure defaults in **Settings → Knowledge → Live Sources**.

### Finance

- **Alpha Vantage** (`alpha_vantage`) — API key required, off by default
- **Bloomberg (API)** (`bloomberg_api`) — API key required, off by default
- **BLS** (`bls`) — API key required
- **Companies House** (`companies_house`) — API key required, off by default
- **Eurostat** (`eurostat`)
- **FRED** (`fred`) — API key required
- **OECD** (`oecd`)
- **SEC EDGAR** (`sec_edgar`)
- **World Bank Open Data** (`world_bank`)

## Legal (`legal_knowledge`)

7 catalog entries (7 implemented). Configure defaults in **Settings → Knowledge → Live Sources**.

### Legal

- **BAILII** (`bailii`) — off by default
- **CanLII** (`canlii`) — API key required, off by default
- **Congress.gov** (`congress_gov`) — API key required
- **CourtListener** (`courtlistener`)
- **EUR-Lex** (`eur_lex`) — off by default
- **GovInfo** (`govinfo`) — API key required
- **legislation.gov.uk** (`legislation_uk`)

## Scientific literature (`scientific_evidence`)

46 catalog entries (46 implemented). Configure defaults in **Settings → Knowledge → Live Sources**.

### Agriculture & Nutrition

- **FAOSTAT** (`fao`) — API key required
- **USDA ERS** (`usda`) — optional API key
- **USDA FoodData Central** (`usda_fdc`) — optional API key

### Biology

- **bioRxiv** (`biorxiv`)
- **Protein Data Bank** (`pdb`)
- **UniProt** (`uniprot`)

### Chemistry

- **ChEMBL** (`chembl`)
- **ChemRxiv** (`chemrxiv`)
- **EPO Espacenet** (`epo_espacenet`) — API key required
- **PubChem** (`pubchem`) — optional API key
- **USPTO PatentsView** (`uspto_patentsview`) — API key required

### Computer Science

- **ACL Anthology** (`acl_anthology`)
- **ACM Digital Library** (`acm_dl`) — optional API key, off by default
- **DBLP** (`dblp`)
- **OpenReview** (`openreview`)

### Earth & Environment

- **Copernicus CDS** (`copernicus_cds`) — optional API key
- **IPCC** (`ipcc`)
- **NASA Earthdata** (`nasa_earthdata`) — off by default
- **NOAA NCEI** (`noaa`) — API key required, off by default
- **USGS Publications** (`usgs`)

### Economics

- **BLS** (`bls`) — API key required
- **Eurostat** (`eurostat`)
- **OECD** (`oecd`)
- **RePEc** (`repec`)
- **SSRN** (`ssrn`) — off by default
- **World Bank Open Data** (`world_bank`)

### Engineering

- **IEEE Xplore** (`ieee_xplore`) — API key required
- **IETF RFCs** (`ietf_rfc`)
- **NIST NVD** (`nist`) — optional API key

### Psychology

- **PsyArXiv** (`psyarxiv`) — off by default
- **PsycINFO** (`psycinfo`) — API key required, off by default

### Science

- **arXiv** (`arxiv`)
- **CDC** (`cdc`)
- **ClinicalTrials.gov** (`clinicaltrials_gov`)
- **Crossref** (`crossref`)
- **Europe PMC** (`europe_pmc`)
- **INSPIRE-HEP** (`inspire_hep`)
- **NASA ADS** (`nasa_ads`) — API key required, off by default
- **NICE** (`nice`) — API key required
- **OpenAlex** (`openalex`) — optional API key
- **openFDA** (`openfda`)
- **PubMed** (`pubmed`) — optional API key
- **Semantic Scholar** (`semantic_scholar`) — API key required
- **WHO GHO** (`who`)

### Social Science

- **SocArXiv** (`socarxiv`)
- **U.S. Census Bureau** (`us_census`) — optional API key

## Also called

internet search adapters, online lookup, external knowledge sources, evidence adapters

## Related

- [Knowledge settings](../features/settings/knowledge.md) — full Live Sources UI (Phase 3)
- [Composer tools](composer-tools.md) — `@evidence`, `@finance`, `@legal`, and related tools
