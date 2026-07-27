"""Provider credential catalog (one row per provider id, not per adapter)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCredentialSpec:
    """Settings / resolver metadata for one knowledge API provider."""

    provider_id: str
    label: str
    adapter_ids: tuple[str, ...]
    discovery_provider_ids: tuple[str, ...] = ()
    env_var: str | None = None
    supports_anonymous: bool = True
    supports_free_api_key: bool = False
    paid_tier_available: bool = False
    key_required: bool = False
    signup_url: str = ""
    docs_url: str = ""
    anonymous_summary: str = "Anonymous access"
    key_benefits: str = ""
    test_probe: str = ""


PROVIDER_CREDENTIAL_SPECS: tuple[ProviderCredentialSpec, ...] = (
    ProviderCredentialSpec(
        provider_id="brave_search",
        label="Brave Search API",
        adapter_ids=(),
        discovery_provider_ids=("brave_search",),
        env_var="QUBE_BRAVE_SEARCH_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=False,
        signup_url="https://brave.com/search/api/",
        docs_url="https://api.search.brave.com/app/documentation/web-search/get-started",
        anonymous_summary="API key not configured",
        key_benefits=(
            "Optional web-search fallback when DuckDuckGo is blocked. "
            "DuckDuckGo remains the primary provider for every turn."
        ),
        test_probe="brave_search_web",
    ),
    ProviderCredentialSpec(
        provider_id="searxng",
        label="SearXNG API key",
        adapter_ids=(),
        discovery_provider_ids=("searxng",),
        env_var="QUBE_SEARXNG_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        key_required=False,
        anonymous_summary="Optional — only if your instance requires auth",
        key_benefits="Bearer token for private SearXNG instances.",
        test_probe="searxng_json_search",
    ),
    ProviderCredentialSpec(
        provider_id="openalex",
        label="OpenAlex",
        adapter_ids=("openalex",),
        env_var="QUBE_OPENALEX_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://openalex.org/settings/api",
        docs_url="https://docs.openalex.org/how-to-use-the-api/api-overview",
        key_benefits=(
            "Adding a free API key increases your daily search budget and may improve reliability."
        ),
        test_probe="openalex_rate_limit",
    ),
    ProviderCredentialSpec(
        provider_id="ncbi",
        label="NCBI (PubMed & PubChem)",
        adapter_ids=("pubmed", "pubchem"),
        env_var="QUBE_NCBI_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://www.ncbi.nlm.nih.gov/account/settings/",
        docs_url="https://www.ncbi.nlm.nih.gov/home/develop/api/",
        key_benefits="Free NCBI key raises the shared rate limit for PubMed and PubChem.",
        test_probe="ncbi_einfo",
    ),
    ProviderCredentialSpec(
        provider_id="courtlistener",
        label="CourtListener",
        adapter_ids=("courtlistener",),
        env_var="QUBE_COURTLISTENER_API_TOKEN",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://www.courtlistener.com/sign-in/register/",
        docs_url="https://www.courtlistener.com/help/api/rest/",
        key_benefits="Free account token unlocks higher CourtListener API limits.",
        test_probe="courtlistener_profile",
    ),
    ProviderCredentialSpec(
        provider_id="semantic_scholar",
        label="Semantic Scholar",
        adapter_ids=("semantic_scholar",),
        env_var="QUBE_SEMANTIC_SCHOLAR_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.semanticscholar.org/product/api",
        docs_url="https://api.semanticscholar.org/",
        anonymous_summary="API key required",
        key_benefits="Semantic Scholar requires an API key for live retrieval in Qube.",
        test_probe="semantic_scholar_search",
    ),
    ProviderCredentialSpec(
        provider_id="nasa_ads",
        label="NASA ADS",
        adapter_ids=("nasa_ads",),
        env_var="QUBE_NASA_ADS_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://ui.adsabs.harvard.edu/user/settings/token",
        docs_url="https://github.com/adsabs/adsabs-dev-api/blob/master/README.md",
        anonymous_summary="API key required",
        key_benefits="NASA ADS requires a personal API token for astrophysics literature search.",
        test_probe="nasa_ads_search",
    ),
    ProviderCredentialSpec(
        provider_id="fred",
        label="FRED (Federal Reserve Economic Data)",
        adapter_ids=("fred",),
        env_var="QUBE_FRED_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
        docs_url="https://fred.stlouisfed.org/docs/api/fred/",
        anonymous_summary="API key required",
        key_benefits="Free FRED API key unlocks macroeconomic series search for @finance queries.",
        test_probe="fred_series_search",
    ),
    ProviderCredentialSpec(
        provider_id="companies_house",
        label="Companies House (UK)",
        adapter_ids=("companies_house",),
        env_var="QUBE_COMPANIES_HOUSE_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://developer.company-information.service.gov.uk/manage-applications",
        docs_url="https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/reference/search/search-companies",
        anonymous_summary="API key required",
        key_benefits="Free UK Companies House API key unlocks company registry search for @finance.",
        test_probe="companies_house_search",
    ),
    ProviderCredentialSpec(
        provider_id="alpha_vantage",
        label="Alpha Vantage",
        adapter_ids=("alpha_vantage",),
        env_var="QUBE_ALPHA_VANTAGE_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.alphavantage.co/support/#api-key",
        docs_url="https://www.alphavantage.co/documentation/",
        anonymous_summary="API key required",
        key_benefits="Free Alpha Vantage key unlocks market symbol search for @finance queries.",
        test_probe="alpha_vantage_symbol_search",
    ),
    ProviderCredentialSpec(
        provider_id="canlii",
        label="CanLII",
        adapter_ids=("canlii",),
        env_var="QUBE_CANLII_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.canlii.org/en/feedback/feedback.html",
        docs_url="https://github.com/canlii/API_documentation/blob/master/EN.md",
        anonymous_summary="API key required",
        key_benefits="Free CanLII API key unlocks Canadian case law search for @legal queries.",
        test_probe="canlii_search",
    ),
    ProviderCredentialSpec(
        provider_id="noaa",
        label="NOAA NCEI",
        adapter_ids=("noaa",),
        env_var="QUBE_NOAA_API_TOKEN",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.ncdc.noaa.gov/cdo-web/token",
        docs_url="https://www.ncdc.noaa.gov/cdo-web/webservices/v2",
        anonymous_summary="API token required",
        key_benefits="Free NOAA token unlocks climate dataset search for earth-science queries.",
        test_probe="noaa_datasets",
    ),
    ProviderCredentialSpec(
        provider_id="ebsco_eds",
        label="EBSCO Discovery (PsycINFO)",
        adapter_ids=("psycinfo",),
        env_var="QUBE_EBSCO_EDS_PASSWORD",
        supports_anonymous=False,
        key_required=True,
        signup_url="https://developer.ebsco.com/home/docs/request-application-credentials",
        docs_url="https://developer.ebsco.com/eds-api/docs/authentication-1",
        anonymous_summary="Institutional EDS credentials required",
        key_benefits=(
            "PsycINFO requires institutional EBSCO EDS API credentials. Set "
            "QUBE_EBSCO_EDS_USER_ID in the environment and paste the EDS password "
            "here (or use user|password|profile in Settings)."
        ),
        test_probe="ebsco_eds_search",
    ),
    ProviderCredentialSpec(
        provider_id="bloomberg",
        label="Bloomberg Open API",
        adapter_ids=("bloomberg_api",),
        env_var="QUBE_BLOOMBERG_API_URL",
        supports_anonymous=False,
        paid_tier_available=True,
        key_required=True,
        signup_url="https://www.bloomberg.com/professional/support/api-library/",
        docs_url="https://github.com/bloomberg/blpapi-http/blob/develop/doc/http-api-guide.md",
        anonymous_summary="Bloomberg HTTP bridge URL required",
        key_benefits=(
            "Enterprise Bloomberg Terminal or B-PIPE HTTP bridge URL "
            "(e.g. https://localhost:8298). Paste the base URL in Settings or set "
            "QUBE_BLOOMBERG_API_URL."
        ),
        test_probe="bloomberg_instruments",
    ),
    ProviderCredentialSpec(
        provider_id="usda_fdc",
        label="USDA FoodData Central",
        adapter_ids=("usda_fdc",),
        env_var="QUBE_USDA_FDC_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://fdc.nal.usda.gov/api-key-signup/",
        docs_url="https://fdc.nal.usda.gov/api-guide/",
        key_benefits="Optional free API key raises USDA FDC rate limits beyond DEMO_KEY.",
        test_probe="usda_fdc_search",
    ),
    ProviderCredentialSpec(
        provider_id="bls",
        label="BLS (U.S. Bureau of Labor Statistics)",
        adapter_ids=("bls",),
        env_var="QUBE_BLS_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://data.bls.gov/registrationEngine/",
        docs_url="https://www.bls.gov/developers/",
        anonymous_summary="API registration key required",
        key_benefits="Free BLS registration key unlocks official labor statistics series search.",
        test_probe="bls_series_search",
    ),
    ProviderCredentialSpec(
        provider_id="us_census",
        label="U.S. Census Bureau",
        adapter_ids=("us_census",),
        env_var="QUBE_CENSUS_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://api.census.gov/data/key_signup.html",
        docs_url="https://www.census.gov/data/developers/guidance/api-user-guide.html",
        key_benefits="Optional Census API key improves rate limits for catalog search.",
        test_probe="us_census_catalog",
    ),
    ProviderCredentialSpec(
        provider_id="nist",
        label="NIST NVD",
        adapter_ids=("nist",),
        env_var="QUBE_NIST_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://nvd.nist.gov/developers/request-an-api-key",
        docs_url="https://nvd.nist.gov/developers/start-here",
        key_benefits="Optional NVD API key raises NIST keyword search rate limits.",
        test_probe="nist_nvd_search",
    ),
    ProviderCredentialSpec(
        provider_id="ieee_xplore",
        label="IEEE Xplore",
        adapter_ids=("ieee_xplore",),
        env_var="QUBE_IEEE_XPLORE_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://developer.ieee.org/",
        docs_url="https://developer.ieee.org/docs/read/Metadata_API_overview",
        anonymous_summary="IEEE developer API key required",
        key_benefits="Free IEEE developer key unlocks engineering literature search.",
        test_probe="ieee_xplore_search",
    ),
    ProviderCredentialSpec(
        provider_id="nice",
        label="NICE Syndication",
        adapter_ids=("nice",),
        env_var="QUBE_NICE_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.nice.org.uk/reusing-our-content/nice-syndication-api",
        docs_url="https://www.nice.org.uk/corporate/ecd10",
        anonymous_summary="NICE syndication API key required",
        key_benefits="NICE syndication licence unlocks UK clinical guidance search for medicine queries.",
        test_probe="nice_guidance_index",
    ),
    ProviderCredentialSpec(
        provider_id="fao",
        label="FAOSTAT",
        adapter_ids=("fao",),
        env_var="QUBE_FAO_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.fao.org/faostat/en/#home",
        docs_url="https://www.fao.org/statistics/highlights-archive/highlights-detail/faostat-launches-a-new-api-developer-portal-to-make-data-access-easier/en",
        anonymous_summary="FAOSTAT API bearer token required",
        key_benefits="FAOSTAT developer portal JWT unlocks agricultural dataset discovery.",
        test_probe="faostat_datasets",
    ),
    ProviderCredentialSpec(
        provider_id="usda",
        label="USDA (api.data.gov)",
        adapter_ids=("usda",),
        env_var="QUBE_USDA_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://api.data.gov/signup/",
        docs_url="https://www.ers.usda.gov/developer/data-apis/arms-data-api",
        key_benefits="Optional api.data.gov key improves USDA ERS API rate limits beyond DEMO_KEY.",
        test_probe="usda_arms_variable",
    ),
    ProviderCredentialSpec(
        provider_id="copernicus_cds",
        label="Copernicus CDS",
        adapter_ids=("copernicus_cds",),
        env_var="QUBE_COPERNICUS_CDS_API_KEY",
        supports_anonymous=True,
        supports_free_api_key=True,
        signup_url="https://cds.climate.copernicus.eu/profile",
        docs_url="https://cds.climate.copernicus.eu/how-to-api",
        key_benefits="Optional CDS personal access token enables authenticated catalogue probes and future data retrieval.",
        test_probe="copernicus_cds_catalogue",
    ),
    ProviderCredentialSpec(
        provider_id="congress_gov",
        label="Congress.gov",
        adapter_ids=("congress_gov",),
        env_var="QUBE_CONGRESS_GOV_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://api.congress.gov/sign-up/",
        docs_url="https://github.com/LibraryOfCongress/api.congress.gov",
        anonymous_summary="API key required",
        key_benefits="Free api.data.gov key unlocks Congress.gov bill metadata search for @legal queries.",
        test_probe="congress_gov_bill_list",
    ),
    ProviderCredentialSpec(
        provider_id="govinfo",
        label="GovInfo",
        adapter_ids=("govinfo",),
        env_var="QUBE_GOVINFO_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://www.govinfo.gov/api-signup",
        docs_url="https://github.com/usgpo/api",
        anonymous_summary="API key required",
        key_benefits="Free api.data.gov key unlocks GovInfo federal publication search for @legal queries.",
        test_probe="govinfo_search",
    ),
    ProviderCredentialSpec(
        provider_id="patentsview",
        label="PatentsView (USPTO)",
        adapter_ids=("uspto_patentsview",),
        env_var="QUBE_PATENTSVIEW_API_KEY",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://search.patentsview.org/docs/",
        docs_url="https://search.patentsview.org/docs/docs/Search%20API/SearchAPIReference/",
        anonymous_summary="PatentsView API key required",
        key_benefits="Free PatentsView PatentSearch API key unlocks U.S. patent metadata search.",
        test_probe="patentsview_patent_search",
    ),
    ProviderCredentialSpec(
        provider_id="epo_ops",
        label="EPO Open Patent Services",
        adapter_ids=("epo_espacenet",),
        env_var="QUBE_EPO_OPS_CONSUMER_SECRET",
        supports_anonymous=False,
        supports_free_api_key=True,
        key_required=True,
        signup_url="https://developers.epo.org/user/register",
        docs_url="https://link.epo.org/web/searching-for-patents/data/en-ops-v3.2-documentation-version-1.3.20.pdf",
        anonymous_summary="EPO OPS consumer credentials required",
        key_benefits=(
            "Register at developers.epo.org and set QUBE_EPO_OPS_CONSUMER_KEY in the environment "
            "plus the consumer secret here (or paste key:secret)."
        ),
        test_probe="epo_ops_search",
    ),
)

_SPEC_BY_ID: dict[str, ProviderCredentialSpec] = {
    spec.provider_id: spec for spec in PROVIDER_CREDENTIAL_SPECS
}

_ADAPTER_TO_PROVIDER: dict[str, str] = {}
for _spec in PROVIDER_CREDENTIAL_SPECS:
    for _adapter_id in _spec.adapter_ids:
        _ADAPTER_TO_PROVIDER[_adapter_id] = _spec.provider_id


def get_provider_credential_spec(provider_id: str) -> ProviderCredentialSpec | None:
    return _SPEC_BY_ID.get((provider_id or "").strip().lower())


def provider_id_for_adapter(adapter_id: str) -> str | None:
    return _ADAPTER_TO_PROVIDER.get((adapter_id or "").strip().lower())


def list_provider_credential_specs() -> tuple[ProviderCredentialSpec, ...]:
    return PROVIDER_CREDENTIAL_SPECS


def list_active_provider_credential_specs() -> tuple[ProviderCredentialSpec, ...]:
    """Specs for providers whose adapters are live in the catalog."""
    return tuple(
        spec for spec in PROVIDER_CREDENTIAL_SPECS if provider_has_implemented_adapter(spec)
    )


def adapter_credentials_hint(adapter_id: str) -> str | None:
    """Tooltip line linking a live adapter to Configure."""
    pid = provider_id_for_adapter(adapter_id)
    if pid is None:
        return None
    spec = get_provider_credential_spec(pid)
    if spec is None or not provider_has_implemented_adapter(spec):
        return None
    if spec.key_required:
        return "Requires an API key — use Configure to add one."
    if spec.supports_free_api_key:
        return "Optional free API key available — use Configure to improve limits."
    return None


def provider_has_implemented_adapter(spec: ProviderCredentialSpec) -> bool:
    """True when at least one catalog adapter or discovery provider is live."""
    if spec.discovery_provider_ids:
        from core.knowledge.discovery.registry import get_discovery_provider

        for provider_id in spec.discovery_provider_ids:
            if get_discovery_provider(provider_id) is not None:
                return True
    from core.knowledge.adapters.catalog import get_adapter_entry

    for adapter_id in spec.adapter_ids:
        entry = get_adapter_entry(adapter_id)
        if entry is not None and entry.implemented:
            return True
    return False

