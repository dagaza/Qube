"""Lightweight connection probes for Settings → Knowledge provider credentials."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from core.knowledge.credentials import resolve_credential
from core.knowledge.http_client import knowledge_get, knowledge_post
from core.knowledge.provider_credentials import get_provider_credential_spec

logger = logging.getLogger("Qube.Knowledge.Credentials")

USER_AGENT = "Qube/1.0 (local@qube.app)"


@dataclass(frozen=True)
class ProviderCredentialTestResult:
    ok: bool
    message: str
    status_code: int | None = None


def test_provider_credential(
    provider_id: str,
    *,
    override_secret: str | None = None,
    timeout: float = 10.0,
) -> ProviderCredentialTestResult:
    """Run a minimal live probe for one provider (optional unsaved key from Settings UI)."""
    pid = (provider_id or "").strip().lower()
    spec = get_provider_credential_spec(pid)
    if spec is None:
        return ProviderCredentialTestResult(False, "Unknown provider.", None)

    secret = (override_secret or "").strip() or None
    if secret is None:
        secret = resolve_credential(pid).secret

    if spec.key_required and not secret:
        return ProviderCredentialTestResult(
            False,
            "API key required before testing this provider.",
            None,
        )

    probe = (spec.test_probe or "").strip().lower()
    try:
        if probe == "openalex_rate_limit":
            return _probe_openalex(secret, timeout=timeout)
        if probe == "ncbi_einfo":
            return _probe_ncbi(secret, timeout=timeout)
        if probe == "courtlistener_profile":
            return _probe_courtlistener(secret, timeout=timeout)
        if probe == "semantic_scholar_search":
            return _probe_semantic_scholar(secret, timeout=timeout)
        if probe == "nasa_ads_search":
            return _probe_nasa_ads(secret, timeout=timeout)
        if probe == "fred_series_search":
            return _probe_fred(secret, timeout=timeout)
        if probe == "companies_house_search":
            return _probe_companies_house(secret, timeout=timeout)
        if probe == "alpha_vantage_symbol_search":
            return _probe_alpha_vantage(secret, timeout=timeout)
        if probe == "canlii_search":
            return _probe_canlii(secret, timeout=timeout)
        if probe == "noaa_datasets":
            return _probe_noaa(secret, timeout=timeout)
        if probe == "ebsco_eds_search":
            return _probe_ebsco_eds(secret, timeout=timeout)
        if probe == "bloomberg_instruments":
            return _probe_bloomberg(secret, timeout=timeout)
        if probe == "usda_fdc_search":
            return _probe_usda_fdc(secret, timeout=timeout)
        if probe == "bls_series_search":
            return _probe_bls(secret, timeout=timeout)
        if probe == "us_census_catalog":
            return _probe_us_census(secret, timeout=timeout)
        if probe == "nist_nvd_search":
            return _probe_nist(secret, timeout=timeout)
        if probe == "ieee_xplore_search":
            return _probe_ieee_xplore(secret, timeout=timeout)
        if probe == "nice_guidance_index":
            return _probe_nice(secret, timeout=timeout)
        if probe == "faostat_datasets":
            return _probe_fao(secret, timeout=timeout)
        if probe == "usda_arms_variable":
            return _probe_usda(secret, timeout=timeout)
        if probe == "copernicus_cds_catalogue":
            return _probe_copernicus_cds(secret, timeout=timeout)
        if probe == "congress_gov_bill_list":
            return _probe_congress_gov(secret, timeout=timeout)
        if probe == "govinfo_search":
            return _probe_govinfo(secret, timeout=timeout)
        if probe == "patentsview_patent_search":
            return _probe_patentsview(secret, timeout=timeout)
        if probe == "epo_ops_search":
            return _probe_epo_ops(secret, timeout=timeout)
        if probe == "brave_search_web":
            return _probe_brave_search(secret, timeout=timeout)
        return ProviderCredentialTestResult(
            False,
            "No test probe configured for this provider.",
            None,
        )
    except Exception as exc:
        logger.warning("[Credentials] probe failed for %s: %s", pid, exc)
        return ProviderCredentialTestResult(False, f"Connection failed: {exc}", None)


def _probe_openalex(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    headers = {"User-Agent": USER_AGENT}
    params: dict[str, str] = {}
    if secret:
        params["api_key"] = secret
    resp = knowledge_get(
        "https://api.openalex.org/rate-limit",
        params=params or None,
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code == 404:
        resp = knowledge_get(
            "https://api.openalex.org/works",
            params={**params, "search": "test", "per_page": "1"},
            headers=headers,
            timeout=timeout,
        )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "OpenAlex connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"OpenAlex returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_ncbi(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    headers = {"User-Agent": USER_AGENT}
    params = {
        "db": "pubmed",
        "retmode": "json",
        "tool": "Qube",
        "email": "local@qube.app",
    }
    if secret:
        params["api_key"] = secret
    resp = knowledge_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
        params=params,
        headers=headers,
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "NCBI E-utilities connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"NCBI returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_courtlistener(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if secret:
        headers["Authorization"] = f"Token {secret}"
    resp = knowledge_get(
        "https://www.courtlistener.com/api/rest/v4/users/",
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code == 401:
        return ProviderCredentialTestResult(
            False,
            "CourtListener rejected the token (HTTP 401).",
            resp.status_code,
        )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "CourtListener connection succeeded.",
            resp.status_code,
        )
    if resp.status_code == 403 and not secret:
        return ProviderCredentialTestResult(
            True,
            "CourtListener reachable anonymously (token optional for higher limits).",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"CourtListener returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_semantic_scholar(
    secret: str | None, *, timeout: float
) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API key required before testing Semantic Scholar.",
            None,
        )
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "x-api-key": secret,
    }
    resp = knowledge_get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": "test", "limit": "1", "fields": "paperId,title"},
        headers=headers,
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "Semantic Scholar connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"Semantic Scholar returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_nasa_ads(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API token required before testing NASA ADS.",
            None,
        )
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Authorization": f"Bearer {secret}",
    }
    resp = knowledge_get(
        "https://api.adsabs.harvard.edu/v1/search/query",
        params={"q": "star", "rows": "1", "fl": "title,bibcode"},
        headers=headers,
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "NASA ADS connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"NASA ADS returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_fred(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API key required before testing FRED.",
            None,
        )
    resp = knowledge_get(
        "https://api.stlouisfed.org/fred/series/search",
        params={
            "api_key": secret,
            "search_text": "gdp",
            "file_type": "json",
            "limit": "1",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "FRED connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"FRED returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_companies_house(
    secret: str | None, *, timeout: float
) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API key required before testing Companies House.",
            None,
        )
    resp = knowledge_get(
        "https://api.company-information.service.gov.uk/search/companies",
        params={"q": "tesco", "items_per_page": "1"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        auth=(secret, ""),
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "Companies House connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"Companies House returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_alpha_vantage(
    secret: str | None, *, timeout: float
) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API key required before testing Alpha Vantage.",
            None,
        )
    resp = knowledge_get(
        "https://www.alphavantage.co/query",
        params={
            "function": "SYMBOL_SEARCH",
            "keywords": "microsoft",
            "apikey": secret,
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            return ProviderCredentialTestResult(
                False,
                str(payload.get("Error Message")),
                resp.status_code,
            )
        return ProviderCredentialTestResult(
            True,
            "Alpha Vantage connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"Alpha Vantage returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_canlii(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API key required before testing CanLII.",
            None,
        )
    resp = knowledge_get(
        "https://api.canlii.org/v1/caseBrowse/en/",
        params={"api_key": secret},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "CanLII connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"CanLII returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_noaa(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "API token required before testing NOAA.",
            None,
        )
    resp = knowledge_get(
        "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets",
        params={"limit": "1"},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "token": secret,
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "NOAA connection succeeded.",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"NOAA returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_ebsco_eds(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    user_id = os.environ.get("QUBE_EBSCO_EDS_USER_ID", "").strip()
    password = (secret or "").strip()
    profile = os.environ.get("QUBE_EBSCO_EDS_PROFILE", "eds").strip() or "eds"
    if "|" in password:
        parts = [part.strip() for part in password.split("|")]
        user_id = user_id or (parts[0] if parts else "")
        password = parts[1] if len(parts) > 1 else password
        if len(parts) > 2 and parts[2]:
            profile = parts[2]
    if not user_id or not password:
        return ProviderCredentialTestResult(
            False,
            "Set QUBE_EBSCO_EDS_USER_ID and EDS password before testing.",
            None,
        )
    from core.knowledge.http_client import knowledge_post

    auth_resp = knowledge_post(
        "https://eds-api.ebscohost.com/authservice/rest/uidauth",
        json={"UserId": user_id, "Password": password, "InterfaceId": "WSapi"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if not (200 <= auth_resp.status_code < 300):
        return ProviderCredentialTestResult(
            False,
            f"EBSCO auth returned HTTP {auth_resp.status_code}.",
            auth_resp.status_code,
        )
    auth_payload = auth_resp.json()
    auth_token = str(auth_payload.get("AuthToken") or "").strip()
    if not auth_token:
        return ProviderCredentialTestResult(
            False,
            "EBSCO auth succeeded but no token was returned.",
            auth_resp.status_code,
        )
    session_resp = knowledge_post(
        "https://eds-api.ebscohost.com/edsapi/rest/createsession",
        json={"Profile": profile, "Guest": "n"},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-authenticationToken": auth_token,
        },
        timeout=timeout,
    )
    if 200 <= session_resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "EBSCO EDS connection succeeded.",
            session_resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"EBSCO session returned HTTP {session_resp.status_code}.",
        session_resp.status_code,
    )


def _probe_bloomberg(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    base_url = (secret or os.environ.get("QUBE_BLOOMBERG_API_URL", "")).strip().rstrip("/")
    if not base_url:
        return ProviderCredentialTestResult(
            False,
            "Bloomberg HTTP API URL required before testing.",
            None,
        )
    from urllib.parse import urlencode, urlparse

    from core.knowledge.http_client import knowledge_post

    host = (urlparse(base_url).hostname or "bloomberg").lower()
    query = urlencode(
        {
            "ns": "blp",
            "service": "instruments",
            "type": "instrumentListRequest",
        }
    )
    resp = knowledge_post(
        f"{base_url}/request?{query}",
        json={"query": "IBM", "maxResults": 1},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Version": "1.0.0",
        },
        host=host,
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("message") == "OK":
            return ProviderCredentialTestResult(
                True,
                "Bloomberg HTTP API connection succeeded.",
                resp.status_code,
            )
        return ProviderCredentialTestResult(
            True,
            "Bloomberg HTTP API responded (verify entitlements for full search).",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"Bloomberg HTTP API returned HTTP {resp.status_code}.",
        resp.status_code,
    )


def _probe_usda_fdc(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    params = {"query": "apple", "pageSize": "1", "api_key": secret or "DEMO_KEY"}
    resp = knowledge_get(
        "https://api.nal.usda.gov/fdc/v1/foods/search",
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "USDA FDC connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"USDA FDC returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_bls(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(False, "BLS registration key required.", None)
    from core.knowledge.http_client import knowledge_post

    resp = knowledge_post(
        "https://api.bls.gov/publicAPI/v2/timeseries/search",
        json={"series_text": "unemployment", "registrationkey": secret, "limit": 1},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "BLS connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"BLS returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_us_census(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    params = {"key": secret} if secret else None
    resp = knowledge_get(
        "https://api.census.gov/data.json",
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True, "U.S. Census catalog connection succeeded.", resp.status_code
        )
    return ProviderCredentialTestResult(
        False, f"U.S. Census returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_nist(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if secret:
        headers["apiKey"] = secret
    resp = knowledge_get(
        "https://services.nvd.nist.gov/rest/json/cves/2.0",
        params={"keywordSearch": "encryption", "resultsPerPage": "1"},
        headers=headers,
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "NIST NVD connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"NIST NVD returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_ieee_xplore(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(False, "IEEE developer API key required.", None)
    resp = knowledge_get(
        "https://ieeexploreapi.ieee.org/api/v1/search/articles",
        params={"querytext": "robotics", "max_records": "1", "apikey": secret},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "IEEE Xplore connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"IEEE Xplore returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_nice(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(False, "NICE syndication API key required.", None)
    resp = knowledge_get(
        "https://api.nice.org.uk/services/guidance/index",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.nice.syndication.services+json",
            "API-Key": secret,
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "NICE syndication connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"NICE syndication returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_fao(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(False, "FAOSTAT API bearer token required.", None)
    resp = knowledge_get(
        "https://faostatservices.fao.org/api/v1/en/data/datasets",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Authorization": f"Bearer {secret}",
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "FAOSTAT connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"FAOSTAT returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_usda(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    api_key = (secret or "").strip() or "DEMO_KEY"
    resp = knowledge_post(
        "https://api.ers.usda.gov/data/arms/variable",
        json={"keyword": "wheat"},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "USDA ERS ARMS connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"USDA ERS returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_copernicus_cds(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if secret:
        headers["PRIVATE-TOKEN"] = secret
    resp = knowledge_get(
        "https://cds.climate.copernicus.eu/api/catalogue/v1/collections",
        params={"limit": 1},
        headers=headers,
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "Copernicus CDS catalogue reachable.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"Copernicus CDS returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_congress_gov(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(False, "Congress.gov API key required.", None)
    resp = knowledge_get(
        "https://api.congress.gov/v3/bill",
        params={"api_key": secret, "limit": 1, "format": "json"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "Congress.gov connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"Congress.gov returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_govinfo(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    api_key = (secret or "").strip() or "DEMO_KEY"
    resp = knowledge_post(
        "https://api.govinfo.gov/search",
        json={"query": "privacy", "pageSize": "1", "offsetMark": "*"},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "GovInfo connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"GovInfo returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_patentsview(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(False, "PatentsView API key required.", None)
    resp = knowledge_post(
        "https://search.patentsview.org/api/v1/patent",
        json={
            "q": {"patent_title": {"_text_any": "battery"}},
            "f": ["patent_id", "patent_title"],
            "o": {"per_page": 1},
        },
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": secret,
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(True, "PatentsView connection succeeded.", resp.status_code)
    return ProviderCredentialTestResult(
        False, f"PatentsView returned HTTP {resp.status_code}.", resp.status_code
    )


def _probe_epo_ops(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    import base64
    import os

    key = os.environ.get("QUBE_EPO_OPS_CONSUMER_KEY", "").strip()
    consumer_secret = (secret or "").strip()
    if not key and ":" in consumer_secret:
        key, consumer_secret = consumer_secret.split(":", 1)
    if not key or not consumer_secret:
        return ProviderCredentialTestResult(
            False,
            "EPO OPS consumer key and secret required (set QUBE_EPO_OPS_CONSUMER_KEY and secret).",
            None,
        )
    auth = base64.b64encode(f"{key}:{consumer_secret}".encode()).decode()
    token_resp = knowledge_post(
        "https://ops.epo.org/3.2/auth/accesstoken",
        data={"grant_type": "client_credentials"},
        headers={
            "User-Agent": USER_AGENT,
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    if not (200 <= token_resp.status_code < 300):
        return ProviderCredentialTestResult(
            False, f"EPO OPS token request returned HTTP {token_resp.status_code}.", token_resp.status_code
        )
    token = str(token_resp.json().get("access_token") or "").strip()
    if not token:
        return ProviderCredentialTestResult(False, "EPO OPS token response missing access_token.", token_resp.status_code)
    search_resp = knowledge_get(
        "https://ops.epo.org/3.2/rest-services/published-data/search/biblio",
        params={"q": "ti=battery", "Range": "1-1"},
        headers={
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {token}",
            "Accept": "application/exchange+xml",
        },
        timeout=timeout,
    )
    if 200 <= search_resp.status_code < 300:
        return ProviderCredentialTestResult(True, "EPO OPS connection succeeded.", search_resp.status_code)
    return ProviderCredentialTestResult(
        False, f"EPO OPS search returned HTTP {search_resp.status_code}.", search_resp.status_code
    )


def _probe_brave_search(secret: str | None, *, timeout: float) -> ProviderCredentialTestResult:
    if not secret:
        return ProviderCredentialTestResult(
            False,
            "Brave Search API key required before testing.",
            None,
        )
    resp = knowledge_get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": "qube connectivity test", "count": 1},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": secret,
        },
        timeout=timeout,
    )
    if 200 <= resp.status_code < 300:
        return ProviderCredentialTestResult(
            True,
            "Brave Search API connection succeeded.",
            resp.status_code,
        )
    if resp.status_code in (401, 403):
        return ProviderCredentialTestResult(
            False,
            "Brave Search API key was rejected (check subscription token).",
            resp.status_code,
        )
    return ProviderCredentialTestResult(
        False,
        f"Brave Search API returned HTTP {resp.status_code}.",
        resp.status_code,
    )
