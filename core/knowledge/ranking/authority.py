"""Authority tiers for web and encyclopedia sources."""

from __future__ import annotations

import re
from urllib.parse import urlparse

TIER_WIKIPEDIA = 0.95
TIER_GOV = 0.88
TIER_EDU = 0.82
TIER_DEFAULT = 0.35

_WIKIPEDIA_HOSTS = frozenset(
    {
        "wikipedia.org",
        "www.wikipedia.org",
        "en.wikipedia.org",
        "wikimedia.org",
    }
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def is_wikipedia_url(url: str | None) -> bool:
    host = _host(url or "")
    if not host:
        return False
    return host in _WIKIPEDIA_HOSTS or host.endswith(".wikipedia.org")


# Plan S8 (D7): national government domains beyond the US ``.gov``.
# ``gov.<cc>`` / ``gouv.<cc>`` / ``gob.<cc>`` / ``govt.<cc>`` / ``go.<cc>``
# second-level zones are reserved for government bodies in the countries that
# use them; the named hosts are government portals outside those zones.
_GOV_SECOND_LEVEL = re.compile(r"\.(?:gov|gouv|gob|govt|go|gv)\.[a-z]{2}$")
_GOV_HOST_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".gc.ca",
    ".canada.ca",
    ".admin.ch",
    ".bund.de",
    ".nhs.uk",
    ".nhs.scot",
    ".overheid.nl",
    ".government.nl",
    ".gov.scot",
    ".gov.wales",
)
_GOV_HOSTS = frozenset({"canada.ca", "gov.uk", "government.nl", "bund.de", "admin.ch", "nhs.uk"})
# National statistics offices and public-health agencies on generic domains
# (curated; a host or any subdomain of it). Agencies inside a government
# zone (``abs.gov.au``, ``stats.govt.nz``) are already covered above.
_GOV_AGENCY_HOSTS = frozenset(
    {
        # statistics
        "insee.fr", "destatis.de", "istat.it", "ine.es", "cbs.nl", "scb.se", "ssb.no",
        "stat.fi", "dst.dk", "cso.ie", "statistik.at", "ine.pt", "stat.gov.pl", "ksh.hu",
        "insse.ro", "czso.cz",
        # public health / medicines regulators
        "rki.de", "rivm.nl", "santepubliquefrance.fr", "has-sante.fr", "ansm.sante.fr",
        "folkhalsomyndigheten.se", "fhi.no", "thl.fi", "ssi.dk", "iss.it", "aemps.gob.es",
        "bfarm.de", "pei.de", "nice.org.uk",
    }
)
# Intergovernmental organisations: the reserved ``.int`` TLD (who.int, wmo.int,
# nato.int) plus UN-system, EU and multilateral bodies on other TLDs.
_INTERGOV_HOST_SUFFIXES: tuple[str, ...] = (
    ".int",
    ".europa.eu",
    ".un.org",
    ".unesco.org",
    ".unicef.org",
    ".undp.org",
    ".unep.org",
    ".unhcr.org",
    ".fao.org",
    ".ilo.org",
    ".imf.org",
    ".iaea.org",
    ".worldbank.org",
    ".oecd.org",
    ".wto.org",
    ".ipcc.ch",
    ".unfccc.int",
    ".bis.org",
)
_INTERGOV_HOSTS = frozenset(s.lstrip(".") for s in _INTERGOV_HOST_SUFFIXES if s != ".int")
# Universities: ``.edu`` (US) and ``.edu.<cc>`` / ``.ac.<cc>`` academic zones.
_EDU_ZONE = re.compile(r"\.(?:edu|ac)\.[a-z]{2}$")


def _host_matches(host: str, hosts: frozenset[str], suffixes: tuple[str, ...]) -> bool:
    return host in hosts or any(host.endswith(sfx) for sfx in suffixes)


def is_gov_url(url: str | None) -> bool:
    """US ``.gov``, ``.gov.uk`` and other national government domains (S8)."""
    host = _host(url or "")
    if not host:
        return False
    return bool(
        _GOV_SECOND_LEVEL.search(host)
        or _host_matches(host, _GOV_HOSTS, _GOV_HOST_SUFFIXES)
        or _host_matches(host, _GOV_AGENCY_HOSTS, tuple(f".{h}" for h in _GOV_AGENCY_HOSTS))
    )


def is_intergovernmental_url(url: str | None) -> bool:
    """WHO, the UN system, the EU, the OECD, the World Bank, … (S8)."""
    host = _host(url or "")
    if not host:
        return False
    return _host_matches(host, _INTERGOV_HOSTS, _INTERGOV_HOST_SUFFIXES)


def is_edu_url(url: str | None) -> bool:
    host = _host(url or "")
    if not host:
        return False
    return bool(host.endswith(".edu") or _EDU_ZONE.search(host))


def is_legacy_allowlisted_url(url: str | None) -> bool:
    """The pre-S8 allowlist: Wikipedia, ``.gov`` / ``.gov.uk``, ``.edu`` / ``.ac.uk``."""
    host = _host(url or "")
    if not host:
        return False
    return (
        is_wikipedia_url(url)
        or host.endswith((".gov", ".gov.uk", ".edu", ".ac.uk"))
    )


def is_allowlisted_url(url: str | None, *, widened: bool = True) -> bool:
    """Trusted-knowledge allowlist: Wikipedia, government, intergovernmental and academic domains.

    ``widened=False`` is the pre-S8 list, for the legacy path without the
    relevance reranker: its token check accepts "WHO guidelines" pages for any
    "WHO guidance on …" question, so more domains there means more noise.
    """
    if not widened:
        return is_legacy_allowlisted_url(url)
    return (
        is_wikipedia_url(url)
        or is_gov_url(url)
        or is_intergovernmental_url(url)
        or is_edu_url(url)
    )


def authority_score_for_url(url: str | None) -> float:
    if is_wikipedia_url(url):
        return TIER_WIKIPEDIA
    if is_gov_url(url) or is_intergovernmental_url(url):
        return TIER_GOV
    if is_edu_url(url):
        return TIER_EDU
    return TIER_DEFAULT


def tier_label_for_url(url: str | None) -> str:
    if is_wikipedia_url(url):
        return "wikipedia"
    if is_gov_url(url):
        return "government"
    if is_intergovernmental_url(url):
        return "intergovernmental"
    if is_edu_url(url):
        return "academic"
    return "web"
