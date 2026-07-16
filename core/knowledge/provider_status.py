"""Aggregate knowledge provider status for Settings source status panel (Slice 11)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from core.knowledge.credentials import CredentialMode, resolve_credential
from core.knowledge.host_scheduler import host_health_snapshot
from core.knowledge.http_metrics import global_http_summary
from core.knowledge.provider_credentials import (
    get_provider_credential_spec,
    list_provider_credential_specs,
    provider_has_implemented_adapter,
)

_PROVIDER_METRICS_HOSTS: dict[str, tuple[str, ...]] = {
    "openalex": ("api.openalex.org",),
    "ncbi": ("ncbi",),
    "courtlistener": ("www.courtlistener.com",),
    "semantic_scholar": ("api.semanticscholar.org",),
    "nasa_ads": ("api.adsabs.harvard.edu",),
    "fred": ("api.stlouisfed.org",),
    "companies_house": ("api.company-information.service.gov.uk",),
    "alpha_vantage": ("www.alphavantage.co",),
    "canlii": ("api.canlii.org",),
    "noaa": ("www.ncei.noaa.gov",),
    "ebsco_eds": ("eds-api.ebscohost.com",),
    "bloomberg": ("bloomberg",),
    "usda_fdc": ("api.nal.usda.gov",),
    "bls": ("api.bls.gov",),
    "us_census": ("api.census.gov",),
    "nist": ("services.nvd.nist.gov",),
    "ieee_xplore": ("ieeexploreapi.ieee.org",),
    "nice": ("api.nice.org.uk",),
    "fao": ("faostatservices.fao.org",),
    "usda": ("api.ers.usda.gov",),
    "copernicus_cds": ("cds.climate.copernicus.eu",),
    "congress_gov": ("api.congress.gov",),
    "govinfo": ("api.govinfo.gov",),
    "patentsview": ("search.patentsview.org",),
    "epo_ops": ("ops.epo.org",),
    "brave_search": ("api.search.brave.com",),
}

_METRICS_HOST_TO_PROVIDER: dict[str, str] = {}
for _pid, _hosts in _PROVIDER_METRICS_HOSTS.items():
    for _host in _hosts:
        _METRICS_HOST_TO_PROVIDER[_host] = _pid


class ProviderHealth(str, Enum):
    GOOD = "Good"
    DEGRADED = "Degraded"
    UNKNOWN = "Unknown"
    NA = "—"


@dataclass(frozen=True)
class ProviderTestSnapshot:
    ok: bool
    message: str
    tested_at: float


@dataclass(frozen=True)
class ProviderStatus:
    provider_id: str
    label: str
    status: str
    quota_label: str
    health: ProviderHealth
    last_error: str | None = None
    last_used_label: str = "—"
    last_test_label: str = "—"
    resets_at: float | None = None


_latest_http_summary: dict[str, Any] | None = None
_last_test_by_provider: dict[str, ProviderTestSnapshot] = {}


def provider_id_for_metrics_host(metrics_host: str) -> str | None:
    return _METRICS_HOST_TO_PROVIDER.get((metrics_host or "").strip().lower())


def metrics_hosts_for_provider(provider_id: str) -> tuple[str, ...]:
    return _PROVIDER_METRICS_HOSTS.get((provider_id or "").strip().lower(), ())


def record_provider_credential_test(
    provider_id: str,
    *,
    ok: bool,
    message: str,
) -> None:
    pid = (provider_id or "").strip().lower()
    _last_test_by_provider[pid] = ProviderTestSnapshot(
        ok=ok,
        message=message,
        tested_at=time.time(),
    )


def apply_http_summary(summary: Mapping[str, Any] | None) -> None:
    global _latest_http_summary
    if summary:
        _latest_http_summary = dict(summary)


def _active_http_summary() -> dict[str, Any]:
    if _latest_http_summary:
        return _latest_http_summary
    return global_http_summary()


def _format_relative_time(epoch: float | None) -> str:
    if not epoch or epoch <= 0:
        return "—"
    delta = max(0.0, time.time() - epoch)
    if delta < 60:
        return "just now"
    if delta < 3600:
        minutes = int(delta // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if delta < 86400:
        hours = int(delta // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(delta // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def _format_test_label(snapshot: ProviderTestSnapshot | None) -> str:
    if snapshot is None:
        return "—"
    outcome = "OK" if snapshot.ok else "Failed"
    return f"{outcome} ({_format_relative_time(snapshot.tested_at)})"


def _status_badge(provider_id: str) -> str:
    cred = resolve_credential(provider_id)
    spec = get_provider_credential_spec(provider_id)
    if spec is not None and spec.key_required and not provider_has_implemented_adapter(spec):
        return "Not available"
    if cred.mode == CredentialMode.ENV_KEY:
        return "Env override"
    if cred.mode == CredentialMode.USER_KEY:
        return "Connected"
    if spec is not None and spec.key_required and not cred.secret:
        return "Not configured"
    return "Anonymous"


def _quota_label(provider_id: str, summary: Mapping[str, Any]) -> str:
    cred = resolve_credential(provider_id)
    by_host = summary.get("by_host") or {}
    if provider_id == "openalex":
        row = by_host.get("api.openalex.org") or {}
        remaining = row.get("rate_limit_remaining")
        if remaining is not None:
            try:
                return f"{float(remaining):.0f} requests remaining"
            except (TypeError, ValueError):
                pass
        if cred.secret:
            return "~$1/day policy"
        return "~$0.10/day"
    if provider_id == "ncbi":
        return "10 req/sec policy" if cred.secret else "3 req/sec"
    if provider_id == "courtlistener":
        if cred.secret:
            return "Connected token"
        return "Anonymous (limited)"
    if provider_id == "semantic_scholar":
        return "1 req/sec policy" if cred.secret else "Key required"
    if provider_id == "nasa_ads":
        row = by_host.get("api.adsabs.harvard.edu") or {}
        remaining = row.get("rate_limit_remaining")
        if remaining is not None:
            try:
                return f"{float(remaining):.0f} requests remaining"
            except (TypeError, ValueError):
                pass
        return "Token required" if not cred.secret else "Daily token policy"
    if provider_id == "fred":
        return "120 req/min policy" if cred.secret else "Key required"
    if provider_id == "companies_house":
        return "600 req / 5 min policy" if cred.secret else "Key required"
    if provider_id == "alpha_vantage":
        return "5 req/min free tier" if cred.secret else "Key required"
    if provider_id == "canlii":
        return "2 req/sec policy" if cred.secret else "Key required"
    if provider_id == "noaa":
        return "5 req/sec token policy" if cred.secret else "Token required"
    if provider_id == "ebsco_eds":
        return "Institutional EDS policy" if cred.secret else "Credentials required"
    if provider_id == "bloomberg":
        return "Enterprise bridge" if cred.secret else "URL required"
    spec = get_provider_credential_spec(provider_id)
    if spec is not None and spec.key_required and not cred.secret:
        return "—"
    return "Policy limits apply"


def _host_health_rows(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    health = summary.get("host_health")
    if isinstance(health, Mapping) and health:
        return health
    return host_health_snapshot()


def _health_for_provider(provider_id: str, summary: Mapping[str, Any]) -> ProviderHealth:
    spec = get_provider_credential_spec(provider_id)
    cred = resolve_credential(provider_id)
    if (
        spec is not None
        and spec.key_required
        and not cred.secret
        and provider_has_implemented_adapter(spec)
    ):
        return ProviderHealth.NA

    hosts = metrics_hosts_for_provider(provider_id)
    if not hosts:
        return ProviderHealth.UNKNOWN

    health_rows = _host_health_rows(summary)
    for host in hosts:
        row = health_rows.get(host) or {}
        if isinstance(row, Mapping) and row.get("state") == "open":
            return ProviderHealth.DEGRADED

    by_host = summary.get("by_host") or {}
    for host in hosts:
        row = by_host.get(host) or {}
        if not isinstance(row, Mapping):
            continue
        if int(row.get("429") or 0) > 0 or int(row.get("503") or 0) > 0:
            return ProviderHealth.DEGRADED

    retry_reasons = summary.get("retry_reasons") or []
    if isinstance(retry_reasons, list):
        for host in hosts:
            prefix = f"{host}:"
            for reason in retry_reasons:
                text = str(reason)
                if text.startswith(prefix) and (
                    "budget_exhausted" in text or "circuit_open" in text
                ):
                    return ProviderHealth.DEGRADED

    any_requests = False
    last_used_at = 0.0
    for host in hosts:
        row = by_host.get(host) or {}
        if not isinstance(row, Mapping):
            continue
        if int(row.get("requests") or 0) > 0:
            any_requests = True
        try:
            last_used_at = max(last_used_at, float(row.get("last_request_at") or 0.0))
        except (TypeError, ValueError):
            pass

    if not any_requests:
        return ProviderHealth.UNKNOWN
    return ProviderHealth.GOOD


def _last_used_label(provider_id: str, summary: Mapping[str, Any]) -> str:
    hosts = metrics_hosts_for_provider(provider_id)
    by_host = summary.get("by_host") or {}
    last_used_at = 0.0
    for host in hosts:
        row = by_host.get(host) or {}
        if not isinstance(row, Mapping):
            continue
        try:
            last_used_at = max(last_used_at, float(row.get("last_request_at") or 0.0))
        except (TypeError, ValueError):
            pass
    return _format_relative_time(last_used_at if last_used_at > 0 else None)


def _last_error_for_provider(provider_id: str) -> str | None:
    snapshot = _last_test_by_provider.get(provider_id)
    if snapshot is not None and not snapshot.ok:
        return snapshot.message
    return None


def build_provider_status(provider_id: str, *, summary: Mapping[str, Any] | None = None) -> ProviderStatus:
    pid = (provider_id or "").strip().lower()
    spec = get_provider_credential_spec(pid)
    label = spec.label if spec is not None else pid
    active_summary = summary if summary is not None else _active_http_summary()
    test_snapshot = _last_test_by_provider.get(pid)
    return ProviderStatus(
        provider_id=pid,
        label=label,
        status=_status_badge(pid),
        quota_label=_quota_label(pid, active_summary),
        health=_health_for_provider(pid, active_summary),
        last_error=_last_error_for_provider(pid),
        last_used_label=_last_used_label(pid, active_summary),
        last_test_label=_format_test_label(test_snapshot),
    )


def provider_quota_hint(provider_id: str) -> str | None:
    """Short limit hint for live source rows; None when not useful to display."""
    label = _quota_label((provider_id or "").strip().lower(), _active_http_summary())
    if label in {
        "—",
        "Policy limits apply",
        "Key required",
        "Token required",
        "Credentials required",
        "URL required",
        "Connected token",
    }:
        return None
    return label


def list_provider_status_rows(*, summary: Mapping[str, Any] | None = None) -> list[ProviderStatus]:
    active_summary = summary if summary is not None else _active_http_summary()
    rows: list[ProviderStatus] = []
    for spec in list_provider_credential_specs():
        if not provider_has_implemented_adapter(spec):
            continue
        rows.append(build_provider_status(spec.provider_id, summary=active_summary))
    return rows


def reset_provider_status_state_for_tests() -> None:
    """Clear in-memory status caches (unit tests only)."""
    global _latest_http_summary
    _latest_http_summary = None
    _last_test_by_provider.clear()
