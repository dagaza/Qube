"""SearXNG setup wizard helpers — detect, probe, and scan local instances."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

from core.knowledge.discovery.searxng import search_searxng

logger = logging.getLogger("Qube.Knowledge.Discovery.SearXNGWizard")

LOCAL_SEARXNG_CANDIDATES: tuple[str, ...] = (
    "http://127.0.0.1:8080",
    "http://localhost:8080",
    "http://127.0.0.1:8888",
    "http://localhost:8888",
    "http://127.0.0.1:8081",
    "http://localhost:8081",
)

SEARXNG_DOCKER_RUN_HINT = (
    "docker run -d --name searxng -p 8080:8080 searxng/searxng"
)

_PROBE_QUERY = "qube connectivity test"
_DEFAULT_SCAN_TIMEOUT = 2.5
_DEFAULT_PROBE_TIMEOUT = 8.0


@dataclass(frozen=True)
class SearXNGProbeResult:
    base_url: str
    ok: bool
    message: str
    http_status: int | None = None
    result_count: int = 0


def normalize_searxng_base_url(url: str) -> str:
    """Normalize user-entered SearXNG base URL."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if not parsed.netloc:
        return ""
    scheme = (parsed.scheme or "http").lower()
    if scheme not in {"http", "https"}:
        return ""
    path = (parsed.path or "").rstrip("/")
    return f"{scheme}://{parsed.netloc}{path}".rstrip("/")


def docker_cli_available() -> bool:
    """Return True when the Docker CLI responds to ``docker info``."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker_searxng_container_running() -> bool:
    """Best-effort check for a running container whose name includes ``searxng``."""
    if not docker_cli_available():
        return False
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return any("searxng" in line.lower() for line in result.stdout.splitlines())


def probe_searxng_base_url(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
) -> SearXNGProbeResult:
    """Run a minimal JSON search against a SearXNG instance."""
    normalized = normalize_searxng_base_url(base_url)
    if not normalized:
        return SearXNGProbeResult(
            base_url=(base_url or "").strip(),
            ok=False,
            message="Enter a valid http(s) base URL (for example http://127.0.0.1:8080).",
        )

    rows, inspection = search_searxng(
        _PROBE_QUERY,
        max_results=3,
        timeout=timeout,
        base_url=normalized,
        api_key=api_key,
    )
    response_kind = str(inspection.get("response_kind") or "").strip().lower()
    http_status = inspection.get("http_status")
    status_code = int(http_status) if http_status is not None else None
    parsed_rows = int(inspection.get("parsed_rows") or len(rows))

    if response_kind == "auth_error":
        return SearXNGProbeResult(
            base_url=normalized,
            ok=False,
            message="Authentication failed — check your SearXNG API key.",
            http_status=status_code,
            result_count=parsed_rows,
        )
    if response_kind == "network_error":
        return SearXNGProbeResult(
            base_url=normalized,
            ok=False,
            message="Could not reach the instance — check the URL and that JSON search is enabled.",
            http_status=status_code,
            result_count=parsed_rows,
        )
    if response_kind == "serp":
        return SearXNGProbeResult(
            base_url=normalized,
            ok=True,
            message=f"Connected — received {parsed_rows} result(s).",
            http_status=status_code,
            result_count=parsed_rows,
        )
    if response_kind == "no_results":
        return SearXNGProbeResult(
            base_url=normalized,
            ok=True,
            message="Connected — instance responded but returned no parseable results.",
            http_status=status_code,
            result_count=0,
        )
    return SearXNGProbeResult(
        base_url=normalized,
        ok=False,
        message="Unexpected response from SearXNG instance.",
        http_status=status_code,
        result_count=parsed_rows,
    )


def _candidate_urls(*, include_configured: bool = True) -> list[str]:
    from core.app_settings import get_discovery_searxng_base_url

    seen: set[str] = set()
    ordered: list[str] = []
    if include_configured:
        configured = normalize_searxng_base_url(get_discovery_searxng_base_url())
        if configured:
            seen.add(configured)
            ordered.append(configured)
    for candidate in LOCAL_SEARXNG_CANDIDATES:
        normalized = normalize_searxng_base_url(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def scan_local_searxng_candidates(
    *,
    api_key: str | None = None,
    timeout: float = _DEFAULT_SCAN_TIMEOUT,
) -> list[SearXNGProbeResult]:
    """Probe common local SearXNG URLs; returns successes only, fastest first."""
    hits: list[SearXNGProbeResult] = []
    for candidate in _candidate_urls():
        result = probe_searxng_base_url(candidate, api_key=api_key, timeout=timeout)
        if result.ok:
            hits.append(result)
    return hits
