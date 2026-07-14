"""Process-wide per-host rate limiting for knowledge adapter HTTP calls."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from core.knowledge.credential_resolver import ncbi_rate_per_sec

PolicyKind = Literal["token_bucket", "serialized_interval"]
CircuitState = Literal["closed", "open", "half_open"]

NCBI_HOSTS = frozenset(
    {
        "eutils.ncbi.nlm.nih.gov",
        "pubchem.ncbi.nlm.nih.gov",
    }
)

DEFAULT_TOKEN_RATE_PER_SEC = 5.0
DEFAULT_TOKEN_BURST = 10.0
DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_FAILURE_WINDOW_SEC = 60.0
DEFAULT_CIRCUIT_OPEN_COOLDOWN_SEC = 300.0
CIRCUIT_FAILURE_CODES = frozenset({429, 502, 503, 504})


class HostCircuitOpenError(Exception):
    """Raised when a host circuit breaker is open (not accepting traffic)."""

    def __init__(self, *, host_key: str) -> None:
        self.host_key = host_key
        super().__init__(f"Circuit open for {host_key}")


@dataclass
class _CircuitBreakerState:
    state: CircuitState = "closed"
    consecutive_failures: int = 0
    last_failure_at: float = 0.0
    opened_at: float = 0.0
    probe_in_flight: bool = False


@dataclass(frozen=True)
class HostPolicySpec:
    kind: PolicyKind
    rate_per_sec: float | None = None
    burst: float | None = None
    min_interval_sec: float | None = None


def circuit_breaker_enabled() -> bool:
    raw = os.getenv("QUBE_CIRCUIT_BREAKER")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def circuit_failure_threshold() -> int:
    raw = os.getenv("QUBE_CIRCUIT_FAILURE_THRESHOLD")
    if raw is None:
        return DEFAULT_CIRCUIT_FAILURE_THRESHOLD
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return DEFAULT_CIRCUIT_FAILURE_THRESHOLD


def circuit_failure_window_sec() -> float:
    raw = os.getenv("QUBE_CIRCUIT_FAILURE_WINDOW_SEC")
    if raw is None:
        return DEFAULT_CIRCUIT_FAILURE_WINDOW_SEC
    try:
        return max(1.0, float(str(raw).strip()))
    except ValueError:
        return DEFAULT_CIRCUIT_FAILURE_WINDOW_SEC


def circuit_open_cooldown_sec() -> float:
    raw = os.getenv("QUBE_CIRCUIT_OPEN_COOLDOWN_SEC")
    if raw is None:
        return DEFAULT_CIRCUIT_OPEN_COOLDOWN_SEC
    try:
        return max(1.0, float(str(raw).strip()))
    except ValueError:
        return DEFAULT_CIRCUIT_OPEN_COOLDOWN_SEC


# Conservative v1 targets (plan §3).
_HOST_POLICIES: dict[str, HostPolicySpec] = {
    "api.openalex.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=8.0, burst=16.0
    ),
    "ncbi": HostPolicySpec(kind="token_bucket", rate_per_sec=None, burst=8.0),
    "export.arxiv.org": HostPolicySpec(
        kind="serialized_interval", min_interval_sec=3.5
    ),
    "inspirehep.net": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.5, burst=5.0
    ),
    "en.wikipedia.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=3.0, burst=6.0
    ),
    "www.courtlistener.com": HostPolicySpec(
        kind="token_bucket", rate_per_sec=4.0 / 60.0, burst=2.0
    ),
    "dblp.org": HostPolicySpec(kind="token_bucket", rate_per_sec=3.0, burst=6.0),
    "www.ebi.ac.uk": HostPolicySpec(kind="token_bucket", rate_per_sec=3.0, burst=6.0),
    "api.econbiz.de": HostPolicySpec(kind="token_bucket", rate_per_sec=3.0, burst=6.0),
    "www.sec.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "data.sec.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api.crossref.org": HostPolicySpec(kind="token_bucket", rate_per_sec=3.0, burst=6.0),
    "api.semanticscholar.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=1.0, burst=2.0
    ),
    "api.adsabs.harvard.edu": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "api.osf.io": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api.stlouisfed.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "api.company-information.service.gov.uk": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "www.alphavantage.co": HostPolicySpec(
        kind="serialized_interval", min_interval_sec=12.0
    ),
    "publications.europa.eu": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "api.canlii.org": HostPolicySpec(
        kind="serialized_interval", min_interval_sec=0.5
    ),
    "www.bailii.org": HostPolicySpec(
        kind="serialized_interval", min_interval_sec=2.0
    ),
    "www.ncei.noaa.gov": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "cmr.earthdata.nasa.gov": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "eds-api.ebscohost.com": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "clinicaltrials.gov": HostPolicySpec(
        kind="token_bucket", rate_per_sec=3.0, burst=6.0
    ),
    "api.fda.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api.worldbank.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "ec.europa.eu": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "pubs.usgs.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api.nal.usda.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "services.nvd.nist.gov": HostPolicySpec(
        kind="serialized_interval", min_interval_sec=6.0
    ),
    "datatracker.ietf.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "api.bls.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api.census.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "ieeexploreapi.ieee.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=1.0, burst=2.0
    ),
    "sdmx.oecd.org": HostPolicySpec(kind="token_bucket", rate_per_sec=1.0, burst=2.0),
    "ghoapi.azureedge.net": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "tools.cdc.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "data.cdc.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api.nice.org.uk": HostPolicySpec(kind="token_bucket", rate_per_sec=1.0, burst=2.0),
    "zenodo.org": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "api2.openreview.net": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "verbatim.krlabs.eu": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "rest.uniprot.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=3.0, burst=6.0
    ),
    "api.congress.gov": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "api.govinfo.gov": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "www.legislation.gov.uk": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "search.patentsview.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "ops.epo.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=1.0, burst=2.0
    ),
    "search.rcsb.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=2.0, burst=4.0
    ),
    "data.rcsb.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=3.0, burst=6.0
    ),
    "faostatservices.fao.org": HostPolicySpec(
        kind="token_bucket", rate_per_sec=1.0, burst=2.0
    ),
    "api.ers.usda.gov": HostPolicySpec(kind="token_bucket", rate_per_sec=2.0, burst=4.0),
    "cds.climate.copernicus.eu": HostPolicySpec(
        kind="token_bucket", rate_per_sec=1.0, burst=2.0
    ),
}


def scheduler_key_for_host(hostname: str) -> str:
    """Map a request hostname to a scheduler bucket key."""
    host = (hostname or "").strip().lower()
    if host in NCBI_HOSTS:
        return "ncbi"
    return host or "unknown"


def metrics_host_for(hostname: str) -> str:
    """Normalize hostname for HTTP metrics grouping."""
    return scheduler_key_for_host(hostname)


def policy_for_host(hostname: str) -> HostPolicySpec:
    key = scheduler_key_for_host(hostname)
    if key in _HOST_POLICIES:
        return _HOST_POLICIES[key]
    return HostPolicySpec(
        kind="token_bucket",
        rate_per_sec=DEFAULT_TOKEN_RATE_PER_SEC,
        burst=DEFAULT_TOKEN_BURST,
    )


class TokenBucket:
    """Thread-safe token bucket limiter."""

    def __init__(self, *, rate_per_sec: float, burst: float) -> None:
        self._rate = max(rate_per_sec, 0.001)
        self._burst = max(burst, 1.0)
        self._tokens = self._burst
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def set_rate(self, rate_per_sec: float) -> None:
        with self._cond:
            self._rate = max(rate_per_sec, 0.001)

    @property
    def rate_per_sec(self) -> float:
        return self._rate

    def _refill_locked(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        if elapsed <= 0:
            return
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self, *, tokens: float = 1.0) -> None:
        need = max(tokens, 0.0)
        with self._cond:
            while True:
                now = time.monotonic()
                self._refill_locked(now)
                if self._tokens >= need:
                    self._tokens -= need
                    return
                deficit = need - self._tokens
                wait_s = deficit / self._rate
                self._cond.wait(timeout=max(wait_s, 0.001))


class SerializedInterval:
    """Strict minimum gap between consecutive requests (arXiv)."""

    def __init__(self, *, min_interval_sec: float) -> None:
        self._min_interval = max(min_interval_sec, 0.0)
        self._last_at = 0.0
        self._lock = threading.Lock()

    @property
    def min_interval_sec(self) -> float:
        return self._min_interval

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_at > 0:
                wait_s = self._min_interval - (now - self._last_at)
                if wait_s > 0:
                    time.sleep(wait_s)
            self._last_at = time.monotonic()


class HostScheduler:
    """Process-wide scheduler keyed by host / shared bucket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, TokenBucket | SerializedInterval] = {}
        self._circuits: dict[str, _CircuitBreakerState] = {}

    def _circuit_key(self, hostname: str) -> str:
        return scheduler_key_for_host(hostname)

    def _get_circuit(self, key: str) -> _CircuitBreakerState:
        with self._lock:
            return self._circuits.setdefault(key, _CircuitBreakerState())

    def _open_circuit_unlocked(
        self,
        key: str,
        circuit: _CircuitBreakerState,
        *,
        now: float,
    ) -> None:
        from core.knowledge.negative_cache import mark_host_negative

        circuit.state = "open"
        circuit.opened_at = now
        circuit.probe_in_flight = False
        mark_host_negative(
            key,
            reason="circuit_open",
            ttl_seconds=int(circuit_open_cooldown_sec()),
        )

    def _check_circuit(self, hostname: str) -> None:
        if not circuit_breaker_enabled():
            return
        key = self._circuit_key(hostname)
        now = time.monotonic()
        with self._lock:
            circuit = self._circuits.setdefault(key, _CircuitBreakerState())
            if circuit.state == "open":
                if (now - circuit.opened_at) >= circuit_open_cooldown_sec():
                    circuit.state = "half_open"
                    circuit.probe_in_flight = False
                    from core.knowledge.negative_cache import clear_host_negative

                    clear_host_negative(key)
                else:
                    raise HostCircuitOpenError(host_key=key)
            if circuit.state == "half_open":
                if circuit.probe_in_flight:
                    raise HostCircuitOpenError(host_key=key)
                circuit.probe_in_flight = True

    def record_outcome(
        self,
        hostname: str,
        status_code: int,
        *,
        budget_exhausted: bool = False,
    ) -> None:
        """Update circuit breaker state from an HTTP response."""
        if not circuit_breaker_enabled() or budget_exhausted:
            return
        key = self._circuit_key(hostname)
        now = time.monotonic()
        with self._lock:
            circuit = self._circuits.setdefault(key, _CircuitBreakerState())
            if 200 <= status_code < 400:
                circuit.consecutive_failures = 0
                circuit.last_failure_at = 0.0
                if circuit.state == "half_open":
                    circuit.state = "closed"
                    circuit.probe_in_flight = False
                    from core.knowledge.negative_cache import clear_host_negative

                    clear_host_negative(key)
                elif circuit.state == "closed":
                    circuit.probe_in_flight = False
                return

            if status_code not in CIRCUIT_FAILURE_CODES and status_code != 0:
                if circuit.state == "half_open":
                    circuit.probe_in_flight = False
                return

            if (
                circuit.last_failure_at > 0
                and (now - circuit.last_failure_at) > circuit_failure_window_sec()
            ):
                circuit.consecutive_failures = 0

            circuit.consecutive_failures += 1
            circuit.last_failure_at = now

            if circuit.state == "half_open":
                self._open_circuit_unlocked(key, circuit, now=now)
                return

            if circuit.consecutive_failures >= circuit_failure_threshold():
                self._open_circuit_unlocked(key, circuit, now=now)

    def ensure_circuit_allows_request(self, hostname: str) -> None:
        """Verify circuit breaker permits a request (raises if open)."""
        self._check_circuit(hostname)

    def host_health_snapshot(self) -> dict[str, dict[str, Any]]:
        """Per-host circuit state for HTTP diagnostics."""
        with self._lock:
            rows = {
                key: {
                    "state": circuit.state,
                    "consecutive_failures": circuit.consecutive_failures,
                }
                for key, circuit in self._circuits.items()
            }
        return dict(sorted(rows.items()))

    def _effective_rate(self, key: str, spec: HostPolicySpec) -> float:
        if key == "ncbi":
            return ncbi_rate_per_sec()
        return float(spec.rate_per_sec or DEFAULT_TOKEN_RATE_PER_SEC)

    def _get_limiter(self, key: str, spec: HostPolicySpec) -> TokenBucket | SerializedInterval:
        with self._lock:
            limiter = self._buckets.get(key)
            if limiter is None:
                if spec.kind == "serialized_interval":
                    limiter = SerializedInterval(
                        min_interval_sec=float(spec.min_interval_sec or 0.0)
                    )
                else:
                    limiter = TokenBucket(
                        rate_per_sec=self._effective_rate(key, spec),
                        burst=float(spec.burst or DEFAULT_TOKEN_BURST),
                    )
                self._buckets[key] = limiter
            elif key == "ncbi" and isinstance(limiter, TokenBucket):
                limiter.set_rate(self._effective_rate(key, spec))
            return limiter

    def acquire(self, hostname: str) -> None:
        """Block until the host may send a request (rate limit only)."""
        key = scheduler_key_for_host(hostname)
        spec = policy_for_host(hostname)
        limiter = self._get_limiter(key, spec)
        limiter.acquire()

    def reset(self) -> None:
        """Clear limiter state (tests only)."""
        with self._lock:
            self._buckets.clear()
            self._circuits.clear()


_scheduler = HostScheduler()


def get_host_scheduler() -> HostScheduler:
    return _scheduler


def reset_host_scheduler() -> None:
    """Reset scheduler state (tests only)."""
    _scheduler.reset()


def host_health_snapshot() -> dict[str, dict[str, Any]]:
    return _scheduler.host_health_snapshot()
