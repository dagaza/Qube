"""Egress policy — SSRF protection for knowledge HTTP requests."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

DEFAULT_MAX_RESPONSE_BYTES = 1_048_576  # 1 MiB
MAX_USER_TIMEOUT_SEC = 30.0
DEFAULT_MAX_RESPONSE_BYTES_USER = 524_288  # 512 KiB


class EgressPolicyError(Exception):
    """URL blocked by egress policy."""


@dataclass(frozen=True)
class EgressPolicy:
    """Per-request or per-source egress constraints."""

    allow_http: bool = False
    allow_localhost: bool = False
    allow_private_network: bool = False
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    local_only: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> EgressPolicy:
        if not isinstance(raw, dict):
            return cls()
        max_bytes = raw.get("max_response_bytes", DEFAULT_MAX_RESPONSE_BYTES)
        try:
            max_bytes = int(max_bytes)
        except (TypeError, ValueError):
            max_bytes = DEFAULT_MAX_RESPONSE_BYTES
        if max_bytes <= 0:
            max_bytes = DEFAULT_MAX_RESPONSE_BYTES
        return cls(
            allow_http=bool(raw.get("allow_http")),
            allow_localhost=bool(raw.get("allow_localhost")),
            allow_private_network=bool(raw.get("allow_private_network")),
            max_response_bytes=max_bytes,
            local_only=bool(raw.get("local_only")),
        )

    @classmethod
    def bloomberg_default(cls) -> EgressPolicy:
        """Bloomberg bridge commonly runs on localhost."""
        return cls(allow_http=True, allow_localhost=True, max_response_bytes=DEFAULT_MAX_RESPONSE_BYTES)

    @classmethod
    def configured_source_default(cls) -> EgressPolicy:
        return cls(
            allow_http=False,
            allow_localhost=False,
            max_response_bytes=DEFAULT_MAX_RESPONSE_BYTES_USER,
        )

    @classmethod
    def local_connector_default(cls) -> EgressPolicy:
        return cls(local_only=True, max_response_bytes=DEFAULT_MAX_RESPONSE_BYTES)


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if addr.is_loopback:
        return True
    if addr.is_link_local:
        return True
    if addr.is_multicast:
        return True
    if addr.is_reserved:
        return True
    if addr.is_private:
        return True
    if addr.is_unspecified:
        return True
    # AWS/GCP metadata endpoints
    if str(addr) == "169.254.169.254":
        return True
    return False


def _resolve_hostname(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise EgressPolicyError(f"Cannot resolve host: {hostname}") from exc
    addrs: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            addrs.append(str(sockaddr[0]))
    return addrs


def _validate_host(hostname: str, policy: EgressPolicy) -> None:
    host = (hostname or "").strip().lower()
    if not host:
        raise EgressPolicyError("Missing hostname")

    if host in {"localhost", "localhost.localdomain"}:
        if not policy.allow_localhost:
            raise EgressPolicyError("Localhost access is not allowed")
        return

    if host.endswith(".local"):
        if not policy.allow_localhost:
            raise EgressPolicyError("Local network hostnames are not allowed")
        return

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None

    if addr is not None:
        if addr.is_loopback and not policy.allow_localhost:
            raise EgressPolicyError("Localhost access is not allowed")
        if addr.is_private and not policy.allow_private_network:
            raise EgressPolicyError("Private network addresses are not allowed")
        if _is_blocked_ip(addr) and not (
            addr.is_loopback and policy.allow_localhost
        ) and not (addr.is_private and policy.allow_private_network):
            raise EgressPolicyError(f"Blocked IP address: {addr}")
        return

    for resolved in _resolve_hostname(host):
        try:
            ip = ipaddress.ip_address(resolved)
        except ValueError:
            continue
        if ip.is_loopback and not policy.allow_localhost:
            raise EgressPolicyError(f"Host {host} resolves to localhost ({ip})")
        if ip.is_private and not policy.allow_private_network:
            raise EgressPolicyError(f"Host {host} resolves to private address ({ip})")
        if _is_blocked_ip(ip) and not (
            ip.is_loopback and policy.allow_localhost
        ) and not (ip.is_private and policy.allow_private_network):
            raise EgressPolicyError(f"Host {host} resolves to blocked address ({ip})")


def validate_url(url: str, policy: EgressPolicy | None = None) -> str:
    """Validate URL against egress policy. Returns normalized URL or raises."""
    pol = policy or EgressPolicy()
    if pol.local_only:
        raise EgressPolicyError("Network egress blocked (local_only policy)")

    text = (url or "").strip()
    if not text:
        raise EgressPolicyError("Empty URL")

    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise EgressPolicyError(f"Unsupported URL scheme: {scheme or '(none)'}")
    if scheme == "http" and not pol.allow_http:
        raise EgressPolicyError("HTTP is not allowed; use HTTPS")

    hostname = parsed.hostname
    if not hostname:
        raise EgressPolicyError("Missing hostname in URL")

    _validate_host(hostname, pol)
    return text


def cap_timeout(timeout: float | None, *, policy: EgressPolicy | None = None) -> float:
    """Cap request timeout to platform maximum."""
    _ = policy
    try:
        value = float(timeout if timeout is not None else 10.0)
    except (TypeError, ValueError):
        value = 10.0
    return min(max(0.1, value), MAX_USER_TIMEOUT_SEC)
