"""Resolve optional knowledge provider credentials (delegates to ``credentials``)."""

from __future__ import annotations

from typing import Any, Mapping

from core.knowledge.credentials import (
    CredentialBundle,
    CredentialMode,
    KnowledgeCredentialProfile,
    clear_provider_api_key,
    credential_mode_label,
    ncbi_rate_per_sec,
    resolve_credential,
    set_provider_api_key,
)
from core.knowledge.provider_credentials import provider_id_for_adapter

# Re-export for backward compatibility with Slice 2 imports.
NCBI_RATE_PER_SEC_ANONYMOUS = 2.5
NCBI_RATE_PER_SEC_WITH_KEY = 8.0

# Slice 2 name retained for adapters and tests.
ResolvedCredential = CredentialBundle


def resolve(provider_id: str) -> CredentialBundle:
    """Resolve credentials for a provider (env → user settings → anonymous)."""
    return resolve_credential(provider_id)


def resolve_for_adapter(adapter_id: str) -> CredentialBundle:
    """Resolve credentials for an adapter via its provider mapping."""
    pid = provider_id_for_adapter(adapter_id) or (adapter_id or "").strip().lower()
    return resolve_credential(pid)


def api_key_query_params(provider_id: str) -> dict[str, str]:
    """Return ``api_key`` query params when a credential is configured."""
    cred = resolve_credential(provider_id)
    if cred.secret:
        return {"api_key": cred.secret}
    return {}


def merge_query_params(
    params: Mapping[str, Any] | None,
    provider_id: str,
) -> dict[str, Any]:
    """Merge provider ``api_key`` into an existing params mapping."""
    merged = dict(params or {})
    merged.update(api_key_query_params(provider_id))
    return merged


def authorization_token(provider_id: str) -> str | None:
    """Bearer-style token for header auth (e.g. CourtListener)."""
    cred = resolve_credential(provider_id)
    return cred.secret


def http_basic_auth(provider_id: str) -> tuple[str, str] | None:
    """HTTP Basic auth tuple for providers like Companies House (key as username)."""
    token = authorization_token(provider_id)
    if token:
        return (token, "")
    return None


__all__ = [
    "CredentialBundle",
    "CredentialMode",
    "KnowledgeCredentialProfile",
    "NCBI_RATE_PER_SEC_ANONYMOUS",
    "NCBI_RATE_PER_SEC_WITH_KEY",
    "ResolvedCredential",
    "api_key_query_params",
    "authorization_token",
    "clear_provider_api_key",
    "credential_mode_label",
    "http_basic_auth",
    "merge_query_params",
    "ncbi_rate_per_sec",
    "provider_id_for_adapter",
    "resolve",
    "resolve_credential",
    "resolve_for_adapter",
    "set_provider_api_key",
]
