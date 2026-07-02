"""Unified knowledge provider credential resolution and storage (Slice 9)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from core.knowledge.provider_credentials import get_provider_credential_spec

# Conservative NCBI E-utilities targets (§3); consumed by host scheduler.
NCBI_RATE_PER_SEC_ANONYMOUS = 2.5
NCBI_RATE_PER_SEC_WITH_KEY = 8.0

# Legacy alias used by courtlistener env docs.
_COURTLISTENER_ENV_ALIASES = ("QUBE_COURTLISTENER_API_TOKEN", "QUBE_COURTLISTENER_TOKEN")
_FRED_ENV_ALIASES = ("QUBE_FRED_API_KEY", "FRED_API_KEY")
_ALPHA_VANTAGE_ENV_ALIASES = ("QUBE_ALPHA_VANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY")
_CANLII_ENV_ALIASES = ("QUBE_CANLII_API_KEY", "CANLII_API_KEY")
_NOAA_ENV_ALIASES = ("QUBE_NOAA_API_TOKEN", "NOAA_TOKEN")
_EBSCO_EDS_ENV_ALIASES = ("QUBE_EBSCO_EDS_PASSWORD",)
_BLOOMBERG_ENV_ALIASES = ("QUBE_BLOOMBERG_API_URL",)


class CredentialMode(str, Enum):
    ANONYMOUS = "anonymous"
    ENV_KEY = "env_key"
    USER_KEY = "user_key"
    FIXTURE = "fixture"


@dataclass(frozen=True)
class KnowledgeCredentialProfile:
    """Static resolver profile for one provider (from ``provider_credentials``)."""

    provider_id: str
    env_var: str | None
    settings_field: str = "api_key"


@dataclass(frozen=True)
class CredentialBundle:
    """Resolved credential for outbound knowledge HTTP (never log ``secret``)."""

    provider_id: str
    mode: CredentialMode
    secret: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        """Alias for ``secret`` (Slice 2 adapter compatibility)."""
        return self.secret

    def __repr__(self) -> str:
        masked = "***" if self.secret else None
        return (
            f"CredentialBundle(provider_id={self.provider_id!r}, "
            f"mode={self.mode!r}, secret={masked!r})"
        )


def knowledge_fixtures_mode() -> bool:
    return os.environ.get("QUBE_KNOWLEDGE_FIXTURES", "").strip() == "1"


def normalize_provider_credentials(raw: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Normalize stored settings to ``{provider_id: {api_key: ...}}``."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, dict[str, str]] = {}
    for provider_id, row in raw.items():
        pid = str(provider_id or "").strip().lower()
        if not pid or get_provider_credential_spec(pid) is None:
            continue
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("api_key") or "").strip()
        if key:
            out[pid] = {"api_key": key}
    return out


def _profile_for(provider_id: str) -> KnowledgeCredentialProfile | None:
    spec = get_provider_credential_spec(provider_id)
    if spec is None:
        return None
    return KnowledgeCredentialProfile(
        provider_id=spec.provider_id,
        env_var=spec.env_var,
    )


def _env_secret(profile: KnowledgeCredentialProfile) -> str | None:
    if not profile.env_var:
        return None
    key = os.environ.get(profile.env_var, "").strip()
    if key:
        return key
    if profile.provider_id == "courtlistener":
        for alias in _COURTLISTENER_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    if profile.provider_id == "fred":
        for alias in _FRED_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    if profile.provider_id == "alpha_vantage":
        for alias in _ALPHA_VANTAGE_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    if profile.provider_id == "canlii":
        for alias in _CANLII_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    if profile.provider_id == "noaa":
        for alias in _NOAA_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    if profile.provider_id == "ebsco_eds":
        for alias in _EBSCO_EDS_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    if profile.provider_id == "bloomberg":
        for alias in _BLOOMBERG_ENV_ALIASES:
            if alias == profile.env_var:
                continue
            alt = os.environ.get(alias, "").strip()
            if alt:
                return alt
    return None


def _user_secret(provider_id: str) -> str | None:
    from core.app_settings import get_knowledge_provider_credentials

    stored = get_knowledge_provider_credentials().get(provider_id, {})
    if not isinstance(stored, dict):
        return None
    key = str(stored.get("api_key") or "").strip()
    return key or None


def resolve_credential(provider_id: str) -> CredentialBundle:
    """
    Resolve provider credentials: env override → user settings → anonymous.

    Fixture mode (``QUBE_KNOWLEDGE_FIXTURES=1``) always returns anonymous with no secret.
    """
    pid = (provider_id or "").strip().lower()
    profile = _profile_for(pid)
    if profile is None:
        return CredentialBundle(provider_id=pid, mode=CredentialMode.ANONYMOUS)

    if knowledge_fixtures_mode():
        return CredentialBundle(
            provider_id=pid,
            mode=CredentialMode.FIXTURE,
            metadata={"fixtures": True},
        )

    env_secret = _env_secret(profile)
    if env_secret:
        return CredentialBundle(
            provider_id=pid,
            mode=CredentialMode.ENV_KEY,
            secret=env_secret,
            metadata={"source": "env"},
        )

    user_secret = _user_secret(pid)
    if user_secret:
        return CredentialBundle(
            provider_id=pid,
            mode=CredentialMode.USER_KEY,
            secret=user_secret,
            metadata={"source": "settings"},
        )

    return CredentialBundle(provider_id=pid, mode=CredentialMode.ANONYMOUS)


def set_provider_api_key(provider_id: str, api_key: str | None) -> dict[str, dict[str, str]]:
    """Persist one provider key in settings; returns normalized store."""
    from core.app_settings import (
        get_knowledge_provider_credentials,
        set_knowledge_provider_credentials,
    )

    pid = (provider_id or "").strip().lower()
    if get_provider_credential_spec(pid) is None:
        return get_knowledge_provider_credentials()

    merged = dict(get_knowledge_provider_credentials())
    text = str(api_key or "").strip()
    if text:
        merged[pid] = {"api_key": text}
    else:
        merged.pop(pid, None)
    set_knowledge_provider_credentials(merged)
    return merged


def clear_provider_api_key(provider_id: str) -> dict[str, dict[str, str]]:
    return set_provider_api_key(provider_id, None)


def ncbi_rate_per_sec() -> float:
    cred = resolve_credential("ncbi")
    if cred.secret:
        return NCBI_RATE_PER_SEC_WITH_KEY
    return NCBI_RATE_PER_SEC_ANONYMOUS


def credential_mode_label(provider_id: str) -> str:
    return resolve_credential(provider_id).mode.value


def connection_mode_display(provider_id: str) -> str:
    """User-facing status line for Settings → Provider credentials."""
    from core.knowledge.provider_credentials import (
        get_provider_credential_spec,
        provider_has_implemented_adapter,
    )

    pid = (provider_id or "").strip().lower()
    spec = get_provider_credential_spec(pid)
    cred = resolve_credential(pid)

    if spec is not None and spec.key_required:
        if not provider_has_implemented_adapter(spec):
            return "API key required — source not available yet"
        if not cred.secret:
            return "API key required"

    if cred.mode == CredentialMode.ENV_KEY:
        return "Env override (development)"
    if cred.mode == CredentialMode.USER_KEY:
        return "Connected with your API key"
    if cred.mode == CredentialMode.FIXTURE:
        return "Fixture mode (no live credentials)"
    if spec is not None:
        return spec.anonymous_summary
    return "Anonymous access"


def env_override_active(provider_id: str) -> bool:
    return resolve_credential(provider_id).mode == CredentialMode.ENV_KEY

