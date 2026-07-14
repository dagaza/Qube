"""Access badge and hint derivation for Settings → Knowledge live source rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.knowledge.adapters.catalog import AdapterCatalogEntry, get_adapter_entry
from core.knowledge.credentials import CredentialMode, resolve_credential
from core.knowledge.provider_credentials import (
    get_provider_credential_spec,
    provider_has_implemented_adapter,
    provider_id_for_adapter,
)
from core.knowledge.provider_status import provider_quota_hint

AccessBadge = Literal[
    "free",
    "optional_key",
    "key_required",
    "connected",
    "env_override",
    "coming_soon",
]

_BADGE_LABELS: dict[AccessBadge, str] = {
    "free": "Free",
    "optional_key": "Optional key",
    "key_required": "Key required",
    "connected": "Connected",
    "env_override": "Env override",
    "coming_soon": "Coming soon",
}


@dataclass(frozen=True)
class SourceAccessSummary:
    badge: AccessBadge
    badge_label: str
    hint: str | None
    configure_available: bool
    provider_id: str | None

    @property
    def needs_setup(self) -> bool:
        """True when the source may benefit from or requires user credential setup."""
        return self.badge in {"optional_key", "key_required", "coming_soon"}


def source_needs_setup(entry: AdapterCatalogEntry) -> bool:
    """Whether a catalog entry should appear when the setup-only filter is on."""
    return summarize_source_access(entry).needs_setup


def _sibling_key_hint(adapter_id: str, provider_id: str) -> str | None:
    spec = get_provider_credential_spec(provider_id)
    if spec is None or len(spec.adapter_ids) <= 1:
        return None
    aid = (adapter_id or "").strip().lower()
    siblings = spec.adapter_ids
    if aid == siblings[0]:
        return None
    primary = get_adapter_entry(siblings[0])
    if primary is None:
        return None
    return f"Same key as {primary.label}"


def summarize_source_access(entry: AdapterCatalogEntry) -> SourceAccessSummary:
    """Derive row badge, hint, and Configure availability for one catalog entry."""
    if not entry.implemented:
        provider_id = provider_id_for_adapter(entry.id)
        spec = get_provider_credential_spec(provider_id) if provider_id else None
        configure = (
            provider_id is not None
            and spec is not None
            and (provider_has_implemented_adapter(spec) or spec.key_required)
        )
        return SourceAccessSummary(
            badge="coming_soon",
            badge_label=_BADGE_LABELS["coming_soon"],
            hint=None,
            configure_available=configure,
            provider_id=provider_id,
        )

    provider_id = provider_id_for_adapter(entry.id)
    if provider_id is None:
        return SourceAccessSummary(
            badge="free",
            badge_label=_BADGE_LABELS["free"],
            hint=None,
            configure_available=False,
            provider_id=None,
        )

    spec = get_provider_credential_spec(provider_id)
    if spec is None or not provider_has_implemented_adapter(spec):
        return SourceAccessSummary(
            badge="free",
            badge_label=_BADGE_LABELS["free"],
            hint=None,
            configure_available=False,
            provider_id=provider_id,
        )

    cred = resolve_credential(provider_id)
    sibling_hint = _sibling_key_hint(entry.id, provider_id)

    if cred.mode == CredentialMode.ENV_KEY:
        return SourceAccessSummary(
            badge="env_override",
            badge_label=_BADGE_LABELS["env_override"],
            hint=sibling_hint,
            configure_available=True,
            provider_id=provider_id,
        )

    if cred.mode == CredentialMode.USER_KEY:
        return SourceAccessSummary(
            badge="connected",
            badge_label=_BADGE_LABELS["connected"],
            hint=sibling_hint,
            configure_available=True,
            provider_id=provider_id,
        )

    if spec.key_required and not cred.secret:
        return SourceAccessSummary(
            badge="key_required",
            badge_label=_BADGE_LABELS["key_required"],
            hint=sibling_hint,
            configure_available=True,
            provider_id=provider_id,
        )

    if spec.supports_free_api_key:
        limit_hint = provider_quota_hint(provider_id)
        hint = sibling_hint or limit_hint
        return SourceAccessSummary(
            badge="optional_key",
            badge_label=_BADGE_LABELS["optional_key"],
            hint=hint,
            configure_available=True,
            provider_id=provider_id,
        )

    return SourceAccessSummary(
        badge="free",
        badge_label=_BADGE_LABELS["free"],
        hint=sibling_hint,
        configure_available=True,
        provider_id=provider_id,
    )
