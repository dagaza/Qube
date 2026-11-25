"""Leave out adapters that need an API key nobody configured (plan S22, D35).

Key-required adapters (IEEE Xplore, NICE, EPO OPS, FRED, CanLII, ...) return
``[]`` without a key, which a trace could not tell apart from a search that
found nothing. The pipelines now drop them before any task is built and name
each one in the diagnostics (``adapter_skips``). An adapter is gated when its
credential spec is ``key_required`` and no secret resolves (environment
variable or settings). Adapters with an optional key are never gated, and
fixture mode (``QUBE_KNOWLEDGE_FIXTURES=1``) gates nothing, since fixture
adapters answer without a key.
"""

from __future__ import annotations

from typing import Iterable

from core.knowledge.credentials import CredentialMode, resolve_credential
from core.knowledge.provider_credentials import (
    get_provider_credential_spec,
    provider_id_for_adapter,
)

SKIP_NO_KEY = "no_key"


def missing_required_key(adapter_id: str) -> bool:
    """True when ``adapter_id`` cannot search: its key is required and not set."""
    provider_id = provider_id_for_adapter(adapter_id)
    if provider_id is None:
        return False
    spec = get_provider_credential_spec(provider_id)
    if spec is None or not spec.key_required:
        return False
    cred = resolve_credential(provider_id)
    return cred.mode != CredentialMode.FIXTURE and not cred.secret


def split_keyless(adapter_ids: Iterable[str]) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """``adapter_ids`` that can search, in order, and one skip record per gated adapter."""
    runnable: list[str] = []
    skips: list[dict[str, str]] = []
    for adapter_id in adapter_ids:
        if missing_required_key(adapter_id):
            skips.append({"adapter": str(adapter_id), "reason": SKIP_NO_KEY})
        else:
            runnable.append(adapter_id)
    return tuple(runnable), skips


__all__ = ["SKIP_NO_KEY", "missing_required_key", "split_keyless"]
