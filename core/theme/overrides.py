"""Draft override helpers for theme customization."""

from __future__ import annotations

from core.theme.tokens import CORE_TOKEN_KEYS, CoreTokenSet


def sparse_core_overrides(
    base_core: CoreTokenSet,
    draft_core: CoreTokenSet | dict[str, str],
) -> dict[str, str]:
    """Return only primitive tokens that differ from ``base_core``."""
    draft = draft_core.as_dict() if isinstance(draft_core, CoreTokenSet) else dict(draft_core)
    base = base_core.as_dict()
    return {key: draft[key] for key in CORE_TOKEN_KEYS if draft.get(key) != base.get(key)}
