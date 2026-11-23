"""History cleaning helpers for prompt assembly (composer tokens, display)."""

from __future__ import annotations

from typing import Any

from core.composer_skills import strip_all_composer_tokens_for_display


def strip_composer_tokens_from_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return a shallow-copied history with composer tokens removed from user text.

    Persistence keeps raw ``@[tool:…]`` tokens for UI fidelity; the model must not
    see those control tokens as part of the user's question.
    """
    if not history:
        return []
    cleaned: list[dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        row = dict(msg)
        role = str(row.get("role") or "").strip().lower()
        if role == "user":
            content = str(row.get("content") or "")
            stripped = strip_all_composer_tokens_for_display(content).strip()
            row["content"] = stripped if stripped else content
        cleaned.append(row)
    return cleaned
