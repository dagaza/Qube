"""Helpers for custom color scheme ids and names."""

from __future__ import annotations

import re


def slugify_scheme_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "custom"


def ensure_user_scheme_id(raw_id: str, *, fallback_name: str) -> str:
    """Normalize imported ids so they never overwrite built-in presets."""
    cleaned = str(raw_id or "").strip()
    if not cleaned or cleaned.startswith("builtin."):
        return f"user.{slugify_scheme_name(fallback_name)}"
    if cleaned.startswith("user."):
        return cleaned
    return f"user.{slugify_scheme_name(cleaned)}"


def uniquify_scheme_id(base_id: str, existing: set[str]) -> str:
    if base_id not in existing:
        return base_id
    stem = base_id
    counter = 2
    while f"{stem}-{counter}" in existing:
        counter += 1
    return f"{stem}-{counter}"
