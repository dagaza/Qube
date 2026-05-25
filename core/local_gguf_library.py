"""Shared helpers for listing downloaded local .gguf models in menus."""

from __future__ import annotations

from pathlib import Path

from core.app_settings import get_llm_models_dir, is_secondary_gguf_shard
from core.local_gguf_display import format_local_gguf_display, local_gguf_sort_key


def list_local_gguf_menu_entries() -> list[tuple[str, str]]:
    """Return ``(menu_label, resolved_primary_path)`` for each local model bundle."""
    root = Path(get_llm_models_dir())
    if not root.is_dir():
        return []

    entries: list[tuple[str, str]] = []
    for path in sorted(
        (fp for fp in root.glob("*.gguf") if not is_secondary_gguf_shard(str(fp))),
        key=local_gguf_sort_key,
    ):
        resolved = str(path.resolve())
        label = format_local_gguf_display(resolved, models_dir=root).menu_label
        entries.append((label, resolved))
    return entries
