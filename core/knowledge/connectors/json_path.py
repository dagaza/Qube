"""JSON path extraction for connector response mapping."""

from __future__ import annotations

from typing import Any


def extract_json_path(data: Any, path: str) -> Any:
    """Extract value using simple ``$.a.b[0]`` style paths."""
    text = (path or "").strip()
    if not text or text == "$":
        return data
    if not text.startswith("$."):
        return None

    current = data
    for segment in text[2:].split("."):
        if not segment:
            continue
        if "[" in segment:
            key, _, rest = segment.partition("[")
            if key:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
            idx_text = rest.rstrip("]")
            try:
                idx = int(idx_text)
            except ValueError:
                return None
            if not isinstance(current, list) or idx >= len(current):
                return None
            current = current[idx]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(segment)
    return current
