"""
Persistent per-model prompt layout overrides (JSON on disk).

Separate from bool-only capability_overrides — layout is a string enum.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_VALID_LAYOUTS = frozenset({"system_ok", "short_system", "flatten_user"})

OVERRIDE_PATH = os.path.expanduser("~/.qube/prompt_layout_overrides.json")


def _normalize_layout(value: object) -> Optional[str]:
    s = str(value or "").strip().lower()
    return s if s in _VALID_LAYOUTS else None


def load_overrides() -> dict[str, str]:
    if not os.path.exists(OVERRIDE_PATH):
        return {}
    with open(OVERRIDE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        layout = _normalize_layout(v)
        if layout is not None:
            out[key] = layout
    return out


def save_overrides(data: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(OVERRIDE_PATH) or ".", exist_ok=True)
    clean = {
        str(k).strip(): _normalize_layout(v) or "system_ok"
        for k, v in data.items()
        if str(k).strip()
    }
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def get_override(model_key: str) -> Optional[str]:
    key = str(model_key or "").strip()
    if not key:
        return None
    data = load_overrides()
    for candidate in (key, key.lower(), os.path.basename(key).lower()):
        if candidate in data:
            return data[candidate]
    return None


def set_override(model_key: str, layout: str) -> None:
    key = str(model_key or "").strip()
    if not key:
        return
    norm = _normalize_layout(layout)
    if norm is None:
        return
    data = load_overrides()
    data[key] = norm
    save_overrides(data)
