"""Shared model parameter count inference for Hub metadata and quant recommendations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_PARAMS_B_IN_NAME_RE = re.compile(
    r"(?:^|[-_.])(\d+(?:\.\d+)?)[Bb](?:[-_.]|$)",
)


def parse_params_b_from_label(label: str) -> float | None:
    """Parse '7B', '3.5B', '70b' into billions as float."""
    s = str(label or "").strip().upper().replace(" ", "")
    if not s or s == "UNKNOWN":
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)B$", s)
    if m:
        return float(m.group(1))
    return None


def parse_params_b_from_filename(name: str) -> float | None:
    """Best-effort parameter count (billions) from a local .gguf filename."""
    from core.app_settings import parse_gguf_shard_info

    basename = Path(str(name or "")).name
    stem = basename[:-5] if basename.lower().endswith(".gguf") else basename

    shard = parse_gguf_shard_info(basename)
    if shard is not None:
        stem = str(shard.get("prefix") or stem)

    matches = [float(m) for m in _PARAMS_B_IN_NAME_RE.findall(stem)]
    if matches:
        return max(matches)

    m = re.search(r"\b(\d+(?:\.\d+)?)[Bb]\b", stem)
    if m:
        return float(m.group(1))
    return None


def infer_params_b(
    *,
    card: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    repo_id: str = "",
    title: str = "",
    description: str = "",
    params_label: str | None = None,
) -> tuple[float | None, str]:
    """
    Return (params_b, source) where source describes confidence input:
    hf_card | repo_inference | unknown
    """
    card = card or {}
    if params_label:
        parsed = parse_params_b_from_label(params_label)
        if parsed is not None:
            return parsed, "hf_card"

    for key in ("params", "parameter_count", "parameters", "model_size"):
        v = card.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), "hf_card"
        if isinstance(v, str) and v.strip():
            parsed = parse_params_b_from_label(v.strip().upper().replace(" ", ""))
            if parsed is not None:
                return parsed, "hf_card"

    hay = " ".join(list(tags or []) + [repo_id, title, description]).lower()
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*b\b", hay)
    if m:
        return float(m.group(1)), "repo_inference"
    return None, "unknown"
