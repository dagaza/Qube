"""Canonical JSON serialization for signed payloads."""

from __future__ import annotations

import json
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for Ed25519 signing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
