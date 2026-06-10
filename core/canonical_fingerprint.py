"""
Provider-agnostic content fingerprints for LLM debugging (requests, prompts, outputs).

Uses stable JSON serialization for structured data and whitespace-normalized text
for string payloads. No model- or vendor-specific assumptions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from core.canonical_request import CanonicalRequest

_SHORT_LEN = 12


def normalize_text_for_fingerprint(text: str) -> str:
    """Normalize line endings and trailing/leading whitespace without altering semantics."""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in s.split("\n")]
    return "\n".join(lines).strip()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _fingerprint_preimage(preimage: str) -> dict[str, Any]:
    digest = hashlib.sha256(preimage.encode("utf-8", errors="replace")).hexdigest()
    return {
        "sha256": digest,
        "short": digest[:_SHORT_LEN],
        "length": len(preimage),
    }


def fingerprint_text(text: str) -> dict[str, Any]:
    """Fingerprint rendered prompts, model outputs, and other plain text."""
    normalized = normalize_text_for_fingerprint(text or "")
    return _fingerprint_preimage(normalized)


def fingerprint_canonical_request(request: CanonicalRequest) -> dict[str, Any]:
    """Fingerprint a normalized CanonicalRequest (stable JSON, sorted keys)."""
    canonical = stable_json_dumps(request.to_dict())
    return _fingerprint_preimage(canonical)


def fingerprint_trace_component(component: dict | str) -> dict[str, Any]:
    """Fingerprint an arbitrary trace component (dict -> stable JSON, str -> text rules)."""
    if isinstance(component, str):
        return fingerprint_text(component)
    if isinstance(component, CanonicalRequest):
        return fingerprint_canonical_request(component)
    return _fingerprint_preimage(stable_json_dumps(component or {}))
