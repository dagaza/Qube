"""File-backed evidence retrieval cache (~/.qube/evidence_cache/)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_TTL_SECONDS = 3600
_CACHE_DIR = Path.home() / ".qube" / "evidence_cache"


def evidence_cache_enabled() -> bool:
    raw = os.getenv("QUBE_EVIDENCE_CACHE")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _cache_path(cache_key: str) -> Path:
    return _CACHE_DIR / f"{cache_key}.json"


def make_cache_key(
    *,
    knowledge_service: str,
    query: str,
    adapter_filter: tuple[str, ...] | None = None,
) -> str:
    adapters = ",".join(sorted(adapter_filter or ()))
    raw = f"{knowledge_service}|{query.strip().lower()}|{adapters}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def get_cached_rows(
    cache_key: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> list[dict[str, Any]] | None:
    if not evidence_cache_enabled():
        return None
    path = _cache_path(cache_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ts = float(payload.get("ts") or 0)
        if (time.time() - ts) > ttl_seconds:
            return None
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, dict)]
    except Exception:
        return None
    return None


def set_cached_rows(cache_key: str, rows: list[dict[str, Any]]) -> None:
    if not evidence_cache_enabled():
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), "rows": rows}
        _CACHE_DIR.joinpath(f"{cache_key}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
