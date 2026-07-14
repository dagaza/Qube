"""Optional RxNorm drug-name lookup with file cache (off by default)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from core.app_settings import rxnorm_entity_lookup_enabled
from core.knowledge.entities.ids import make_entity_id

logger = logging.getLogger("Qube.EntityRxNorm")

_CACHE_PATH = Path.home() / ".qube" / "rxnorm_cache.json"
_CACHE_TTL_SECONDS = 7 * 24 * 3600
_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name="


def _load_cache() -> dict[str, Any]:
    try:
        if _CACHE_PATH.is_file():
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.debug("RxNorm cache write failed", exc_info=True)


def lookup_rxnorm_entity(drug_name: str) -> str | None:
    """Return entity:rxnorm:{rxcui} when lookup succeeds; None if disabled or miss."""
    if not rxnorm_entity_lookup_enabled():
        return None
    name = (drug_name or "").strip().lower()
    if len(name) < 3:
        return None

    cache = _load_cache()
    entry = cache.get(name)
    now = time.time()
    if isinstance(entry, dict):
        ts = float(entry.get("ts") or 0)
        rxcui = entry.get("rxcui")
        if rxcui and (now - ts) < _CACHE_TTL_SECONDS:
            return make_entity_id("rxnorm", str(rxcui))

    try:
        url = f"{_RXNORM_BASE}{quote(name)}"
        with urlopen(url, timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        group = (payload.get("idGroup") or {}) if isinstance(payload, dict) else {}
        rxcui = group.get("rxnormId")
        if isinstance(rxcui, list):
            rxcui = rxcui[0] if rxcui else None
        if not rxcui:
            cache[name] = {"ts": now, "rxcui": None}
            _save_cache(cache)
            return None
        cache[name] = {"ts": now, "rxcui": str(rxcui)}
        _save_cache(cache)
        return make_entity_id("rxnorm", str(rxcui))
    except Exception:
        logger.debug("RxNorm lookup failed for %r", name, exc_info=True)
        return None
