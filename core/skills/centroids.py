"""Optional embedding centroids for skill activation boost."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Qube.Skills")

_CENTROIDS_PATH = Path(__file__).resolve().parent / "centroids.json"
_EMBEDDING_WEIGHT = 0.35
_cache: dict[str, np.ndarray] | None = None


def _load_centroids() -> dict[str, np.ndarray]:
    global _cache
    if _cache is not None:
        return _cache
    out: dict[str, np.ndarray] = {}
    if not _CENTROIDS_PATH.is_file():
        _cache = out
        return out
    try:
        raw = json.loads(_CENTROIDS_PATH.read_text(encoding="utf-8"))
        skills = raw.get("skills") if isinstance(raw, dict) else None
        if isinstance(skills, dict):
            for skill_id, vec in skills.items():
                arr = np.asarray(vec, dtype=np.float64).reshape(-1)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-9:
                    out[str(skill_id)] = arr / norm
    except Exception as exc:
        logger.warning("[Skills] Failed to load centroids.json: %s", exc)
    _cache = out
    return out


def embedding_boost_score(
    skill_id: str,
    query_embedding: Any | None,
    *,
    enabled: bool,
) -> tuple[float, str | None]:
    """Cosine similarity scaled by EMBEDDING_WEIGHT; boost-only, not required."""
    if not enabled or query_embedding is None:
        return 0.0, None
    centroids = _load_centroids()
    centroid = centroids.get(skill_id)
    if centroid is None:
        return 0.0, None
    try:
        v = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return 0.0, None
        v = v / norm
        if v.shape != centroid.shape:
            return 0.0, None
        sim = float(np.dot(v, centroid))
        if sim <= 0:
            return 0.0, None
        weighted = min(1.0, sim * _EMBEDDING_WEIGHT)
        return weighted, f"embedding:{weighted:.3f}"
    except Exception:
        return 0.0, None


def reset_centroid_cache_for_tests() -> None:
    global _cache
    _cache = None
