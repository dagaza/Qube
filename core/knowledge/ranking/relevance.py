"""Relevance scoring for evidence candidate rows."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def token_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_TOKEN_RE.findall((query or "").lower()))
    if not q_tokens:
        return 0.0
    t_tokens = set(_TOKEN_RE.findall((text or "").lower()))
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("title", "snippet", "full_text")
    ).strip()


def score_evidence_row(
    row: dict[str, Any],
    *,
    query: str,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
) -> float:
    """Combined token + optional embedding relevance in [0, 1]."""
    text = row_text(row)
    token = token_overlap_score(query, text)
    semantic: float | None = None
    if query_vector is not None and embed_fn is not None and text:
        try:
            doc_vec = embed_fn(text[:512])
            if doc_vec is not None and len(doc_vec) == len(query_vector):
                qn = float(np.linalg.norm(query_vector))
                dn = float(np.linalg.norm(doc_vec))
                if qn > 0 and dn > 0:
                    semantic = float(np.dot(query_vector, doc_vec) / (qn * dn))
                    semantic = max(0.0, min(1.0, (semantic + 1.0) / 2.0))
        except Exception:
            semantic = None
    if semantic is not None:
        return max(0.0, min(1.0, token * 0.45 + semantic * 0.55))
    return max(0.0, min(1.0, token))


def score_rows(
    rows: list[dict[str, Any]],
    *,
    query: str,
    query_vector: np.ndarray | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    min_score: float = 0.12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score rows and partition into kept vs rejected by min_score."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        rel = score_evidence_row(
            copy,
            query=query,
            query_vector=query_vector,
            embed_fn=embed_fn,
        )
        copy["_scientific_relevance"] = rel
        if rel >= min_score:
            kept.append(copy)
        else:
            rejected.append(copy)
    kept.sort(key=lambda r: float(r.get("_scientific_relevance") or 0.0), reverse=True)
    return kept, rejected
