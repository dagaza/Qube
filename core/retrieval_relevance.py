"""
Deterministic query–snippet relevance helpers for retrieval channels.

Pure functions; no Qt, no LanceDB, no network I/O.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

import numpy as np

DEFAULT_WEB_MIN_TOKEN_OVERLAP = 0.15
DEFAULT_WEB_MIN_SEMANTIC_SCORE = 0.30

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall((text or "").lower())}


_WEB_META_STOPWORDS: frozenset[str] = frozenset(
    {
        "online",
        "search",
        "web",
        "internet",
        "answer",
        "please",
        "also",
        "nice",
        "would",
        "can",
        "you",
        "the",
        "for",
        "an",
        "do",
        "a",
        "yes",
        "that",
    }
)


def _query_tokens_for_relevance(query: str) -> set[str]:
    tokens = _token_set(query)
    if not tokens:
        return set()
    stripped = tokens - _WEB_META_STOPWORDS
    if len(stripped) >= 2:
        return stripped
    if stripped:
        return stripped
    return set()


def query_snippet_token_overlap(query: str, snippet: str) -> float:
    """Fraction of query tokens present in snippet (0.0–1.0)."""
    q_tokens = _query_tokens_for_relevance(query)
    s_tokens = _token_set(snippet)
    if not q_tokens:
        return 0.0
    if not s_tokens:
        return 0.0
    return len(q_tokens & s_tokens) / len(q_tokens)


def passes_web_relevance_gate(
    query: str,
    title: str,
    snippet: str,
    *,
    min_ratio: float = DEFAULT_WEB_MIN_TOKEN_OVERLAP,
) -> bool:
    combined = f"{title or ''} {snippet or ''}".strip()
    return query_snippet_token_overlap(query, combined) >= min_ratio


def _semantic_score_from_vectors(
    query_vector: np.ndarray,
    text_vector: np.ndarray,
) -> float:
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    t = np.asarray(text_vector, dtype=np.float32).reshape(-1)
    if q.size == 0 or t.size == 0 or q.shape != t.shape:
        return 0.0
    q_norm = float(np.linalg.norm(q))
    t_norm = float(np.linalg.norm(t))
    if q_norm <= 1e-9 or t_norm <= 1e-9:
        return 0.0
    cosine = float(np.dot(q, t) / (q_norm * t_norm))
    return max(0.0, min(1.0, cosine))


def filter_web_results(
    query: str,
    items: list[dict[str, Any]],
    *,
    min_token_ratio: float = DEFAULT_WEB_MIN_TOKEN_OVERLAP,
    query_vector: Optional[np.ndarray] = None,
    embed_text_fn: Optional[Callable[[str], np.ndarray]] = None,
    min_semantic_score: float = DEFAULT_WEB_MIN_SEMANTIC_SCORE,
    use_embedding_gate: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Drop web hits that fail lexical (and optional embedding) relevance vs ``query``.

    Returns (kept_items, diagnostics_dict).
    """
    raw_count = len(items or [])
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    use_embed = bool(
        use_embedding_gate
        and query_vector is not None
        and embed_text_fn is not None
    )

    for item in items or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        token_overlap = query_snippet_token_overlap(
            query, f"{title} {snippet}".strip()
        )
        semantic_score: Optional[float] = None
        passes_token = token_overlap >= min_token_ratio
        passes_semantic = True
        if use_embed:
            body = f"{title}\n{snippet}".strip()
            if body:
                try:
                    text_vector = embed_text_fn(body)
                    semantic_score = _semantic_score_from_vectors(
                        query_vector, text_vector
                    )
                    passes_semantic = semantic_score >= min_semantic_score
                except Exception:
                    passes_semantic = passes_token
            else:
                passes_semantic = False

        if passes_token and passes_semantic:
            row = dict(item)
            if semantic_score is not None:
                row["_web_semantic_score"] = round(semantic_score, 4)
            row["_web_token_overlap"] = round(token_overlap, 4)
            kept.append(row)
        else:
            dropped.append(
                {
                    "title": title[:80],
                    "token_overlap": round(token_overlap, 4),
                    "semantic_score": (
                        round(semantic_score, 4)
                        if semantic_score is not None
                        else None
                    ),
                }
            )

    diagnostics = {
        "web_results_raw_count": raw_count,
        "web_results_kept_count": len(kept),
        "web_relevance_min_overlap": min_token_ratio,
        "web_relevance_dropped": dropped,
        "web_relevance_embedding_gate": use_embed,
    }
    if use_embed:
        diagnostics["web_relevance_min_semantic"] = min_semantic_score
    return kept, diagnostics
