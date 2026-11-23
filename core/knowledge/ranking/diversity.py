"""MMR diversity reranking for evidence candidate rows."""

from __future__ import annotations

import difflib
from typing import Any

MMR_LAMBDA = 0.72


def _adapter_key(row: dict[str, Any]) -> str:
    return str(row.get("_adapter") or "unknown")


def _row_content(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "")
    snippet = str(row.get("snippet") or "")
    return f"{title}\n{snippet}".strip().lower()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def mmr_select_rows(
    rows: list[dict[str, Any]],
    *,
    max_results: int,
    lambda_: float = MMR_LAMBDA,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance with adapter-diversity penalty.

    ``normalize=True`` (legacy token/embedding scores, compressed into a narrow
    band) min-max scales relevance across the pool. ``normalize=False`` uses
    ``_scientific_relevance`` as-is on its 0–1 scale; the S4 relevance gate
    passes calibrated reranker probabilities that way.
    """
    if not rows or max_results <= 0:
        return []
    if len(rows) <= max_results:
        return list(rows)

    rel_scores = [
        max(0.0, min(1.0, float(r.get("_scientific_relevance") or 0.0))) for r in rows
    ]
    lo = min(rel_scores)
    span = max(rel_scores) - lo

    def _rel(score: float) -> float:
        if not normalize:
            return score
        if span <= 1e-9:
            return 1.0 if score > 0 else 0.0
        return (score - lo) / span

    # Relevance travels with its row. Fixed 2026-11-23 (S4 progress log): it was
    # looked up by position in the shrinking ``remaining`` list, so after the
    # first pick every candidate was scored with another row's relevance.
    selected: list[dict[str, Any]] = []
    selected_adapters: set[str] = set()
    remaining = [(row, _rel(score)) for row, score in zip(rows, rel_scores)]

    while remaining and len(selected) < max_results:
        best_idx = 0
        best_mmr = float("-inf")
        for idx, (cand, rel) in enumerate(remaining):
            cand_text = _row_content(cand)
            max_sim = 0.0
            for picked in selected:
                max_sim = max(max_sim, _similarity(cand_text, _row_content(picked)))
            adapter = _adapter_key(cand)
            adapter_bonus = 0.08 if adapter not in selected_adapters else 0.0
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim + adapter_bonus
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
        picked_row, _ = remaining.pop(best_idx)
        selected.append(picked_row)
        selected_adapters.add(_adapter_key(picked_row))

    return selected
