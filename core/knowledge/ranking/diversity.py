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
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance with adapter-diversity penalty."""
    if not rows or max_results <= 0:
        return []
    if len(rows) <= max_results:
        return list(rows)

    rel_scores = [float(r.get("_scientific_relevance") or 0.0) for r in rows]
    lo = min(rel_scores)
    hi = max(rel_scores)
    span = hi - lo

    def _norm_rel(idx: int) -> float:
        if span <= 1e-9:
            return 1.0 if rel_scores[idx] > 0 else 0.0
        return (rel_scores[idx] - lo) / span

    selected: list[dict[str, Any]] = []
    selected_adapters: set[str] = set()
    remaining = list(rows)

    while remaining and len(selected) < max_results:
        best_idx = 0
        best_mmr = float("-inf")
        for idx, cand in enumerate(remaining):
            rel = _norm_rel(idx)
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
        picked_row = remaining.pop(best_idx)
        selected.append(picked_row)
        selected_adapters.add(_adapter_key(picked_row))

    return selected
