"""Offline helpers for Phase 0 knowledge-quality evaluation / replay fixtures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.knowledge.bundle_builder import build_scientific_evidence_bundle
from core.knowledge.ranking.relevance import score_rows
from core.knowledge.ui_adapter import evidence_to_ui_source
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt
from core.prompt_history import strip_composer_tokens_from_history
from core.prompt_renderers import render_system_ok_messages

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def title_matches_reject(title: str, patterns: list[str] | tuple[str, ...]) -> bool:
    t = (title or "").lower()
    for pat in patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def title_has_expected_token(title: str, tokens: list[str] | tuple[str, ...]) -> bool:
    t = (title or "").lower()
    return any(tok.lower() in t for tok in tokens if tok)


def score_fixture_candidates(
    fixture: dict[str, Any],
    *,
    query: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the scientific relevance scorer on fixture candidates (token path)."""
    q = (query or fixture.get("query_clean") or fixture.get("query_raw") or "").strip()
    rows = [dict(r) for r in (fixture.get("candidates") or [])]
    return score_rows(rows, query=q, min_score=0.12)


def build_fixture_bundle(
    fixture: dict[str, Any],
    *,
    kept: list[dict[str, Any]] | None = None,
) -> Any:
    query = (fixture.get("query_clean") or fixture.get("query_raw") or "").strip()
    rows = kept
    if rows is None:
        kept_rows, _ = score_fixture_candidates(fixture, query=query)
        rows = kept_rows
    return build_scientific_evidence_bundle(
        query_raw=str(fixture.get("query_raw") or query),
        query_resolved=query,
        kept_rows=rows,
        rejected_count=0,
        latency_ms=1.0,
        adapter_calls=tuple(
            sorted({str(r.get("_adapter") or "") for r in rows if r.get("_adapter")})
        ),
        medical_query=False,
    )


def prompt_allows_abstention(system_text: str, user_turn_text: str = "") -> bool:
    blob = f"{system_text}\n{user_turn_text}".lower()
    markers = (
        "if none of the sources are relevant",
        "if the retrieved sources do not address",
        "if none of the sources address",
        "say so plainly",
        "say so clearly",
    )
    return any(m in blob for m in markers)


def evaluate_ant_fixture(fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Baseline checks for the recorded ANT/@evidence failure mode."""
    if fixture is None:
        fixture = load_json(
            "eval/fixtures/knowledge_quality/ant_ai_candidates.json"
        )

    history = list(fixture.get("history") or [])
    history.append(
        {
            "role": "user",
            "content": str(fixture.get("query_raw") or fixture.get("query_clean") or ""),
        }
    )
    cleaned = strip_composer_tokens_from_history(history)
    last_user = next(
        (
            str(m.get("content") or "")
            for m in reversed(cleaned)
            if str(m.get("role") or "").lower() == "user"
        ),
        "",
    )
    token_stripped = "@[tool:" not in last_user and not last_user.lstrip().startswith(
        "@evidence"
    )

    kept, rejected = score_fixture_candidates(fixture)
    reject_patterns = list(fixture.get("reject_title_patterns") or [])
    kept_rejects = [
        r for r in kept if title_matches_reject(str(r.get("title") or ""), reject_patterns)
    ]
    # Phase 0 baseline: token scorer still keeps irrelevant ANT/AI puns.
    # We record this as a known gap until Phase 2 reranking.
    relevant_in_pool = any(
        str(r.get("label") or "") == "relevant" for r in (fixture.get("candidates") or [])
    )
    relevant_kept = [
        r for r in kept if str(r.get("label") or "") == "relevant"
    ]

    bundle = build_fixture_bundle(fixture, kept=kept)
    ui_sources = [
        evidence_to_ui_source(obj, ui_id=i)
        for i, obj in enumerate(bundle.sources, start=1)
    ]
    has_metadata = any(
        src.get("venue") or src.get("publication_date") or src.get("preprint") is not None
        for src in ui_sources
    )
    has_fuller_body = any(
        len(str(src.get("content") or "")) > 200 for src in ui_sources
    )

    blocks = build_prompt_blocks(
        execution_route="WEB",
        explicit_remember_active=False,
        has_retrieval_sources=True,
        retrieval_source_count=len(ui_sources),
        web_hit_count=len(ui_sources),
        knowledge_service="scientific_evidence",
        conversation_history=cleaned,
        retrieval_context="--- [1]: Example ---\nBody.",
        current_date="2026-11-23",
        engine_mode="external",
    )
    system = compose_system_prompt(blocks)
    messages = render_system_ok_messages(blocks)
    user_turn = messages[-1]["content"] if messages else ""

    return {
        "fixture_id": fixture.get("id"),
        "composer_token_stripped": token_stripped,
        "prompt_allows_abstention": prompt_allows_abstention(system, user_turn),
        "scientific_persona": "Scientific literature results" in system,
        "date_injected": "Today's date is 2026-11-23" in system,
        "evidence_pack_has_metadata": has_metadata,
        "evidence_pack_fuller_body": has_fuller_body,
        "coverage": bundle.coverage,
        "confidence": round(float(bundle.confidence), 4),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "kept_reject_title_count": len(kept_rejects),
        "relevant_in_pool": relevant_in_pool,
        "relevant_kept_count": len(relevant_kept),
        "phase0_known_gap_token_ranker_keeps_irrelevant": len(kept_rejects) > 0,
        "freshness_populated": any(
            getattr(s, "freshness_score", None) is not None for s in bundle.sources
        ),
    }
