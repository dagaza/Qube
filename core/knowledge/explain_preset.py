"""Derived Explain Preset view — transparency without duplicating config."""

from __future__ import annotations

from typing import Any

from core.knowledge.presets import KnowledgePreset, load_preset
from core.knowledge.types import SERVICE_GENERAL_WEB
from core.knowledge.retrieval_profiles import get_profile_spec
from core.knowledge.retrieval_trace_reader import read_retrieval_traces

_RANKING_LABELS = {
    "generic": "General relevance",
    "literature": "Literature-oriented",
    "regulatory": "Regulatory filings",
    "market_data": "Market data",
}

_SERVICE_LABELS = {
    "scientific_evidence": "Scientific Evidence",
    "finance_knowledge": "Finance",
    "legal_knowledge": "Legal",
    "general_web": "General Web",
    "trusted_knowledge": "Trusted Knowledge",
    "internal_corpus": "Internal Corpus",
    "preset_knowledge": "My Knowledge",
}


def _adapter_label(adapter_id: str) -> str:
    from core.knowledge.configured_sources import load_configured_source

    source = load_configured_source(adapter_id)
    if source is not None:
        return f"{source.label} (custom source)"
    return adapter_id


def _latency_stats(preset_id: str) -> dict[str, Any]:
    strategy = f"preset:{preset_id}"
    traces = read_retrieval_traces(limit=100)
    latencies = [
        float(t.get("latency_ms") or 0)
        for t in traces
        if str(t.get("retrieval_strategy") or "") == strategy
    ]
    if not latencies:
        return {"sample_count": 0, "avg_latency_ms": None}
    return {
        "sample_count": len(latencies),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
    }


def build_explain_preset(
    preset_id: str,
    *,
    retrieval_profile: str = "balanced",
) -> dict[str, Any]:
    preset = load_preset(preset_id)
    if preset is None:
        return {"error": f"Preset '{preset_id}' not found."}

    profile = get_profile_spec(retrieval_profile)
    stats = _latency_stats(preset.id)
    if preset.base_service == SERVICE_GENERAL_WEB:
        uses = [
            {"id": domain, "label": domain}
            for domain in preset.site_bias
        ]
        fetch_hint = (
            str(preset.fetch_url_count)
            if preset.fetch_url_count is not None
            else "profile default"
        )
        extra_fields = {
            "fetch_url_count": fetch_hint,
            "mode": "Web fetch (source profile)",
        }
    else:
        uses = [
            {"id": aid, "label": _adapter_label(aid)} for aid in preset.adapters
        ]
        extra_fields = {"mode": "API adapters"}

    return {
        "preset_id": preset.id,
        "label": preset.label,
        "description": preset.description or "",
        "uses": uses,
        "base_service": _SERVICE_LABELS.get(
            preset.base_service, preset.base_service
        ),
        "retrieval_profile": profile.label,
        "ranking_strategy": _RANKING_LABELS.get(
            preset.ranking_profile, preset.ranking_profile
        ),
        "query_planner": preset.query_planner,
        "composer_token": f"@[tool:user:{preset.id}]",
        "typical_latency_ms": stats.get("avg_latency_ms"),
        "latency_sample_count": stats.get("sample_count", 0),
        "diagnostics_hint": (
            "Open Retrieval Inspector on an answer that used this tool, "
            "or check Settings → Knowledge → Diagnostics."
        ),
        **extra_fields,
    }


def format_explain_preset_text(explain: dict[str, Any]) -> str:
    if explain.get("error"):
        return str(explain["error"])

    lines = [
        f"{explain.get('label', explain.get('preset_id'))}",
        "",
        f"Composer token: {explain.get('composer_token', '—')}",
        f"Mode: {explain.get('mode', '—')}",
        f"Base service: {explain.get('base_service', '—')}",
        f"Retrieval profile (active): {explain.get('retrieval_profile', '—')}",
        f"Ranking strategy: {explain.get('ranking_strategy', '—')}",
        "",
    ]
    if explain.get("fetch_url_count") is not None:
        lines.append(f"Fetch URL count: {explain.get('fetch_url_count')}")
        lines.append("")
    lines.append("Uses:")
    for item in explain.get("uses") or []:
        lines.append(f"  • {item.get('label', item.get('id', '—'))}")

    desc = str(explain.get("description") or "").strip()
    if desc:
        lines.extend(["", f"Description: {desc}"])

    avg = explain.get("typical_latency_ms")
    samples = int(explain.get("latency_sample_count") or 0)
    if avg is not None and samples > 0:
        lines.append("")
        lines.append(
            f"Typical latency: ~{avg} ms (from {samples} recent trace{'s' if samples != 1 else ''})"
        )

    hint = explain.get("diagnostics_hint")
    if hint:
        lines.extend(["", str(hint)])
    return "\n".join(lines)
