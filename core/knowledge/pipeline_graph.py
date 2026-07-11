"""Read-only pipeline graph generation from trace stages."""

from __future__ import annotations

from typing import Any


def build_pipeline_graph_from_trace(trace: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return ordered nodes for explanation UI (not a workflow editor)."""
    if not trace:
        return []

    nodes: list[dict[str, str]] = [
        {"id": "question", "label": "Question", "detail": str(trace.get("query_raw") or "")[:120]},
    ]

    strategy = str(trace.get("retrieval_strategy") or "")
    if strategy.startswith("preset:"):
        nodes.append(
            {
                "id": "preset",
                "label": "Knowledge Preset",
                "detail": strategy.replace("preset:", "", 1),
            }
        )

    nodes.append(
        {
            "id": "service",
            "label": "Knowledge Service",
            "detail": str(trace.get("knowledge_service") or ""),
        }
    )
    nodes.append({"id": "pipeline", "label": "Pipeline", "detail": strategy or "domain pipeline"})

    adapters = trace.get("adapter_calls") or []
    if adapters:
        nodes.append(
            {
                "id": "adapters",
                "label": "Adapters",
                "detail": ", ".join(str(a) for a in adapters),
            }
        )

    profile = trace.get("retrieval_profile")
    if profile:
        nodes.append(
            {"id": "profile", "label": "Retrieval Profile", "detail": str(profile)}
        )

    stages = trace.get("pipeline_stages") or []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("stage") or f"stage_{idx}")
        adapter = stage.get("adapter")
        label = stage_name.replace("_", " ").title()
        if adapter:
            label = f"{label}: {adapter}"
        detail_parts = []
        if stage.get("outputs_count") is not None:
            detail_parts.append(f"out={stage.get('outputs_count')}")
        if stage.get("latency_ms") is not None:
            detail_parts.append(f"{stage.get('latency_ms')} ms")
        nodes.append(
            {
                "id": f"stage_{idx}",
                "label": label,
                "detail": ", ".join(detail_parts) or stage_name,
            }
        )

    nodes.append(
        {
            "id": "ranking",
            "label": "Evidence Ranking",
            "detail": f"kept {len(trace.get('evidence_ids_kept') or [])}, "
            f"rejected {trace.get('candidates_rejected_count', 0)}",
        }
    )
    nodes.append(
        {
            "id": "bundle",
            "label": "EvidenceBundle",
            "detail": f"coverage={trace.get('coverage', '—')}, "
            f"confidence={trace.get('confidence', '—')}",
        }
    )
    nodes.append({"id": "prompt", "label": "Prompt Assembly", "detail": "Evidence injected into context"})
    nodes.append({"id": "llm", "label": "LLM", "detail": "Answer generation"})
    nodes.append({"id": "answer", "label": "Answer", "detail": "User-visible response"})
    return nodes


def format_pipeline_graph_text(trace: dict[str, Any] | None) -> str:
    nodes = build_pipeline_graph_from_trace(trace)
    if not nodes:
        return "No pipeline graph available for this retrieval."
    lines = []
    for node in nodes:
        detail = str(node.get("detail") or "").strip()
        if detail:
            lines.append(f"{node.get('label', '—')}\n  {detail}")
        else:
            lines.append(str(node.get("label", "—")))
        lines.append("  ↓")
    if lines and lines[-1] == "  ↓":
        lines.pop()
    return "\n".join(lines)
