"""Plain-language routing lines for the Retrieval Inspector Summary tab."""

from __future__ import annotations

from typing import Any, Mapping


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def format_routing_inspect_text(
    routing_record: Mapping[str, Any] | None,
    *,
    retrieval_outcome: Mapping[str, Any] | None = None,
) -> str:
    """Format cognitive-router explainability for INSPECT RETRIEVAL."""
    record = _as_dict(routing_record)
    outcome = _as_dict(retrieval_outcome)
    if not outcome and record:
        trace = _as_dict(record.get("trace"))
        outcome = _as_dict(trace.get("retrieval_outcome"))

    if not record and not outcome:
        return ""

    lines: list[str] = ["", "Routing (this turn):"]

    route = str(record.get("route") or outcome.get("execution_route_final") or "—")
    pre = record.get("route_pre_policy") or outcome.get("execution_route_pre_downgrade")
    if pre and str(pre).lower() != str(route).lower():
        lines.append(f"  Route: {pre} → {route} (after retrieval / policy)")
    else:
        lines.append(f"  Route: {route}")

    if record.get("strategy"):
        lines.append(f"  Strategy: {record.get('strategy')}")
    if record.get("top_intent"):
        score = record.get("top_score")
        if score is not None:
            lines.append(f"  Top intent: {record.get('top_intent')} ({float(score):.2f})")
        else:
            lines.append(f"  Top intent: {record.get('top_intent')}")

    if outcome:
        mem = int(outcome.get("memory_hits") or 0)
        rag = int(outcome.get("rag_hits") or 0)
        web = int(outcome.get("web_hits") or 0)
        lines.append(f"  Hits: memory={mem}, library={rag}, web={web}")
        if outcome.get("downgrade_fired"):
            lines.append(
                "  Empty-source downgrade: retrieval ran but zero hits survived → plain chat"
            )
        override = outcome.get("override_reason") or outcome.get("failure_reason")
        if override:
            lines.append(f"  Note: {override}")

    summary = str(record.get("summary") or "").strip()
    if summary:
        lines.append(f"  Summary: {summary}")

    decision = _as_dict(record.get("decision"))
    policy = decision.get("tier5_policy") or decision.get("policy")
    if policy and policy not in ("accept", "no_action", None):
        lines.append(f"  Policy trace: {policy}")

    lines.append(
        "  Tip: enable **Routing debug log** (Settings → Privacy & data) for full JSONL per turn."
    )
    return "\n".join(lines)
