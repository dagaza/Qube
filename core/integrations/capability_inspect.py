"""Pure builders for INSPECT capability provenance steps (T16).

Provider-agnostic projection of attachment → invoke → returned → ranked → cited.
No MCP imports or provider branches (P6). Used by retrieval traces and the
Retrieval Inspector — not a parallel debug subsystem.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.integrations.capabilities.model import CapabilityDescriptor

__all__ = [
    "CAPABILITY_STEP_KINDS",
    "build_attachment_step",
    "build_invoke_step",
    "build_returned_step",
    "build_ranked_step",
    "build_cited_step",
    "build_capability_inspect_trace",
    "capability_steps_from_trace",
    "merge_capability_steps_into_trace",
    "serialize_capability_steps",
    "format_capability_steps_text",
    "format_capability_steps_summary_line",
]

CAPABILITY_STEP_KINDS = (
    "attachment",
    "invoke",
    "returned",
    "ranked",
    "cited",
)


def _step_number(steps: list[dict[str, Any]], kind: str) -> int:
    return len(steps) + 1


def build_attachment_step(
    *,
    urn: str,
    label: str | None = None,
    tier: str | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step or 1,
        "kind": "attachment",
        "urn": str(urn),
    }
    if label:
        payload["label"] = str(label)
    if tier:
        payload["tier"] = str(tier)
    return payload


def build_invoke_step(
    *,
    urn: str,
    query: str,
    allowed: bool,
    reason: str = "",
    latency_ms: float = 0.0,
    action: str | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step or 2,
        "kind": "invoke",
        "urn": str(urn),
        "query": str(query or ""),
        "allowed": bool(allowed),
        "reason": str(reason or ""),
        "latency_ms": round(float(latency_ms), 2),
    }
    if action:
        payload["action"] = str(action)
    return payload


def build_returned_step(
    *,
    raw_count: int,
    latency_ms: float = 0.0,
    step: int | None = None,
) -> dict[str, Any]:
    return {
        "step": step or 3,
        "kind": "returned",
        "raw_count": max(0, int(raw_count)),
        "latency_ms": round(float(latency_ms), 2),
    }


def build_ranked_step(
    *,
    kept_count: int,
    rejected_count: int = 0,
    threshold: float | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step or 4,
        "kind": "ranked",
        "kept_count": max(0, int(kept_count)),
        "rejected_count": max(0, int(rejected_count)),
    }
    if threshold is not None:
        payload["threshold"] = round(float(threshold), 4)
    return payload


def build_cited_step(
    *,
    cited_ids: Sequence[str],
    source_capabilities: Sequence[str] | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step or 5,
        "kind": "cited",
        "cited_ids": [str(item) for item in cited_ids if str(item).strip()],
    }
    if source_capabilities:
        payload["source_capabilities"] = [
            str(item) for item in source_capabilities if str(item).strip()
        ]
    return payload


def _descriptor_label(descriptor: CapabilityDescriptor | None) -> str | None:
    if descriptor is None:
        return None
    if descriptor.description:
        return descriptor.description.strip()
    group = (descriptor.group or "").strip()
    action = (descriptor.action or "").strip()
    if group and action:
        return f"{group} — {action}"
    return group or action or None


def build_capability_inspect_trace(
    *,
    urn: str,
    query: str,
    allowed: bool,
    reason: str = "",
    rows: Sequence[Mapping[str, Any]] | None = None,
    bundle_source_count: int = 0,
    rejected_count: int = 0,
    latency_ms: float = 0.0,
    descriptor: CapabilityDescriptor | None = None,
    cited_ids: Sequence[str] | None = None,
    rank_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Build ordered capability INSPECT steps for one composer attach→invoke turn."""
    tier = descriptor.tier.value if descriptor is not None else None
    label = _descriptor_label(descriptor)
    action = descriptor.action if descriptor is not None else None

    steps: list[dict[str, Any]] = [
        build_attachment_step(
            urn=urn,
            label=label,
            tier=tier,
            step=1,
        ),
        build_invoke_step(
            urn=urn,
            query=query,
            allowed=allowed,
            reason=reason,
            latency_ms=latency_ms,
            action=action,
            step=2,
        ),
    ]

    if not allowed:
        return steps

    raw_count = len(rows or ())
    if raw_count <= 0 and bundle_source_count > 0:
        raw_count = bundle_source_count

    if raw_count > 0 or bundle_source_count > 0:
        steps.append(
            build_returned_step(
                raw_count=raw_count,
                latency_ms=latency_ms,
                step=3,
            )
        )
        steps.append(
            build_ranked_step(
                kept_count=bundle_source_count,
                rejected_count=rejected_count,
                threshold=rank_threshold,
                step=4,
            )
        )

    if cited_ids:
        caps = sorted(
            {
                str(row.get("_capability") or row.get("source_capability") or urn)
                for row in (rows or ())
                if row.get("_capability") or row.get("source_capability") or urn
            }
        )
        steps.append(
            build_cited_step(
                cited_ids=cited_ids,
                source_capabilities=caps or None,
                step=len(steps) + 1,
            )
        )

    return steps


def serialize_capability_steps(steps: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """JSON-safe copy of capability step dicts for trace persistence."""
    out: list[dict[str, Any]] = []
    for item in steps or ():
        if not isinstance(item, Mapping):
            continue
        out.append({str(key): value for key, value in dict(item).items()})
    return out


def capability_steps_from_trace(trace: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not trace:
        return []
    raw = trace.get("capability_steps")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def merge_capability_steps_into_trace(
    trace: dict[str, Any],
    steps: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    merged = dict(trace)
    serialized = serialize_capability_steps(steps)
    if serialized:
        merged["capability_steps"] = serialized
    return merged


def _format_invoke_detail(step: Mapping[str, Any]) -> str:
    action = str(step.get("action") or "invoke").strip()
    query = str(step.get("query") or "").strip()
    if query:
        escaped = query.replace('"', '\\"')
        return f'{action}(query="{escaped}")'
    return action


def format_capability_steps_text(steps: Sequence[Mapping[str, Any]] | None) -> str:
    """Human-readable INSPECT capability step chain."""
    if not steps:
        return ""

    lines = ["Capability inspect:"]
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        number = step.get("step", "?")
        kind = str(step.get("kind") or "step").replace("_", " ").title()
        detail = ""

        if step.get("kind") == "attachment":
            urn = step.get("urn") or "—"
            detail = f"user attached {urn}"
            label = step.get("label")
            tier = step.get("tier")
            if label:
                detail = f"{label} ({urn})"
            if tier:
                detail = f"{detail} [{tier}]"
        elif step.get("kind") == "invoke":
            if not step.get("allowed", True):
                detail = f"denied ({step.get('reason') or 'not permitted'})"
            else:
                detail = _format_invoke_detail(step)
                latency = step.get("latency_ms")
                if latency is not None:
                    detail = f"{detail} ({latency} ms)"
        elif step.get("kind") == "returned":
            count = step.get("raw_count", 0)
            detail = f"{count} result(s)"
        elif step.get("kind") == "ranked":
            kept = step.get("kept_count", 0)
            rejected = step.get("rejected_count", 0)
            detail = f"kept {kept}"
            if rejected:
                detail = f"{detail}, rejected {rejected}"
            threshold = step.get("threshold")
            if threshold is not None:
                detail = f"{detail} (relevance >= {threshold})"
        elif step.get("kind") == "cited":
            cited = step.get("cited_ids") or []
            if cited:
                detail = "model cited " + ", ".join(str(item) for item in cited)
            else:
                detail = "no citations recorded"

        lines.append(f"  {number}  {kind:<11}{detail}")

    return "\n".join(lines)


def format_capability_steps_summary_line(
    steps: Sequence[Mapping[str, Any]] | None,
) -> str:
    if not steps:
        return ""
    kinds = [
        str(step.get("kind") or "?")
        for step in steps
        if isinstance(step, Mapping)
    ]
    return f"Capability steps: {' → '.join(kinds)} ({len(kinds)})"
