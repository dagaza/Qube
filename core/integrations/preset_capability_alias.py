"""Resolve ``@[tool:user:…]`` to preset capability bundles (T17).

Dual grammar (Option A scoped): ``@[cap:…]`` remains canonical for individual
integration capabilities; ``@[tool:user:…]`` is a permanent alias that resolves
to the preset's bundled capability URNs before invoke. Adapter-only presets
(without ``capabilities``) keep the existing knowledge-service routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.capability_inspect import (
    build_attachment_step,
    build_capability_inspect_trace,
)
from core.integrations.capability_invoke import (
    CapabilityInvokeResult,
    invoke_gated_capability,
)
from core.knowledge.presets import load_preset, parse_user_preset_tool

__all__ = [
    "PresetCapabilityBundle",
    "build_preset_capability_inspect_trace",
    "invoke_preset_capability_bundle",
    "normalize_preset_capability_urn",
    "preset_capability_bundle",
    "resolve_preset_capability_urns",
]


@dataclass(frozen=True, slots=True)
class PresetCapabilityBundle:
    """Bundled capability URNs for a My Knowledge preset alias."""

    preset_id: str
    preset_label: str
    urns: tuple[str, ...]


def normalize_preset_capability_urn(raw: str) -> str | None:
    """Return canonical ``cap:…`` string or ``None`` when invalid."""
    parsed = CapabilityURN.try_parse((raw or "").strip())
    if parsed is None:
        return None
    return str(parsed)


def resolve_preset_capability_urns(preset_id: str) -> tuple[str, ...]:
    """Load a preset and return its canonical capability URN strings."""
    preset = load_preset((preset_id or "").strip().lower())
    if preset is None or not preset.capabilities:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in preset.capabilities:
        canonical = normalize_preset_capability_urn(raw)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return tuple(out)


def preset_capability_bundle(tool_id: str) -> PresetCapabilityBundle | None:
    """Resolve ``user:{preset_id}`` to a cap bundle when the preset defines caps."""
    preset_id = parse_user_preset_tool(tool_id)
    if not preset_id:
        return None
    preset = load_preset(preset_id)
    if preset is None:
        return None
    urns = resolve_preset_capability_urns(preset.id)
    if not urns:
        return None
    return PresetCapabilityBundle(
        preset_id=preset.id,
        preset_label=preset.label,
        urns=urns,
    )


def invoke_preset_capability_bundle(
    preset_id: str,
    query: str,
    *,
    max_results: int = 5,
    timeout_s: float = 15.0,
    provider_factory_kwargs: dict[str, Any] | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    agent_scope=None,
    step_approved: bool = False,
) -> tuple[CapabilityInvokeResult, list[CapabilityInvokeResult]]:
    """Invoke each capability in a preset bundle; merge allowed rows."""
    urns = resolve_preset_capability_urns(preset_id)
    if not urns:
        return (
            CapabilityInvokeResult(False, "preset has no capabilities"),
            [],
        )

    per_cap: list[CapabilityInvokeResult] = []
    merged_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    primary_descriptor = None
    primary_urn: CapabilityURN | None = None

    for urn in urns:
        result = invoke_gated_capability(
            urn,
            query,
            max_results=max_results,
            timeout_s=timeout_s,
            provider_factory_kwargs=provider_factory_kwargs,
            session_id=session_id,
            turn_id=turn_id,
            agent_scope=agent_scope,
            step_approved=step_approved,
        )
        per_cap.append(result)
        if result.allowed and result.rows:
            merged_rows.extend(result.rows)
            if primary_descriptor is None:
                primary_descriptor = result.descriptor
                primary_urn = result.urn
        elif not result.allowed:
            reasons.append(f"{urn}: {result.reason}")

    if merged_rows:
        return (
            CapabilityInvokeResult(
                True,
                "ok",
                rows=tuple(merged_rows),
                descriptor=primary_descriptor,
                urn=primary_urn,
            ),
            per_cap,
        )

    if reasons:
        return (
            CapabilityInvokeResult(
                False,
                "; ".join(reasons),
                descriptor=per_cap[-1].descriptor if per_cap else None,
                urn=per_cap[-1].urn if per_cap else None,
            ),
            per_cap,
        )

    return (
        CapabilityInvokeResult(
            False,
            "capability bundle returned no results",
            descriptor=per_cap[-1].descriptor if per_cap else None,
            urn=per_cap[-1].urn if per_cap else None,
        ),
        per_cap,
    )


def build_preset_capability_inspect_trace(
    *,
    preset_id: str,
    preset_label: str,
    query: str,
    per_cap_results: Sequence[CapabilityInvokeResult],
    bundle_result: CapabilityInvokeResult,
    latency_ms: float = 0.0,
) -> list[dict[str, Any]]:
    """Build INSPECT steps for a preset alias → cap bundle turn."""
    alias_token = f"tool:user:{preset_id}"
    steps: list[dict[str, Any]] = [
        build_attachment_step(
            urn=alias_token,
            label=f"{preset_label} (preset alias → {len(per_cap_results)} cap(s))",
            step=1,
        )
    ]

    for index, result in enumerate(per_cap_results, start=1):
        urn = str(result.urn or "")
        cap_steps = build_capability_inspect_trace(
            urn=urn,
            query=query,
            allowed=result.allowed,
            reason=result.reason,
            rows=result.rows,
            bundle_source_count=len(result.rows) if result.allowed else 0,
            rejected_count=0,
            latency_ms=latency_ms,
            descriptor=result.descriptor,
        )
        for step in cap_steps:
            if step.get("kind") == "attachment":
                continue
            merged = dict(step)
            merged["step"] = len(steps) + 1
            merged["bundle_index"] = index
            steps.append(merged)

    if bundle_result.allowed and bundle_result.rows:
        kept = len(bundle_result.rows)
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "ranked",
                "kept_count": kept,
                "rejected_count": 0,
                "bundle": True,
            }
        )

    return steps
