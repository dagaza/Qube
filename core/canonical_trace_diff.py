"""
Provider-agnostic comparator for two canonical LLM execution traces.

Compares baseline (A) vs test (B) traces without vendor- or model-specific assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.canonical_fingerprint import fingerprint_canonical_request, fingerprint_text
from core.canonical_request import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalRequestExporter,
    CanonicalSampling,
)

DivergenceLevel = Literal["REQUEST", "PROMPT", "OUTPUT"]


def build_trace_fingerprints(
    request: CanonicalRequest,
    prompt: str,
    output: str,
) -> dict[str, Any]:
    return {
        "request": fingerprint_canonical_request(request),
        "prompt": fingerprint_text(prompt or ""),
        "output": fingerprint_text(output or ""),
    }


@dataclass
class CanonicalTrace:
    request: CanonicalRequest
    prompt: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        fps = self.fingerprints or build_trace_fingerprints(
            self.request, self.prompt, self.output
        )
        return {
            "request": self.request.to_dict(),
            "prompt": self.prompt or "",
            "output": self.output or "",
            "fingerprints": dict(fps),
            "metadata": dict(self.metadata or {}),
        }


def _coerce_sampling(raw: Any) -> CanonicalSampling:
    if isinstance(raw, CanonicalSampling):
        return raw
    data = raw if isinstance(raw, dict) else {}
    return CanonicalSampling(
        temperature=float(data.get("temperature", 1.0)),
        top_p=float(data.get("top_p", 1.0)),
        top_k=data.get("top_k"),
        repeat_penalty=data.get("repeat_penalty"),
        presence_penalty=data.get("presence_penalty"),
        frequency_penalty=data.get("frequency_penalty"),
    )


def _coerce_messages(raw: Any) -> list[CanonicalMessage]:
    if not isinstance(raw, list):
        return []
    out: list[CanonicalMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            CanonicalMessage(
                role=str(item.get("role") or "user"),
                content=str(item.get("content") or ""),
            )
        )
    return out


def _coerce_request(raw: Any) -> CanonicalRequest:
    if isinstance(raw, CanonicalRequest):
        return raw
    if isinstance(raw, dict):
        if "messages" in raw or "model" in raw or "sampling" in raw:
            return CanonicalRequest(
                model=str(raw.get("model") or ""),
                messages=_coerce_messages(raw.get("messages")),
                sampling=_coerce_sampling(raw.get("sampling")),
                stop=list(raw.get("stop") or []),
                metadata=dict(raw.get("metadata") or {}),
            )
        return CanonicalRequestExporter.export_canonical_request(raw)
    return CanonicalRequest(
        model="",
        messages=[],
        sampling=CanonicalSampling(),
    )


def coerce_canonical_trace(trace: CanonicalTrace | dict[str, Any]) -> CanonicalTrace:
    if isinstance(trace, CanonicalTrace):
        return trace
    data = dict(trace or {})
    request = _coerce_request(data.get("request"))
    prompt = str(data.get("prompt") or "")
    output = str(data.get("output") or "")
    fingerprints = dict(data.get("fingerprints") or {})
    if not fingerprints:
        fingerprints = build_trace_fingerprints(request, prompt, output)
    return CanonicalTrace(
        request=request,
        prompt=prompt,
        output=output,
        metadata=dict(data.get("metadata") or {}),
        fingerprints=fingerprints,
    )


def _messages_to_dicts(messages: list[CanonicalMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _sampling_dict(sampling: CanonicalSampling) -> dict[str, Any]:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "repeat_penalty": sampling.repeat_penalty,
        "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
    }


def _compare_canonical_request(
    req_a: CanonicalRequest,
    req_b: CanonicalRequest,
) -> tuple[bool, list[dict[str, Any]]]:
    fp_a = fingerprint_canonical_request(req_a)
    fp_b = fingerprint_canonical_request(req_b)
    if fp_a["sha256"] == fp_b["sha256"]:
        return True, []
    return False, [
        {
            "level": "REQUEST",
            "aspect": "canonical_request",
            "baseline_fingerprint": fp_a,
            "test_fingerprint": fp_b,
            "summary": (
                f"canonical request fingerprint mismatch "
                f"(baseline={fp_a['short']}, test={fp_b['short']})"
            ),
        }
    ]


def _compare_messages(
    req_a: CanonicalRequest,
    req_b: CanonicalRequest,
) -> tuple[bool, list[dict[str, Any]]]:
    msgs_a = _messages_to_dicts(req_a.messages)
    msgs_b = _messages_to_dicts(req_b.messages)
    if msgs_a == msgs_b:
        return True, []
    diff: dict[str, Any] = {
        "level": "REQUEST",
        "aspect": "messages",
        "baseline_count": len(msgs_a),
        "test_count": len(msgs_b),
        "summary": (
            f"message list mismatch (baseline={len(msgs_a)} messages, "
            f"test={len(msgs_b)} messages)"
        ),
    }
    if len(msgs_a) == len(msgs_b):
        for idx, (left, right) in enumerate(zip(msgs_a, msgs_b)):
            if left != right:
                diff["first_index"] = idx
                diff["baseline_message"] = left
                diff["test_message"] = right
                break
    return False, [diff]


def _compare_sampling(
    req_a: CanonicalRequest,
    req_b: CanonicalRequest,
) -> tuple[bool, list[dict[str, Any]]]:
    samp_a = _sampling_dict(req_a.sampling)
    samp_b = _sampling_dict(req_b.sampling)
    if samp_a == samp_b:
        return True, []
    differences: list[dict[str, Any]] = []
    for key in sorted(set(samp_a) | set(samp_b)):
        if samp_a.get(key) != samp_b.get(key):
            differences.append(
                {
                    "level": "REQUEST",
                    "aspect": "sampling",
                    "field": key,
                    "baseline": samp_a.get(key),
                    "test": samp_b.get(key),
                    "summary": (
                        f"sampling.{key} mismatch "
                        f"(baseline={samp_a.get(key)!r}, test={samp_b.get(key)!r})"
                    ),
                }
            )
    return False, differences


def _compare_prompt(prompt_a: str, prompt_b: str) -> tuple[bool, list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    strings_match = (prompt_a or "") == (prompt_b or "")
    fp_a = fingerprint_text(prompt_a or "")
    fp_b = fingerprint_text(prompt_b or "")
    fingerprints_match = fp_a["sha256"] == fp_b["sha256"]

    if not strings_match:
        differences.append(
            {
                "level": "PROMPT",
                "aspect": "string",
                "baseline_length": len(prompt_a or ""),
                "test_length": len(prompt_b or ""),
                "summary": (
                    f"prompt string mismatch "
                    f"(baseline_len={len(prompt_a or '')}, test_len={len(prompt_b or '')})"
                ),
            }
        )
    if not fingerprints_match:
        differences.append(
            {
                "level": "PROMPT",
                "aspect": "fingerprint",
                "baseline_fingerprint": fp_a,
                "test_fingerprint": fp_b,
                "summary": (
                    f"prompt fingerprint mismatch "
                    f"(baseline={fp_a['short']}, test={fp_b['short']})"
                ),
            }
        )
    return strings_match and fingerprints_match, differences


def _compare_output(output_a: str, output_b: str) -> tuple[bool, list[dict[str, Any]]]:
    if (output_a or "") == (output_b or ""):
        return True, []
    fp_a = fingerprint_text(output_a or "")
    fp_b = fingerprint_text(output_b or "")
    return False, [
        {
            "level": "OUTPUT",
            "aspect": "string",
            "baseline_length": len(output_a or ""),
            "test_length": len(output_b or ""),
            "baseline_fingerprint": fp_a,
            "test_fingerprint": fp_b,
            "summary": (
                f"output string mismatch "
                f"(baseline_len={len(output_a or '')}, test_len={len(output_b or '')}, "
                f"baseline_fp={fp_a['short']}, test_fp={fp_b['short']})"
            ),
        }
    ]


def find_first_divergence(
    trace_a: CanonicalTrace | dict[str, Any],
    trace_b: CanonicalTrace | dict[str, Any],
) -> dict[str, Any]:
    """
    Compare baseline trace A against test trace B in REQUEST → PROMPT → OUTPUT order.

    Returns match flags, the first diverging level, a human summary, and detail rows.
    """
    a = coerce_canonical_trace(trace_a)
    b = coerce_canonical_trace(trace_b)

    differences: list[dict[str, Any]] = []

    req_full_match = True
    for compare in (_compare_canonical_request, _compare_messages, _compare_sampling):
        matched, diffs = compare(a.request, b.request)
        if not matched:
            req_full_match = False
            differences.extend(diffs)

    prompt_match, prompt_diffs = _compare_prompt(a.prompt, b.prompt)
    if not prompt_match:
        differences.extend(prompt_diffs)

    output_match, output_diffs = _compare_output(a.output, b.output)
    if not output_diffs:
        pass
    elif not output_match:
        differences.extend(output_diffs)

    if not req_full_match:
        first_level: DivergenceLevel | None = "REQUEST"
        diff_summary = differences[0].get("summary", "request mismatch")
    elif not prompt_match:
        first_level = "PROMPT"
        diff_summary = next(
            (d.get("summary") for d in prompt_diffs if d.get("summary")),
            "prompt mismatch",
        )
    elif not output_match:
        first_level = "OUTPUT"
        diff_summary = output_diffs[0].get("summary", "output mismatch")
    else:
        first_level = None
        diff_summary = "traces match"

    result: dict[str, Any] = {
        "request_match": req_full_match,
        "prompt_match": prompt_match,
        "output_match": output_match,
        "first_divergence_level": first_level,
        "diff_summary": diff_summary,
        "differences": differences,
    }
    return result


def traces_equal(
    trace_a: CanonicalTrace | dict[str, Any],
    trace_b: CanonicalTrace | dict[str, Any],
) -> bool:
    """True when find_first_divergence reports no differences at any level."""
    report = find_first_divergence(trace_a, trace_b)
    return (
        report["request_match"]
        and report["prompt_match"]
        and report["output_match"]
    )
