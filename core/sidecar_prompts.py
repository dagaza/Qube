"""
Per-task ChatML prompts, inference params, and parsers for the sidecar model.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from core.sidecar_types import SidecarResult, SidecarTask

IM_END = "<|im_end|>"
CHATML_STOPS = [IM_END, "\n\n"]

_VALID_REFLECTION_LABELS = frozenset({
    "durable_user_fact",
    "third_party_stub",
    "system_claim",
    "transient",
    "unclear",
    "tier_mismatch",
    "orphan_knowledge",
})

_JUDGE_WORDS = ("duplicate", "contradiction", "complement")


def task_inference_params(task: SidecarTask) -> dict[str, Any]:
    defaults: dict[SidecarTask, dict[str, Any]] = {
        SidecarTask.title: {"max_tokens": 12, "temperature": 0.2},
        SidecarTask.contradiction_judge: {"max_tokens": 8, "temperature": 0.1},
        SidecarTask.reflection_label: {"max_tokens": 64, "temperature": 0.1},
        SidecarTask.episode_summary: {"max_tokens": 220, "temperature": 0.2},
        SidecarTask.query_rewrite: {"max_tokens": 120, "temperature": 0.15},
        SidecarTask.source_digest: {"max_tokens": 400, "temperature": 0.2},
        SidecarTask.ingest_blurb: {"max_tokens": 48, "temperature": 0.2},
    }
    return dict(defaults.get(task, {"max_tokens": 128, "temperature": 0.2}))


def _chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user}{IM_END}\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_for_task(task: SidecarTask, **kwargs: Any) -> str:
    if task == SidecarTask.title:
        user_prompt = (kwargs.get("user_prompt") or "").strip()
        system = (
            "You are an automated titling engine. Extract the thematic core of the "
            "user's message into a 2-4 word title. No filler, punctuation, or quotes. "
            "Output ONLY the title text."
        )
        return _chatml(system, user_prompt)

    if task == SidecarTask.contradiction_judge:
        old_s = (kwargs.get("old_content") or "").strip()
        new_s = (kwargs.get("new_content") or "").strip()
        system = (
            "Classify the relationship between two short facts ABOUT THE SAME USER. "
            "Respond with EXACTLY one word: duplicate, contradiction, or complement."
        )
        user = f"A: {old_s}\nB: {new_s}\n\nAnswer:"
        return _chatml(system, user)

    if task == SidecarTask.reflection_label:
        return kwargs.get("prompt") or _chatml(
            "Return STRICT JSON: {\"label\": \"durable_user_fact|...\"}",
            (kwargs.get("content") or "")[:800],
        )

    if task == SidecarTask.episode_summary:
        return kwargs.get("prompt") or _chatml(
            "Summarize the conversation per instructions.",
            (kwargs.get("conversation") or "")[:6000],
        )

    if task == SidecarTask.query_rewrite:
        original = (kwargs.get("original_query") or "").strip()
        topic = (kwargs.get("topic") or "").strip()
        kind = (kwargs.get("follow_up_kind") or "none").strip()
        tail = (kwargs.get("history_tail") or "").strip()[:1200]
        system = (
            "Expand deictic follow-up queries using the active topic ONLY. "
            "Do NOT invent levels, bosses, names, or facts not in the topic or history. "
            "If unsure, set expanded_query to the original and confidence below 0.5. "
            "Return STRICT JSON only: "
            '{"expanded_query":"...","confidence":0.0,"topic_source":"discourse_state|none"}'
        )
        user = (
            f"original_query: {original}\n"
            f"active_topic: {topic or '(none)'}\n"
            f"follow_up_kind: {kind}\n"
            f"recent_turns:\n{tail}"
        )
        return _chatml(system, user)

    if task == SidecarTask.source_digest:
        sources_text = (kwargs.get("sources_text") or "").strip()[:8000]
        system = (
            "Compress retrieval sources into claim-oriented bullets. "
            "Keep each source citation id [N] from the input. "
            "One or two bullets per source. No prose outside bullets."
        )
        return _chatml(system, sources_text)

    if task == SidecarTask.ingest_blurb:
        sample = (kwargs.get("sample_text") or "").strip()[:2500]
        system = (
            "Write ONE sentence describing what this document is about. "
            "No quotes. No markdown. Under 30 words."
        )
        return _chatml(system, sample)

    return _chatml("Respond concisely.", str(kwargs))


def parse_task_output(task: SidecarTask, raw: str, **kwargs: Any) -> SidecarResult:
    text = (raw or "").strip()
    if not text:
        return SidecarResult(ok=False, error="empty_output", task=task)

    if task == SidecarTask.title:
        title = text.replace('"', "").replace("'", "").strip()
        if title:
            title = title.title()
        return SidecarResult(
            text=title,
            parsed={"title": title},
            confidence=0.9 if title else 0.0,
            ok=bool(title),
            task=task,
        )

    if task == SidecarTask.contradiction_judge:
        low = text.lower()
        first = low.split()[0] if low else ""
        if first.startswith("duplicate") or "duplicate" in low:
            verdict = "duplicate"
        elif first.startswith("contradict") or "contradiction" in low:
            verdict = "contradiction"
        elif first.startswith("complement") or "complement" in low:
            verdict = "complement"
        else:
            verdict = "complement"
        return SidecarResult(
            text=verdict,
            parsed={"verdict": verdict},
            confidence=0.85,
            ok=True,
            task=task,
        )

    if task == SidecarTask.reflection_label:
        label = _parse_reflection_label(text)
        return SidecarResult(
            text=label,
            parsed={"label": label},
            confidence=0.8 if label != "unclear" else 0.3,
            ok=True,
            task=task,
        )

    if task == SidecarTask.episode_summary:
        summary, topics = _parse_episode_lines(text)
        return SidecarResult(
            text=text,
            parsed={"summary": summary, "topics": topics},
            confidence=0.75 if summary and summary.upper() != "SKIP" else 0.2,
            ok=bool(summary),
            task=task,
        )

    if task == SidecarTask.query_rewrite:
        parsed = _parse_query_rewrite_json(text)
        if not parsed:
            return SidecarResult(ok=False, error="parse_fail", task=task, text=text)
        conf = float(parsed.get("confidence") or 0.0)
        return SidecarResult(
            text=str(parsed.get("expanded_query") or ""),
            parsed=parsed,
            confidence=conf,
            ok=True,
            task=task,
        )

    if task == SidecarTask.source_digest:
        if not _digest_preserves_citation_ids(text, kwargs.get("expected_ids") or []):
            return SidecarResult(ok=False, error="citation_ids_missing", task=task, text=text)
        return SidecarResult(
            text=text[:1200],
            parsed={"digest": text[:1200]},
            confidence=0.7,
            ok=True,
            task=task,
        )

    if task == SidecarTask.ingest_blurb:
        blurb = re.sub(r"\s+", " ", text).strip()[:300]
        return SidecarResult(
            text=blurb,
            parsed={"blurb": blurb},
            confidence=0.75 if len(blurb) > 10 else 0.2,
            ok=bool(blurb),
            task=task,
        )

    return SidecarResult(text=text, ok=bool(text), task=task, confidence=0.5)


def _parse_reflection_label(raw: str) -> str:
    match = re.search(r"\{[^{}]*\}", raw)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                label = str(obj.get("label") or "").strip().lower()
                if label in _VALID_REFLECTION_LABELS:
                    return label
        except Exception:
            pass
    lower = raw.lower()
    for label in _VALID_REFLECTION_LABELS:
        if label in lower:
            return label
    return "unclear"


def _parse_episode_lines(raw: str) -> tuple[str, list[str]]:
    text = raw.strip()
    summary = ""
    topics: list[str] = []
    m_sum = re.search(
        r"SUMMARY\s*:\s*(.*?)(?:\n\s*TOPICS\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_sum:
        summary = re.sub(r"\s+", " ", m_sum.group(1)).strip()
    m_top = re.search(r"TOPICS\s*:\s*(.*?)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m_top:
        raw_topics = m_top.group(1).strip()
        topics = [
            t.strip().lower()
            for t in re.split(r"[,\n]", raw_topics)
            if t.strip()
        ][:6]
    return summary, topics


def _parse_query_rewrite_json(raw: str) -> Optional[dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    expanded = str(obj.get("expanded_query") or "").strip()
    if not expanded:
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    topic_source = str(obj.get("topic_source") or "discourse_state").strip() or "discourse_state"
    return {
        "expanded_query": expanded,
        "confidence": conf,
        "topic_source": topic_source,
    }


def _digest_preserves_citation_ids(text: str, expected_ids: list) -> bool:
    if not expected_ids:
        return True
    for eid in expected_ids:
        token = f"[{eid}]"
        if token not in text:
            return False
    return True
