"""
Per-task ChatML prompts, inference params, and parsers for the sidecar model.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from core.cognition_prompt_adapter import (
    apply_qwen3_no_think_to_prompt,
    build_cognition_prompt,
    cognition_stop_tokens,
)
from core.companion_verbal_prompts import (
    COMPANION_LINE_MAX_CHARS,
    build_companion_line_prompt,
    truncate_companion_caption,
)
from core.companion_line_quality import (
    is_acceptable_companion_line,
    is_meta_companion_prose,
    strip_companion_tutorial_prefix,
)
from core.redacted_thinking_filter import strip_reasoning_blocks_from_text
from core.sidecar_types import SidecarResult, SidecarTask

IM_END = "<|im_end|>"
CHATML_STOPS = cognition_stop_tokens("chatml")

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

_VALID_COMPANION_KINDS = frozenset({"idle_quip", "ingest_ack", "download_ack", "skip"})


def task_inference_params(task: SidecarTask) -> dict[str, Any]:
    defaults: dict[SidecarTask, dict[str, Any]] = {
        SidecarTask.title: {"max_tokens": 128, "temperature": 0.1},
        SidecarTask.contradiction_judge: {"max_tokens": 8, "temperature": 0.1},
        SidecarTask.reflection_label: {"max_tokens": 64, "temperature": 0.1},
        SidecarTask.episode_summary: {"max_tokens": 220, "temperature": 0.2},
        SidecarTask.query_rewrite: {"max_tokens": 120, "temperature": 0.15},
        SidecarTask.source_digest: {"max_tokens": 400, "temperature": 0.2},
        SidecarTask.ingest_blurb: {"max_tokens": 48, "temperature": 0.2},
        SidecarTask.companion_line: {"max_tokens": 64, "temperature": 0.35},
    }
    return dict(defaults.get(task, {"max_tokens": 128, "temperature": 0.2}))


def build_prompt_for_task(
    task: SidecarTask,
    *,
    chat_format: str = "chatml",
    model_path: str = "",
    **kwargs: Any,
) -> str:
    def _prompt(system: str, user: str) -> str:
        return build_cognition_prompt(
            system, user, chat_format, model_path=model_path
        )

    def _finish(prebuilt: str) -> str:
        return apply_qwen3_no_think_to_prompt(prebuilt, model_path)

    if task == SidecarTask.title:
        user_prompt = (kwargs.get("user_prompt") or "").strip()
        assistant_reply = (kwargs.get("assistant_reply") or "").strip()
        system = (
            "You name chat conversations for a sidebar history list. Read the user's "
            "first message and the assistant's reply, then write a short topic title "
            "(2-5 words) that summarizes what the chat is about. Do not quote, copy, "
            "or paraphrase the user's opening sentence. No quotes, punctuation-only "
            "lines, or meta commentary. Output ONLY the title on one line."
        )
        return _prompt(
            system,
            format_title_exchange_context(user_prompt, assistant_reply),
        )

    if task == SidecarTask.contradiction_judge:
        old_s = (kwargs.get("old_content") or "").strip()
        new_s = (kwargs.get("new_content") or "").strip()
        system = (
            "Classify the relationship between two short facts ABOUT THE SAME USER. "
            "Respond with EXACTLY one word: duplicate, contradiction, or complement."
        )
        user = f"A: {old_s}\nB: {new_s}\n\nAnswer:"
        return _prompt(system, user)

    if task == SidecarTask.reflection_label:
        prebuilt = kwargs.get("prompt")
        if prebuilt:
            return _finish(str(prebuilt))
        return _prompt(
            "Return STRICT JSON: {\"label\": \"durable_user_fact|...\"}",
            (kwargs.get("content") or "")[:800],
        )

    if task == SidecarTask.episode_summary:
        prebuilt = kwargs.get("prompt")
        if prebuilt:
            return _finish(str(prebuilt))
        return _prompt(
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
        return _prompt(system, user)

    if task == SidecarTask.source_digest:
        sources_text = (kwargs.get("sources_text") or "").strip()[:8000]
        system = (
            "Compress retrieval sources into claim-oriented bullets. "
            "Keep each source citation id [N] from the input. "
            "One or two bullets per source. No prose outside bullets."
        )
        return _prompt(system, sources_text)

    if task == SidecarTask.ingest_blurb:
        sample = (kwargs.get("sample_text") or "").strip()[:2500]
        system = (
            "Write ONE sentence describing what this document is about. "
            "No quotes. No markdown. Under 30 words."
        )
        return _prompt(system, sample)

    if task == SidecarTask.companion_line:
        from core import app_settings

        payload = dict(kwargs)
        if payload.get("expression_level") is not None or payload.get("thought"):
            from core.companion_cognition.expression_prompts import build_companion_line_prompt_v2

            return build_companion_line_prompt_v2(
                chat_format=chat_format,
                model_path=model_path,
                payload=payload,
                trait_preset=kwargs.get("trait_preset")
                or app_settings.get_companion_verbal_trait_preset(),
                user_system_prompt=kwargs.get("user_system_prompt")
                or app_settings.get_companion_verbal_system_prompt(),
            )

        return build_companion_line_prompt(
            chat_format=chat_format,
            model_path=model_path,
            trait_preset=kwargs.get("trait_preset")
            or app_settings.get_companion_verbal_trait_preset(),
            user_system_prompt=kwargs.get("user_system_prompt")
            or app_settings.get_companion_verbal_system_prompt(),
            trigger=str(kwargs.get("trigger") or "idle"),
            file_count=kwargs.get("file_count"),
            filename=kwargs.get("filename"),
            basename=kwargs.get("basename"),
        )

    return _prompt("Respond concisely.", str(kwargs))


def _normalize_sidecar_completion_text(raw: str) -> str:
    return strip_reasoning_blocks_from_text((raw or "").strip())


_POST_THINK_TAIL_RE = re.compile(
    r"(?is)</(?:redacted_)?think(?:ing)?>\s*(.+)$"
)
_THINK_INTERIOR_RE = re.compile(
    r"(?is)<(?:redacted_)?think(?:ing)?>\s*(.*?)(?:</(?:redacted_)?think(?:ing)?>|$)"
)
_TITLE_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_META_TITLE_LINE = re.compile(
    r"(?i)\b(user|message|title|titling|engine|output|respond|extract)\b"
)
_TITLE_USER_MAX_CHARS = 1200
_TITLE_ASSISTANT_MAX_CHARS = 600
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "please", "help", "tell", "explain", "show",
    "write", "give", "make", "need", "want", "like", "just", "also", "about",
})


def format_title_exchange_context(
    user_prompt: str,
    assistant_reply: str = "",
) -> str:
    """Bounded first-turn context for sidecar titling."""
    user = (user_prompt or "").strip()
    assistant = (assistant_reply or "").strip()
    parts: list[str] = []
    if user:
        parts.append(f"User: {user[:_TITLE_USER_MAX_CHARS]}")
    if assistant:
        parts.append(f"Assistant: {assistant[:_TITLE_ASSISTANT_MAX_CHARS]}")
    return "\n\n".join(parts) if parts else user


def _normalize_title_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip(" \"'.,!?;:")


def _title_is_verbatim_of_prompt(title: str, user_prompt: str) -> bool:
    title_norm = _normalize_title_compare(title)
    prompt_norm = _normalize_title_compare(user_prompt)
    if not title_norm or not prompt_norm:
        return False
    if title_norm == prompt_norm:
        return True
    if len(title_norm) >= 12 and (
        title_norm in prompt_norm or prompt_norm.startswith(title_norm)
    ):
        return True
    title_words = _TITLE_WORD_RE.findall(title_norm)
    prompt_words = _TITLE_WORD_RE.findall(prompt_norm)
    if len(title_words) >= 3 and title_words == prompt_words[: len(title_words)]:
        return True
    return False


def _topic_words_from_text(text: str, *, max_words: int = 5) -> list[str]:
    words = _TITLE_WORD_RE.findall((text or "").strip())
    picked: list[str] = []
    for word in words:
        low = word.lower()
        if low in _TITLE_STOPWORDS:
            continue
        picked.append(word)
        if len(picked) >= max_words:
            break
    if len(picked) < 2:
        picked = words[:max_words]
    return picked[:max_words]


def _polish_title_candidate(candidate: str) -> str:
    line = re.sub(r"\s+", " ", (candidate or "").replace('"', "").replace("'", "")).strip()
    if not line or re.search(r"<\s*/?\s*think", line, re.IGNORECASE):
        return ""
    words = line.split()
    if len(words) < 2 or len(words) > 8:
        return ""
    if _META_TITLE_LINE.search(line):
        return ""
    return line.title()


def _fallback_title_from_exchange(
    user_prompt: str,
    *,
    assistant_reply: str = "",
) -> str:
    for source in ((assistant_reply or "").strip(), (user_prompt or "").strip()):
        if not source:
            continue
        topic_words = _topic_words_from_text(source)
        if len(topic_words) < 2:
            continue
        snippet = " ".join(topic_words)
        if len(snippet) > 48:
            snippet = snippet[:48].rsplit(" ", 1)[0] or snippet[:48]
        polished = _polish_title_candidate(snippet) or snippet.title()
        if polished and not _title_is_verbatim_of_prompt(polished, user_prompt):
            return polished
    return ""


def _fallback_title_from_user_prompt(user_prompt: str) -> str:
    return _fallback_title_from_exchange(user_prompt)


def _title_from_post_think_tail(raw: str) -> str:
    m = _POST_THINK_TAIL_RE.search(raw or "")
    if not m:
        return ""
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    for line in reversed(lines):
        polished = _polish_title_candidate(line)
        if polished:
            return polished
    return ""


def _title_from_think_interior(raw: str) -> str:
    m = _THINK_INTERIOR_RE.search(raw or "")
    if not m:
        return ""
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    for line in sorted(lines, key=lambda ln: len(ln.split())):
        polished = _polish_title_candidate(line)
        if polished:
            return polished
    return ""


def _accept_title_candidate(
    candidate: str,
    *,
    user_prompt: str,
) -> str:
    polished = _polish_title_candidate(candidate)
    if not polished:
        return ""
    if _title_is_verbatim_of_prompt(polished, user_prompt):
        return ""
    return polished


def _finalize_title_text(
    raw: str,
    *,
    user_prompt: str = "",
    assistant_reply: str = "",
) -> str:
    """Single-line title after stripping Qwen3 / R1-style reasoning blocks."""
    raw_s = (raw or "").strip()
    if not raw_s:
        return _fallback_title_from_exchange(
            user_prompt, assistant_reply=assistant_reply
        )

    cleaned = _normalize_sidecar_completion_text(raw_s)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for line in reversed(lines):
        accepted = _accept_title_candidate(line, user_prompt=user_prompt)
        if accepted:
            return accepted
        coerced_words = _topic_words_from_text(line)
        if len(coerced_words) >= 2:
            coerced = _accept_title_candidate(
                " ".join(coerced_words), user_prompt=user_prompt
            )
            if coerced:
                return coerced

    for extractor in (_title_from_post_think_tail,):
        polished = extractor(raw_s)
        if polished and not _title_is_verbatim_of_prompt(polished, user_prompt):
            return polished

    fallback = _fallback_title_from_exchange(
        user_prompt, assistant_reply=assistant_reply
    )
    if fallback:
        return fallback

    interior = _title_from_think_interior(raw_s)
    if interior and not _title_is_verbatim_of_prompt(interior, user_prompt):
        return interior

    return ""


def parse_task_output(task: SidecarTask, raw: str, **kwargs: Any) -> SidecarResult:
    raw_s = (raw or "").strip()
    if not raw_s:
        return SidecarResult(ok=False, error="empty_output", task=task)

    if task == SidecarTask.title:
        title = _finalize_title_text(
            raw_s,
            user_prompt=str(kwargs.get("user_prompt") or ""),
            assistant_reply=str(kwargs.get("assistant_reply") or ""),
        )
        return SidecarResult(
            text=title,
            parsed={"title": title},
            confidence=0.9 if title else 0.0,
            ok=bool(title),
            task=task,
        )

    text = _normalize_sidecar_completion_text(raw_s)
    if not text:
        return SidecarResult(ok=False, error="empty_output", task=task)

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

    if task == SidecarTask.companion_line:
        trigger = str(kwargs.get("trigger") or "idle")
        parsed = _parse_companion_line_json(text)
        if not parsed:
            parsed = _parse_companion_line_fallback(text, trigger=trigger)
        if not parsed:
            return SidecarResult(ok=False, error="parse_fail", task=task, text=text)
        kind = str(parsed.get("kind") or "skip")
        line = str(parsed.get("line") or "").strip()
        if kind == "skip" or not line:
            return SidecarResult(
                text=line,
                parsed=parsed,
                confidence=0.0,
                ok=False,
                error="skip",
                task=task,
            )
        if not is_acceptable_companion_line(line):
            return SidecarResult(
                text=line,
                parsed=parsed,
                confidence=0.0,
                ok=False,
                error="low_quality",
                task=task,
            )
        conf = 0.7 if parsed.get("_strict_json") else 0.55
        return SidecarResult(
            text=line,
            parsed=parsed,
            confidence=conf,
            ok=True,
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


def _companion_line_kind_for_trigger(trigger: str) -> str:
    trig = str(trigger or "idle").strip().lower()
    if trig == "ingest_complete":
        return "ingest_ack"
    if trig == "download_complete":
        return "download_ack"
    return "idle_quip"


def _normalize_companion_line_dict(
    line: str,
    *,
    trigger: str,
    strict_json: bool = False,
) -> Optional[dict[str, Any]]:
    kind = _companion_line_kind_for_trigger(trigger)
    cleaned = truncate_companion_caption(
        re.sub(r"\s+", " ", (line or "")).strip().strip("\"'"),
        COMPANION_LINE_MAX_CHARS,
    )
    if not is_acceptable_companion_line(cleaned):
        return None
    return {
        "line": cleaned,
        "kind": kind,
        "trigger": trigger,
        "_strict_json": strict_json,
    }


def _parse_companion_line_json(raw: str) -> Optional[dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    kind = str(obj.get("kind") or "skip").strip().lower()
    if kind not in _VALID_COMPANION_KINDS:
        kind = "skip"
    line = truncate_companion_caption(
        re.sub(r"\s+", " ", str(obj.get("line") or "")).strip().strip('"\''),
        COMPANION_LINE_MAX_CHARS,
    )
    result = {
        "line": line,
        "kind": kind,
        "trigger": str(obj.get("trigger") or ""),
        "_strict_json": True,
    }
    return result


def _parse_companion_line_fallback(raw: str, *, trigger: str = "idle") -> Optional[dict[str, Any]]:
    """Salvage a caption when the cognition model returns prose instead of JSON."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    m_line = re.search(
        r'["\']line["\']\s*:\s*["\']([^"\']{3,72})',
        text,
        flags=re.IGNORECASE,
    )
    if m_line:
        line = m_line.group(1).strip()
        if line and line.lower() != "skip" and not is_meta_companion_prose(line):
            parsed = _normalize_companion_line_dict(line, trigger=trigger, strict_json=False)
            if parsed:
                return parsed

    quotes = re.findall(r'"([^"\n]{3,72})"', text) or re.findall(r"'([^'\n]{3,72})'", text)
    if quotes:
        meta = (
            "welcome",
            "settings",
            "preview",
            "sample",
            "json",
            "companion caption",
            "write your own",
            "something about",
            "maybe something",
        )

        def _quote_score(candidate: str) -> int:
            low = candidate.lower()
            penalty = sum(50 for token in meta if token in low)
            if not is_acceptable_companion_line(candidate):
                penalty += 200
            return len(candidate) + penalty

        candidates = [q for q in quotes if q.strip().lower() not in ("skip",)]
        if candidates:
            line = min(candidates, key=_quote_score)
            if line and _quote_score(line) < 120:
                parsed = _normalize_companion_line_dict(line, trigger=trigger)
                if parsed:
                    return parsed

    for pattern in (
        r"sample\s*:\s*[\"']?([^\"'\n]{3,72})",
        r"caption\s*:\s*[\"']?([^\"'\n]{3,72})",
        r"line\s*:\s*[\"']?([^\"'\n]{3,72})",
    ):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            line = m.group(1).strip().strip('"\'')
            if line:
                parsed = _normalize_companion_line_dict(line, trigger=trigger)
                if parsed:
                    return parsed

    flat = re.sub(r"\s+", " ", text).strip().strip('"\'')
    sentences = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"[.!?]+", flat)
        if part.strip()
    ]
    for candidate in reversed(sentences):
        if 3 <= len(candidate) <= COMPANION_LINE_MAX_CHARS and not is_meta_companion_prose(
            candidate
        ):
            parsed = _normalize_companion_line_dict(candidate, trigger=trigger)
            if parsed:
                return parsed

    if len(flat) > COMPANION_LINE_MAX_CHARS:
        flat = flat[:COMPANION_LINE_MAX_CHARS].rstrip()
    if 3 <= len(flat) <= COMPANION_LINE_MAX_CHARS and not is_meta_companion_prose(flat):
        parsed = _normalize_companion_line_dict(flat, trigger=trigger)
        if parsed:
            return parsed

    stripped = strip_companion_tutorial_prefix(text)
    if stripped and stripped != flat:
        if 3 <= len(stripped) <= COMPANION_LINE_MAX_CHARS and not is_meta_companion_prose(stripped):
            parsed = _normalize_companion_line_dict(stripped, trigger=trigger)
            if parsed:
                return parsed
        for candidate in reversed(
            [p.strip() for p in re.split(r"[.!?]+", stripped) if p.strip()]
        ):
            if (
                3 <= len(candidate) <= COMPANION_LINE_MAX_CHARS
                and not is_meta_companion_prose(candidate)
            ):
                parsed = _normalize_companion_line_dict(candidate, trigger=trigger)
                if parsed:
                    return parsed

    return None
