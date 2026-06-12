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
            "first message and the assistant's reply, then write a short topic label "
            "(2-5 words) for the SUBJECT — like a folder name, not a sentence. "
            "Name the topic, not the assignment: ignore word counts, format requirements, "
            "and task verbs (write, essay, comprehensive, draft). "
            "Use the core topic name when obvious (e.g. 'Lord of the Rings', "
            "'Nginx Reverse Proxy', 'Human Problem Solving'). "
            "Examples: "
            "'Write a 1000-word essay on climate change' → Climate Change; "
            "'Draft a scholarly paper on quantum tunneling' → Quantum Tunneling. "
            "Do not describe plot, list characters, or write meta commentary. "
            "No quotes. Output ONLY the title on one line."
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
        conversation = (kwargs.get("conversation") or "")[:6000]
        system = (
            "You are writing a single-paragraph summary of the recent conversation below.\n\n"
            "Goal: capture what the user was DOING or DECIDING in this session, so the "
            'assistant can later answer "what have we been working on?" or '
            '"recap this conversation".\n\n'
            "STRICT RULES:\n"
            "- One paragraph. <= 120 words. Plain English prose.\n"
            "- Describe the USER's project / goal / decisions / open questions. "
            "Never describe the assistant.\n"
            "- Never invent facts that are not in the conversation.\n"
            "- If the conversation is small talk, trivial, or has no narrative arc, "
            "output EXACTLY:\n"
            "  SUMMARY: SKIP\n"
            "  TOPICS:\n"
            "- Otherwise output EXACTLY this format (two labeled lines):\n\n"
            "SUMMARY: <one paragraph summary>\n"
            "TOPICS: <comma-separated short topic keywords>"
        )
        return _prompt(system, conversation)

    if task == SidecarTask.query_rewrite:
        original = (kwargs.get("original_query") or "").strip()
        entity = (kwargs.get("topic") or "").strip()
        aspect = (kwargs.get("active_aspect") or "").strip()
        kind = (kwargs.get("follow_up_kind") or "none").strip()
        tail = (kwargs.get("history_tail") or "").strip()[:1200]
        tentative_route = (kwargs.get("tentative_route") or "none").strip().lower()
        retrieval_query = (kwargs.get("retrieval_query") or original).strip()
        system = (
            "Expand deictic follow-up queries using the conversation entity ONLY. "
            "The entity is the durable subject (city, person, game); the aspect is "
            "the current facet being discussed. Do NOT invent names or latch onto "
            "examples from prior assistant replies. "
            "tentative_route and retrieval_query are read-only context from the "
            "cognitive router — do NOT override routing; only improve the search query. "
            "If unsure, set expanded_query to the original and confidence below 0.5. "
            "recommended_target is optional telemetry (chat|memory|rag|web); "
            "echo tentative_route when unsure. "
            "Return STRICT JSON only: "
            '{"expanded_query":"...","confidence":0.0,'
            '"topic_source":"discourse_state|none",'
            '"recommended_target":"chat|memory|rag|web|none"}'
        )
        aspect_line = f"current_aspect: {aspect}\n" if aspect else ""
        user = (
            f"original_query: {original}\n"
            f"retrieval_query: {retrieval_query}\n"
            f"tentative_route: {tentative_route}\n"
            f"conversation_entity: {entity or '(none)'}\n"
            f"{aspect_line}"
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
    "generate", "comprehensive", "detailed", "overview", "summary", "story",
})
_TITLE_FRAGMENT_VERBS = frozenset({
    "follows", "explains", "describes", "discusses", "covers", "explores",
    "begins", "starts", "tells", "walks", "details", "summarizes", "outlines",
    "lets", "involves", "features", "centers", "focuses", "revolves",
})
_TITLE_SMALL_WORDS = frozenset({"of", "the", "and", "in", "a", "an", "for", "to"})
_OF_THE_PHRASE_RE = re.compile(
    r"\b([A-Z][\w']+\s+of\s+the\s+[A-Z][\w']+(?:\s+[A-Z][\w']+)?)\b"
)
_PHRASE_LEAD_SKIP = frozenset({
    "generate", "comprehensive", "detailed", "explanation", "overview", "summary",
    "describe", "discuss", "tell", "write", "give", "provide", "create",
})
_TITLE_INSTRUCTION_WORDS = frozenset({
    "essay", "scholarly", "comprehensive", "least", "words", "word", "pages", "page",
    "minimum", "maximum", "paragraph", "paragraphs", "draft", "approximately",
    "roughly", "assignment", "paper",
})
_TITLE_TASK_VERBS = frozenset({
    "write", "compose", "draft", "generate", "create", "produce", "prepare",
})
_TITLE_FORMAT_CUES = frozenset({
    "words", "word", "pages", "page", "least", "minimum", "essay", "paper",
})
_SUBJECT_CLAUSE_RE = re.compile(
    r"\b(?:on|about|regarding|concerning)\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_HEADING_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)
_TITLE_MIN_ACCEPT_SCORE = 4.0
_TITLE_SOURCE_BASE_SCORE: dict[str, float] = {
    "assistant_heading": 12.0,
    "subject_clause": 11.0,
    "subject_clause_snippet": 10.0,
    "assistant_proper_phrase": 9.0,
    "assistant_topic": 7.0,
    "sidecar_proper_phrase": 6.0,
    "sidecar_line": 5.0,
    "user_proper_phrase": 4.0,
    "post_think_tail": 4.0,
    "think_interior": 2.0,
    "sidecar_coerced": 1.0,
    "user_topic": 1.0,
}


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
    title_words = _TITLE_WORD_RE.findall(title_norm)
    prompt_words = _TITLE_WORD_RE.findall(prompt_norm)
    # Reject long openings copied from the user message, not short embedded topic names.
    if len(title_words) >= 5 and title_words == prompt_words[: len(title_words)]:
        return True
    if len(title_norm) >= max(24, int(len(prompt_norm) * 0.65)):
        if title_norm in prompt_norm or prompt_norm.startswith(title_norm):
            return True
    return False


def _contains_title_fragment_verb(text: str) -> bool:
    words = {w.lower() for w in _TITLE_WORD_RE.findall(text or "")}
    return bool(words & _TITLE_FRAGMENT_VERBS)


def _format_title_display(line: str) -> str:
    words = line.split()
    if not words:
        return ""
    if any(len(w) > 1 and w[1:].islower() and w[0].isupper() for w in words[1:]):
        return line
    formatted: list[str] = []
    for i, word in enumerate(words):
        if i > 0 and word.lower() in _TITLE_SMALL_WORDS:
            formatted.append(word.lower())
        else:
            formatted.append(word.capitalize())
    return " ".join(formatted)


def _title_from_proper_phrases(text: str) -> str:
    candidates: list[tuple[str, int]] = []
    for match in _OF_THE_PHRASE_RE.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" \"'.,!?;:")
        lead = phrase.split()[0].lower() if phrase else ""
        if lead in _PHRASE_LEAD_SKIP:
            continue
        polished = _polish_title_candidate(phrase)
        if polished:
            candidates.append((polished, len(polished.split())))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def _topic_words_from_text(text: str, *, max_words: int = 5) -> list[str]:
    text = (text or "").strip()
    for match in _OF_THE_PHRASE_RE.finditer(text):
        phrase_words = _TITLE_WORD_RE.findall(match.group(1))
        if 2 <= len(phrase_words) <= max_words:
            return phrase_words[:max_words]

    words = _TITLE_WORD_RE.findall(text)
    picked: list[str] = []
    i = 0
    while i < len(words) and len(picked) < max_words:
        word = words[i]
        low = word.lower()
        if low in _TITLE_STOPWORDS:
            if (
                low in _TITLE_SMALL_WORDS
                and picked
                and i + 1 < len(words)
                and words[i + 1].lower() not in _TITLE_STOPWORDS
            ):
                picked.append(word)
            i += 1
            continue
        if low in _TITLE_FRAGMENT_VERBS:
            i += 1
            continue
        picked.append(word)
        i += 1
    if len(picked) < 2:
        picked = [w for w in words if w.lower() not in _TITLE_STOPWORDS][:max_words]
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
    if _contains_title_fragment_verb(line):
        return ""
    return _format_title_display(line)


def _is_task_wrapper_prompt(user_prompt: str) -> bool:
    words = {w.lower() for w in _TITLE_WORD_RE.findall(user_prompt or "")}
    if not words & _TITLE_TASK_VERBS:
        return False
    if words & _TITLE_FORMAT_CUES:
        return True
    return bool(_SUBJECT_CLAUSE_RE.search(user_prompt or ""))


def _subject_clause_from_user_prompt(user_prompt: str) -> str:
    text = (user_prompt or "").strip()
    match = _SUBJECT_CLAUSE_RE.search(text)
    if not match:
        return ""
    clause = match.group(1).strip().strip(".,!?;:\"'")
    if len(clause) > 96:
        clause = clause[:96].rsplit(" ", 1)[0] or clause[:96]
    return clause


def _title_from_markdown_heading(text: str) -> str:
    for match in _MARKDOWN_HEADING_RE.finditer(text or ""):
        heading = re.sub(r"\s+", " ", match.group(1)).strip(" \"'.,!?;:")
        if heading:
            return heading
    return ""


def _title_is_instruction_like(title: str) -> bool:
    words = [w.lower() for w in _TITLE_WORD_RE.findall(title or "")]
    instruction_count = sum(1 for word in words if word in _TITLE_INSTRUCTION_WORDS)
    if instruction_count >= 2:
        return True
    if re.search(r"\b\d{3,}\b", title or ""):
        return True
    if re.search(r"\b\d+\s*(?:words?|pages?|paragraphs?)\b", title or "", re.IGNORECASE):
        return True
    if instruction_count >= 1 and re.search(r"\d", title or ""):
        return True
    return False


def _prepare_title_candidate(candidate: str, *, user_prompt: str) -> str:
    polished = _polish_title_candidate(candidate)
    if not polished:
        return ""
    if _title_is_verbatim_of_prompt(polished, user_prompt):
        return ""
    if _title_is_instruction_like(polished):
        return ""
    return polished


def _score_title_candidate(
    title: str,
    *,
    source: str,
    user_prompt: str,
    assistant_reply: str,
    task_wrapper: bool,
) -> float:
    score = _TITLE_SOURCE_BASE_SCORE.get(source, 0.0)
    words = [w.lower() for w in _TITLE_WORD_RE.findall(title)]
    word_count = len(words)

    if task_wrapper:
        if source.startswith("assistant") or source == "assistant_heading":
            score += 8.0
        elif source.startswith("subject"):
            score += 10.0
        elif source == "user_topic":
            score -= 12.0
        elif source.startswith("sidecar"):
            score -= 6.0

    if 2 <= word_count <= 5:
        score += 4.0
    elif word_count > 6:
        score -= 3.0

    for word in words:
        if word in _TITLE_INSTRUCTION_WORDS:
            score -= 10.0

    if re.search(r"\d", title):
        if re.search(r"\b\d{3,}\b", title):
            score -= 15.0
        else:
            score -= 4.0

    assistant_words = {
        w.lower()
        for w in _topic_words_from_text(assistant_reply, max_words=8)
    }
    title_words = set(words)
    overlap = len(title_words & assistant_words)
    if source not in {
        "assistant_topic",
        "assistant_proper_phrase",
        "assistant_heading",
    }:
        if overlap >= 2:
            score += 6.0
        elif overlap == 1:
            score += 2.0

    if task_wrapper and source.startswith("sidecar") and overlap == 0:
        score -= 8.0

    embedded = _title_from_proper_phrases(user_prompt)
    if source.startswith("sidecar") and embedded:
        if _normalize_title_compare(title) == _normalize_title_compare(embedded):
            score += 5.0

    return score


def _collect_title_candidates(
    raw_s: str,
    *,
    user_prompt: str,
    assistant_reply: str,
) -> list[tuple[str, str]]:
    by_key: dict[str, tuple[str, str]] = {}

    def add(raw: str, source: str) -> None:
        title = _prepare_title_candidate(raw, user_prompt=user_prompt)
        if not title:
            return
        key = _normalize_title_compare(title)
        existing = by_key.get(key)
        if existing is not None and _TITLE_SOURCE_BASE_SCORE.get(source, 0) <= _TITLE_SOURCE_BASE_SCORE.get(
            existing[1], 0
        ):
            return
        by_key[key] = (title, source)

    subject_clause = _subject_clause_from_user_prompt(user_prompt)
    if subject_clause:
        add(subject_clause, "subject_clause")
        clause_words = _topic_words_from_text(subject_clause)
        if len(clause_words) >= 2:
            add(" ".join(clause_words), "subject_clause_snippet")

    heading = _title_from_markdown_heading(assistant_reply)
    if heading:
        add(heading, "assistant_heading")

    for source_text, source in (
        (assistant_reply, "assistant_proper_phrase"),
        (user_prompt, "user_proper_phrase"),
    ):
        labeled = _title_from_proper_phrases(source_text)
        if labeled:
            add(labeled, source)

    for source_text, source in (
        (assistant_reply, "assistant_topic"),
        (user_prompt, "user_topic"),
    ):
        topic_words = _topic_words_from_text(source_text)
        if len(topic_words) >= 2:
            snippet = " ".join(topic_words)
            if len(snippet) > 48:
                snippet = snippet[:48].rsplit(" ", 1)[0] or snippet[:48]
            add(snippet, source)

    if raw_s:
        cleaned = _normalize_sidecar_completion_text(raw_s)
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        for line in reversed(lines):
            add(line, "sidecar_line")
            labeled = _title_from_proper_phrases(line)
            if labeled:
                add(labeled, "sidecar_proper_phrase")
            coerced_words = _topic_words_from_text(line)
            if len(coerced_words) >= 2:
                add(" ".join(coerced_words), "sidecar_coerced")

        post_think = _title_from_post_think_tail(raw_s)
        if post_think:
            add(post_think, "post_think_tail")

        think_match = _THINK_INTERIOR_RE.search(raw_s)
        if think_match:
            interior_lines = [
                ln.strip()
                for ln in think_match.group(1).splitlines()
                if ln.strip()
            ]
            for line in sorted(interior_lines, key=lambda ln: len(ln.split())):
                add(line, "think_interior")

    return list(by_key.values())


def _select_best_title(
    candidates: list[tuple[str, str]],
    *,
    user_prompt: str,
    assistant_reply: str,
) -> str:
    if not candidates:
        return ""
    task_wrapper = _is_task_wrapper_prompt(user_prompt)
    best_title = ""
    best_score = float("-inf")
    for title, source in candidates:
        score = _score_title_candidate(
            title,
            source=source,
            user_prompt=user_prompt,
            assistant_reply=assistant_reply,
            task_wrapper=task_wrapper,
        )
        if score > best_score:
            best_score = score
            best_title = title
    if best_score < _TITLE_MIN_ACCEPT_SCORE:
        return ""
    return best_title


def _fallback_title_from_exchange(
    user_prompt: str,
    *,
    assistant_reply: str = "",
) -> str:
    candidates = _collect_title_candidates(
        "",
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
    )
    return _select_best_title(
        candidates,
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
    )


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


def _accept_title_candidate(
    candidate: str,
    *,
    user_prompt: str,
) -> str:
    return _prepare_title_candidate(candidate, user_prompt=user_prompt)


def _finalize_title_text(
    raw: str,
    *,
    user_prompt: str = "",
    assistant_reply: str = "",
) -> str:
    """Single-line title after stripping Qwen3 / R1-style reasoning blocks."""
    raw_s = (raw or "").strip()
    candidates = _collect_title_candidates(
        raw_s,
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
    )
    return _select_best_title(
        candidates,
        user_prompt=user_prompt,
        assistant_reply=assistant_reply,
    )


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
    recommended_target = str(obj.get("recommended_target") or "").strip().lower()
    if recommended_target not in ("chat", "memory", "rag", "web", "none", ""):
        recommended_target = ""
    return {
        "expanded_query": expanded,
        "confidence": conf,
        "topic_source": topic_source,
        "recommended_target": recommended_target,
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
