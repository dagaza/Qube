"""
Gated LLM slot extraction for follow-up retrieval queries (plan S9, shadow mode).

When the deterministic resolver (S5/S6, ``core.discourse_query``) leaves a
follow-up turn unresolved, the sidecar is asked for two slots only:

    SUBJECT: <what the conversation is about>
    FACET: <what the last message asks about it>

No JSON grammar (V6: grammar sampling costs 2-3x on CPU). Code validates the
subject against the conversation (an LLM subject that appears nowhere in the
history is rejected) and composes the search text with the same composer S6
uses, so the LLM never writes the query itself.

Shadow mode: the extraction runs on a background thread after the turn has
been routed, never delays the turn and never changes retrieval. Each outcome is
logged as one ``[QuerySlots] shadow {json}`` INFO line (persisted in
``qube.log``), next to the deterministic text, so field data can decide
whether the slots ever earn a steering role.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from core.cognition_prompt_adapter import (  # QWEN3_EMPTY_THINK re-exported for S9 tests
    QWEN3_EMPTY_THINK,
    build_cognition_prompt,
    prefill_qwen3_empty_think,
)
from core.discourse_subject import (
    RELATION_CONTINUATION,
    SubjectTrack,
    clean_turn,
    compose_retrieval_text,
    has_back_reference,
)
from core.sidecar_types import SidecarTask

logger = logging.getLogger("Qube.QuerySlots")

SLOT_SUBJECT_PREFIX = "SUBJECT:"

SLOT_HISTORY_MESSAGES = 4
SLOT_MESSAGE_CHARS = 400
SLOT_MAX_SUBJECT_WORDS = 8
SLOT_MAX_CHARS = 80

SLOT_SYSTEM = (
    "You read a conversation and say what the user's last message is about. "
    "Reply with exactly two lines and nothing else:\n"
    "SUBJECT: <the thing being discussed, as a short noun phrase taken from the conversation>\n"
    "FACET: <the aspect of it the last message asks about, or none>"
)

GATE_SCHEDULED = "scheduled"
GATE_DISABLED = "disabled"
GATE_NOT_FOLLOW_UP = "not_follow_up"
GATE_DETERMINISTIC_RESOLVED = "deterministic_resolved"
GATE_NO_HISTORY = "no_history"
GATE_SIDECAR_UNAVAILABLE = "sidecar_unavailable"
GATE_SIDECAR_BUSY = "sidecar_busy"

_EMPTY_SLOT = frozenset({"", "none", "n/a", "na", "unknown", "nothing", "-", "null"})
_SLOT_JUNK = re.compile(r"^[\s\"'`*<>\[\](){}.,;:!?-]+|[\s\"'`*<>\[\](){}.,;:!?-]+$")
# Also an unterminated "<think>…" (the truncated output the S9 audit found).
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.I)
_CONTENT_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'+#.-]{2,}")
_ANCHOR_STOPWORDS = frozenset(
    (
        "the a an of in on at to for and or but with by from about what which who how why when "
        "where is are was were be been do does did it its this that these those they them there "
        "one ones some any more most other such new"
    ).split()
)


@dataclass(frozen=True)
class QuerySlots:
    subject: str
    facet: Optional[str] = None


@dataclass
class SlotOutcome:
    """Result of one extraction attempt (JSON-safe via ``to_dict``)."""

    status: str  # ok | rejected | parse_fail | timeout | sidecar_error
    subject: Optional[str] = None
    facet: Optional[str] = None
    composed_query: Optional[str] = None
    reason: str = ""
    latency_ms: float = 0.0
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["latency_ms"] = round(float(self.latency_ms), 1)
        out["raw"] = (self.raw or "")[:200]
        return out


def _role(message: Any) -> str:
    return str((message or {}).get("role") or "user") if isinstance(message, dict) else "user"


def _content(message: Any) -> str:
    return str((message or {}).get("content") or "").strip() if isinstance(message, dict) else ""


def _prior_messages(history: list[dict[str, Any]] | None, prompt: str) -> list[dict[str, Any]]:
    """History without the current prompt (the worker persists it before routing)."""
    turns = [m for m in (history or []) if isinstance(m, dict) and _content(m)]
    current = " ".join((prompt or "").split())
    if turns and _role(turns[-1]) == "user" and " ".join(_content(turns[-1]).split()) == current:
        turns = turns[:-1]
    return turns


def build_slot_prompt(
    *,
    history: list[dict[str, Any]] | None,
    prompt: str,
    chat_format: str = "chatml",
    model_path: str = "",
) -> str:
    """Rendered sidecar prompt, ending in the assistant turn pre-filled with ``SUBJECT:``."""
    lines = [
        f"{_role(m)}: {_content(m)[:SLOT_MESSAGE_CHARS]}"
        for m in _prior_messages(history, prompt)[-SLOT_HISTORY_MESSAGES:]
    ]
    user = (
        "Conversation:\n"
        + "\n".join(lines)
        + f"\nuser (last message): {clean_turn(prompt)[:SLOT_MESSAGE_CHARS]}"
    )
    rendered = build_cognition_prompt(SLOT_SYSTEM, user, chat_format, model_path=model_path)
    return prefill_qwen3_empty_think(rendered, chat_format, model_path) + SLOT_SUBJECT_PREFIX


def _clean_slot(value: str) -> Optional[str]:
    v = _SLOT_JUNK.sub("", (value or "").strip())
    v = " ".join(v.split())
    if v.lower() in _EMPTY_SLOT:
        return None
    return v[:SLOT_MAX_CHARS] or None


def parse_slot_output(raw: str) -> Optional[QuerySlots]:
    """Parse the completion that follows the pre-filled ``SUBJECT:``."""
    text = _THINK_BLOCK.sub("", raw or "").strip()
    if not text:
        return None
    if not text.upper().startswith(SLOT_SUBJECT_PREFIX):
        text = f"{SLOT_SUBJECT_PREFIX} {text}"
    subject = facet = None
    for line in text.splitlines():
        s = line.strip()
        head, _, value = s.partition(":")
        key = head.strip().upper()
        if key == "SUBJECT" and subject is None:
            subject = _clean_slot(value)
        elif key == "FACET" and facet is None:
            facet = _clean_slot(value)
    if not subject or len(subject.split()) > SLOT_MAX_SUBJECT_WORDS:
        return None
    return QuerySlots(subject=subject, facet=facet)


def _word_forms(word: str) -> set[str]:
    w = word.lower().strip(".'")
    forms = {w}
    if len(w) > 3 and w.endswith("s"):
        forms.add(w[:-1])
    else:
        forms.add(w + "s")
    return forms


def validate_slots(
    slots: QuerySlots,
    history: list[dict[str, Any]] | None,
    prompt: str,
) -> tuple[bool, str]:
    """Accept a subject only when it is grounded in the conversation."""
    from core.sidecar_query_rewrite import expansion_adds_unanchored_proper_nouns

    subject = slots.subject
    if has_back_reference(subject):
        return False, "subject_is_reference"
    prior = [_content(m) for m in _prior_messages(history, prompt)]
    if expansion_adds_unanchored_proper_nouns(subject, prompt, *prior):
        return False, "unanchored_proper_noun"
    blob_words: set[str] = set()
    for text in (*prior, prompt):
        for word in _CONTENT_WORD.findall(text or ""):
            blob_words |= _word_forms(word)
    content = [w for w in _CONTENT_WORD.findall(subject) if w.lower() not in _ANCHOR_STOPWORDS]
    if not content:
        return False, "no_content_word"
    if not any(_word_forms(w) & blob_words for w in content):
        return False, "unanchored_subject"
    return True, "ok"


def compose_slot_query(prompt: str, slots: QuerySlots) -> str:
    """Search text from the slots, built by the S6 composer (subject + cleaned turn)."""
    track = SubjectTrack(
        subject=slots.subject,
        facet=slots.facet,
        relation=RELATION_CONTINUATION,
        reasons=("llm_slots",),
    )
    return compose_retrieval_text(prompt, track) or slots.subject


def slot_gate(
    follow_up: Any,
    resolved: Any,
    history: list[dict[str, Any]] | None,
    prompt: str,
    sidecar_client: Any,
    *,
    enabled: bool,
    queue_depth: int = 0,
) -> str:
    """Why the slot extractor does (``scheduled``) or does not run on this turn."""
    if not enabled:
        return GATE_DISABLED
    # Any follow-up classification counts, including one below the activation
    # threshold: most residual turns S6 misses classify as ``ellipsis`` 0.50
    # (S9 measurement: 6 of 8 residuals gated this way, 2 of 8 with ``active``).
    kind = getattr(getattr(follow_up, "kind", None), "value", "none") if follow_up is not None else "none"
    if follow_up is None or not (getattr(follow_up, "active", False) or kind != "none"):
        return GATE_NOT_FOLLOW_UP
    if resolved is not None and (
        getattr(resolved, "web_subject_relation", "") == RELATION_CONTINUATION
        or getattr(resolved, "web_rewritten", False)
    ):
        return GATE_DETERMINISTIC_RESOLVED
    if not any(_role(m) == "user" for m in _prior_messages(history, prompt)):
        return GATE_NO_HISTORY
    if sidecar_client is None or not hasattr(sidecar_client, "complete"):
        return GATE_SIDECAR_UNAVAILABLE
    if hasattr(sidecar_client, "available") and not sidecar_client.available:
        return GATE_SIDECAR_UNAVAILABLE
    if int(queue_depth or 0) > 0:
        return GATE_SIDECAR_BUSY
    return GATE_SCHEDULED


def extract_query_slots(
    prompt: str,
    history: list[dict[str, Any]] | None,
    sidecar_client: Any,
    *,
    timeout_sec: float,
) -> SlotOutcome:
    """Blocking extraction (call it off the turn thread). Never raises."""
    t0 = time.perf_counter()
    try:
        result = sidecar_client.complete(
            SidecarTask.query_slots,
            timeout_sec=timeout_sec,
            history=list(history or []),
            prompt=prompt,
        )
    except Exception as exc:  # noqa: BLE001 - shadow path must never raise
        return SlotOutcome("sidecar_error", reason=str(exc)[:120],
                           latency_ms=(time.perf_counter() - t0) * 1000)
    latency = (time.perf_counter() - t0) * 1000
    if not getattr(result, "ok", False):
        err = str(getattr(result, "error", "") or "fail")
        status = "timeout" if err == "timeout" else ("parse_fail" if err in ("parse_fail", "empty_output") else "sidecar_error")
        return SlotOutcome(status, reason=err, latency_ms=latency, raw=str(getattr(result, "text", "") or ""))
    parsed = getattr(result, "parsed", None) or {}
    slots = QuerySlots(subject=str(parsed.get("subject") or ""), facet=parsed.get("facet") or None)
    raw = str(getattr(result, "text", "") or "")
    if not slots.subject:
        return SlotOutcome("parse_fail", reason="no_subject", latency_ms=latency, raw=raw)
    ok, why = validate_slots(slots, history, prompt)
    if not ok:
        return SlotOutcome("rejected", slots.subject, slots.facet, reason=why, latency_ms=latency, raw=raw)
    return SlotOutcome(
        "ok",
        slots.subject,
        slots.facet,
        composed_query=compose_slot_query(prompt, slots),
        reason="ok",
        latency_ms=latency,
        raw=raw,
    )


def run_shadow_slot_extraction(
    *,
    prompt: str,
    history: list[dict[str, Any]] | None,
    follow_up: Any,
    resolved: Any,
    sidecar_client: Any,
    enabled: bool,
    timeout_sec: float,
    queue_depth: int = 0,
    session_id: str = "",
    on_done: Callable[[dict[str, Any]], None] | None = None,
    background: bool = True,
) -> str:
    """Gate, then (when scheduled) extract on a daemon thread and log the comparison.

    Returns the gate reason immediately; never blocks the caller when
    ``background`` is true, and never changes retrieval.
    """
    reason = slot_gate(
        follow_up, resolved, history, prompt, sidecar_client,
        enabled=enabled, queue_depth=queue_depth,
    )
    if reason != GATE_SCHEDULED:
        return reason
    hist = [dict(m) for m in (history or []) if isinstance(m, dict)]
    deterministic = {
        "web_text": getattr(resolved, "web_text", None),
        "web_subject": getattr(resolved, "web_subject", None),
        "relation": getattr(resolved, "web_subject_relation", None),
        "web_rewrite_reason": getattr(resolved, "web_rewrite_reason", None),
    }
    kind = getattr(getattr(follow_up, "kind", None), "value", None)

    def _work() -> None:
        outcome = extract_query_slots(prompt, hist, sidecar_client, timeout_sec=timeout_sec)
        record = {
            "session_id": session_id,
            "prompt": (prompt or "")[:200],
            "follow_up_kind": kind,
            "follow_up_confidence": round(float(getattr(follow_up, "confidence", 0.0) or 0.0), 3),
            "deterministic": deterministic,
            "slots": outcome.to_dict(),
            "timeout_ms": int(timeout_sec * 1000),
        }
        logger.info("[QuerySlots] shadow %s", json.dumps(record, ensure_ascii=False))
        if on_done is not None:
            try:
                on_done(record)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[QuerySlots] on_done failed: %s", exc)

    if background:
        threading.Thread(target=_work, name="QuerySlotsShadow", daemon=True).start()
    else:
        _work()
    return reason


__all__ = [
    "GATE_DETERMINISTIC_RESOLVED",
    "GATE_DISABLED",
    "GATE_NOT_FOLLOW_UP",
    "GATE_NO_HISTORY",
    "GATE_SCHEDULED",
    "GATE_SIDECAR_BUSY",
    "GATE_SIDECAR_UNAVAILABLE",
    "QuerySlots",
    "SlotOutcome",
    "build_slot_prompt",
    "compose_slot_query",
    "extract_query_slots",
    "parse_slot_output",
    "run_shadow_slot_extraction",
    "slot_gate",
    "validate_slots",
]
