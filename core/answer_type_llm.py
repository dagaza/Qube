"""
Shadow answer type for questions sent to the Evidence service (plan S30, D67).

The relevance gate cannot see "on topic but the wrong kind of source" (V1): an
OAuth how-to sent to scholarly indexes kept OAuth-plus-blockchain papers, and the
shipped gate calls that pool sufficient. S30 measured which kind of source answers
a question, on the knowledge-quality fixtures (dev) and on a held-out set frozen
before any run (``eval/retrieval_corpus/source_kind_v1.json``):

- the sidecar, asked for one word with a few examples (``ANSWER_TYPE_SYSTEM``),
  flagged 11 of 12 held-out how-tos and 16 of 18 non-scholarly questions, and 1 of
  12 research questions wrongly (dev: 8 of 8, 27 of 32, 2 of 31);
- ``rule_answer_type``, built from the dev set's how-to wordings only, flagged 9 of
  12 held-out how-tos and no research question (dev: 4 of 8, 0 of 31).

Both miss a protocol question shaped like OAuth's ("How does Kerberos issue a
service ticket?"), so neither steers yet (§12.5 rule 4). On a web turn whose
service is ``scientific_evidence`` one ``[AnswerType] shadow {json}`` INFO line is
logged: the sidecar's answer on a background thread when it is free, the rule's
always. Nothing changes the service, the sources or the answer, and the line holds
no question text, so field data is read as counts.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from core.cognition_prompt_adapter import build_cognition_prompt, prefill_qwen3_empty_think
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
from core.sidecar_types import SidecarTask

logger = logging.getLogger("Qube.AnswerType")

ANSWER_TYPE_SCHOLARLY = "scholarly"
ANSWER_TYPE_PRACTICAL = "practical"
ANSWER_TYPE_REFERENCE = "reference"

# Prompt v4 of the S30 measurement, chosen on the dev set. v1 (the three kinds with
# RESEARCH first) answered RESEARCH for almost everything; v2 (RESEARCH last, "only
# questions about evidence") flagged 10 of 31 research questions; v3 (PAPERS or
# OTHER) flagged no how-to. The examples are in neither set.
ANSWER_TYPE_SYSTEM = (
    "You decide what kind of source best answers a question. Reply with one word:\n"
    "GUIDE - documentation, a specification, a manual or a step-by-step guide answers it.\n"
    "REFERENCE - an encyclopedia, an official information page or the news answers it.\n"
    "RESEARCH - research papers, clinical trials or scientific data answer it.\n\n"
    "Examples:\n"
    "Question: How do I change a bicycle tyre? -> GUIDE\n"
    "Question: How does the HTTP/2 protocol multiplex streams? -> GUIDE\n"
    "Question: What is the capital of Peru? -> REFERENCE\n"
    "Question: Who won the 2018 World Cup? -> REFERENCE\n"
    "Question: Does vitamin D supplementation prevent fractures? -> RESEARCH\n"
    "Question: How does chronic stress affect the immune system? -> RESEARCH"
)
ANSWER_TYPE_QUESTION_CHARS = 400
ANSWER_TYPE_TIMEOUT_SEC = 4.0

_WORD_TO_TYPE = {
    "RESEARCH": ANSWER_TYPE_SCHOLARLY,
    "GUIDE": ANSWER_TYPE_PRACTICAL,
    "REFERENCE": ANSWER_TYPE_REFERENCE,
}
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.I)
_FIRST_WORD = re.compile(r"[A-Za-z]+")

# The dev set's how-to wordings ("how do I undo…", "how do i keep it alive", "how do
# you maintain it", "how to descale…"); nothing was added from the held-out set.
_HOW_TO = re.compile(r"\bhow\s+do\s+(?:i|you)\b|\bhow\s+to\b", re.I)

# Composer tools that choose the Evidence service; any other Evidence turn has it
# as the default knowledge service.
EXPLICIT_EVIDENCE_TOOLS = frozenset({"evidence", "science", "pubmed", "arxiv"})
MODE_EXPLICIT = "explicit"
MODE_DEFAULT = "default"

GATE_SCHEDULED = "scheduled"
GATE_DISABLED = "disabled"
GATE_NOT_EVIDENCE = "not_evidence"
GATE_NO_QUESTION = "no_question"
GATE_SIDECAR_UNAVAILABLE = "sidecar_unavailable"
GATE_SIDECAR_BUSY = "sidecar_busy"


@dataclass
class AnswerTypeOutcome:
    """One sidecar attempt (JSON-safe via ``to_dict``)."""

    status: str  # ok | parse_fail | timeout | sidecar_error
    answer_type: Optional[str] = None
    reason: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["latency_ms"] = round(float(self.latency_ms), 1)
        return out


def rule_answer_type(question: str) -> Optional[str]:
    """``practical`` for a how-to wording, else ``None`` (the rule says nothing else)."""
    return ANSWER_TYPE_PRACTICAL if _HOW_TO.search(question or "") else None


def answer_type_mode(composer_tool: str | None) -> str:
    tool = (composer_tool or "").strip().lower()
    return MODE_EXPLICIT if tool in EXPLICIT_EVIDENCE_TOOLS else MODE_DEFAULT


def build_answer_type_prompt(*, question: str, chat_format: str = "chatml", model_path: str = "") -> str:
    """Rendered sidecar prompt, ending in the assistant turn (Qwen3: empty think block pre-filled, as S9)."""
    user = f"Question: {(question or '').strip()[:ANSWER_TYPE_QUESTION_CHARS]}"
    rendered = build_cognition_prompt(ANSWER_TYPE_SYSTEM, user, chat_format, model_path=model_path)
    return prefill_qwen3_empty_think(rendered, chat_format, model_path)


def parse_answer_type_output(raw: str) -> Optional[str]:
    """The kind named by the completion's first word, or ``None``."""
    text = _THINK_BLOCK.sub(" ", raw or "")
    match = _FIRST_WORD.search(text)
    return _WORD_TO_TYPE.get(match.group(0).upper()) if match else None


def answer_type_gate(
    service: str,
    question: str,
    sidecar_client: Any,
    *,
    enabled: bool,
    queue_depth: int = 0,
) -> str:
    """Why the sidecar is (``scheduled``) or is not asked on this turn."""
    if not enabled:
        return GATE_DISABLED
    if (service or "").strip().lower() != SERVICE_SCIENTIFIC_EVIDENCE:
        return GATE_NOT_EVIDENCE
    if not (question or "").strip():
        return GATE_NO_QUESTION
    if sidecar_client is None or not hasattr(sidecar_client, "complete"):
        return GATE_SIDECAR_UNAVAILABLE
    if hasattr(sidecar_client, "available") and not sidecar_client.available:
        return GATE_SIDECAR_UNAVAILABLE
    if int(queue_depth or 0) > 0:
        return GATE_SIDECAR_BUSY
    return GATE_SCHEDULED


def classify_answer_type(question: str, sidecar_client: Any, *, timeout_sec: float) -> AnswerTypeOutcome:
    """Blocking sidecar call (call it off the turn thread). Never raises."""
    t0 = time.perf_counter()
    try:
        result = sidecar_client.complete(SidecarTask.answer_type, timeout_sec=timeout_sec, question=question)
    except Exception as exc:  # noqa: BLE001 - shadow path must never raise
        return AnswerTypeOutcome("sidecar_error", reason=str(exc)[:120], latency_ms=(time.perf_counter() - t0) * 1000)
    latency = (time.perf_counter() - t0) * 1000
    if not getattr(result, "ok", False):
        err = str(getattr(result, "error", "") or "fail")
        status = "timeout" if err == "timeout" else ("parse_fail" if err in ("parse_fail", "empty_output") else "sidecar_error")
        return AnswerTypeOutcome(status, reason=err, latency_ms=latency)
    kind = str((getattr(result, "parsed", None) or {}).get("answer_type") or "") or None
    if kind not in _WORD_TO_TYPE.values():
        return AnswerTypeOutcome("parse_fail", reason="no_answer_type", latency_ms=latency)
    return AnswerTypeOutcome("ok", kind, reason="ok", latency_ms=latency)


def run_shadow_answer_type(
    *,
    question: str,
    service: str,
    composer_tool: str | None,
    sidecar_client: Any,
    enabled: bool,
    timeout_sec: float = ANSWER_TYPE_TIMEOUT_SEC,
    queue_depth: int = 0,
    session_id: str = "",
    on_done: Callable[[dict[str, Any]], None] | None = None,
    background: bool = True,
) -> str:
    """Gate, log one line per Evidence turn, and return the gate reason at once.

    When the sidecar is scheduled the line is logged from a daemon thread after its
    answer; otherwise it is logged now with the rule's answer only. Never changes
    retrieval, and never blocks the caller when ``background`` is true.
    """
    reason = answer_type_gate(service, question, sidecar_client, enabled=enabled, queue_depth=queue_depth)
    if reason in (GATE_DISABLED, GATE_NOT_EVIDENCE, GATE_NO_QUESTION):
        return reason
    base = {
        "session_id": session_id,
        "service": SERVICE_SCIENTIFIC_EVIDENCE,
        "mode": answer_type_mode(composer_tool),
        "gate": reason,
        "rule": rule_answer_type(question),
        "question_chars": len(question or ""),
    }

    def _emit(record: dict[str, Any]) -> None:
        logger.info("[AnswerType] shadow %s", json.dumps(record, ensure_ascii=False))
        if on_done is not None:
            try:
                on_done(record)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[AnswerType] on_done failed: %s", exc)

    if reason != GATE_SCHEDULED:
        _emit({**base, "llm": None})
        return reason

    text = question

    def _work() -> None:
        outcome = classify_answer_type(text, sidecar_client, timeout_sec=timeout_sec)
        _emit({**base, "llm": outcome.to_dict(), "timeout_ms": int(timeout_sec * 1000)})

    if background:
        threading.Thread(target=_work, name="AnswerTypeShadow", daemon=True).start()
    else:
        _work()
    return reason


__all__ = [
    "ANSWER_TYPE_PRACTICAL",
    "ANSWER_TYPE_REFERENCE",
    "ANSWER_TYPE_SCHOLARLY",
    "ANSWER_TYPE_SYSTEM",
    "AnswerTypeOutcome",
    "EXPLICIT_EVIDENCE_TOOLS",
    "GATE_DISABLED",
    "GATE_NOT_EVIDENCE",
    "GATE_NO_QUESTION",
    "GATE_SCHEDULED",
    "GATE_SIDECAR_BUSY",
    "GATE_SIDECAR_UNAVAILABLE",
    "MODE_DEFAULT",
    "MODE_EXPLICIT",
    "answer_type_gate",
    "answer_type_mode",
    "build_answer_type_prompt",
    "classify_answer_type",
    "parse_answer_type_output",
    "rule_answer_type",
    "run_shadow_answer_type",
]
