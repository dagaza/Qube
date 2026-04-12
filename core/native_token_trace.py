"""
Token-level execution trace for the native llama.cpp path (observer only).

Enable with QUBE_LLM_TOKEN_TRACE=1.

- **Ground truth (sampler):** `llm_token_trace_ground_truth` events use token IDs captured from
  `Llama.generate` (same path as `Llama._create_completion`), with per-token `detokenize` — not
  from chat deltas.

- **Live stream path:** each chat stream delta is re-tokenized for an approximate live trace
  (`llm_token_trace_live` / `llm_token_trace`).

- **Post-hoc (fallback):** full-string tokenize of assistant output for comparison / non-stream calls.

QUBE_LLM_TOKEN_TRACE_N=20 (default), QUBE_LLM_TOKEN_TRACE_EARLY=5 (first early event threshold).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from core.native_sampler_gt import (
    detokenize_sampler_token_ids,
    sequence_prefix_match_score,
)

logger = logging.getLogger("Qube.NativeLLM.Debug")


def token_trace_enabled() -> bool:
    return os.environ.get("QUBE_LLM_TOKEN_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def token_trace_n() -> int:
    try:
        n = int(os.environ.get("QUBE_LLM_TOKEN_TRACE_N", "20"))
        return max(1, min(256, n))
    except ValueError:
        return 20


def token_trace_early_n() -> int:
    try:
        n = int(os.environ.get("QUBE_LLM_TOKEN_TRACE_EARLY", "5"))
        return max(1, min(64, n))
    except ValueError:
        return 5


def extract_first_n_generation_tokens(
    llama: Any, assistant_text: str, n: int
) -> tuple[list[int], list[str], str]:
    """Post-hoc: tokenize full assistant string (fallback / divergence check)."""
    if not assistant_text or llama is None:
        return [], [], ""

    try:
        raw = assistant_text.encode("utf-8")
        ids = llama.tokenize(raw, add_bos=False, special=False)
    except Exception as e:
        logger.debug("[token_trace] posthoc tokenize failed: %s", e)
        return [], [], ""

    ids = ids[:n]
    pieces: list[str] = []
    cumulative = ""
    for i, tid in enumerate(ids):
        try:
            prev = ids[:i]
            b = llama.detokenize(
                [tid],
                prev_tokens=prev if i else None,
                special=False,
            )
            piece = b.decode("utf-8", errors="replace")
        except Exception:
            piece = ""
        pieces.append(piece)
        cumulative += piece
    return ids, pieces, cumulative


def classify_first_token_piece(first_piece: str) -> str:
    t = (first_piece or "").lstrip()
    if not t:
        return "unknown"
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return "role_continuation"
    if re.match(r"^we\b", low) or re.match(r"^let\'?s\b", low):
        return "reasoning_prefix"
    if low.startswith(("step ", "first,", "first ", "to answer", "to address")):
        return "reasoning_prefix"
    if low.startswith(
        (
            "provide",
            "answer",
            "the answer",
            "here's",
            "here is",
            "summary:",
            "response:",
        )
    ):
        return "instruction_echo"
    if t[0].isalnum() or t[0] in "-*•":
        return "normal_start"
    return "unknown"


def risk_from_classification(cls: str) -> str:
    if cls in ("role_continuation", "instruction_echo"):
        return "HIGH"
    if cls in ("reasoning_prefix", "unknown"):
        return "MEDIUM"
    return "LOW"


def _payload_preflight(pre: dict[str, Any], chat_format: str) -> dict[str, Any]:
    return {
        "prompt_tail": pre.get("prompt_tail", ""),
        "assistant_anchor_present": pre.get("assistant_anchor_present"),
        "chat_format": chat_format,
        "preflight_stop_sample": pre.get("merged_stops", [])[:25],
        "preflight_eos_token_str": pre.get("eos_token_str", ""),
        "preflight_sampling": pre.get("sampling_snapshot", {}),
    }


class LiveStreamTokenTrace:
    """
    Incremental capture from chat completion stream deltas (observer; does not touch model output).
    """

    __slots__ = (
        "llama",
        "preflight",
        "chat_format",
        "n_max",
        "n_early",
        "token_ids",
        "token_pieces",
        "text_buffer",
        "early_emitted",
        "last_finish_reason",
    )

    def __init__(
        self,
        llama: Any,
        trace_preflight: Optional[dict[str, Any]],
        chat_format: str,
    ) -> None:
        self.llama = llama
        self.preflight = trace_preflight or {}
        self.chat_format = chat_format
        self.n_max = token_trace_n()
        self.n_early = token_trace_early_n()
        self.token_ids: list[int] = []
        self.token_pieces: list[str] = []
        self.text_buffer = ""
        self.early_emitted = False
        self.last_finish_reason: Optional[str] = None

    def feed_delta(self, delta: str, finish_reason: Optional[str] = None) -> None:
        if finish_reason is not None:
            self.last_finish_reason = finish_reason
        if not delta or self.llama is None:
            return
        self.text_buffer += delta
        try:
            chunk_ids = self.llama.tokenize(
                delta.encode("utf-8"), add_bos=False, special=False
            )
        except Exception:
            chunk_ids = []
        for tid in chunk_ids:
            if len(self.token_ids) >= self.n_max:
                break
            self.token_ids.append(tid)
            i = len(self.token_ids) - 1
            try:
                prev = self.token_ids[:i]
                b = self.llama.detokenize(
                    [tid],
                    prev_tokens=prev if i else None,
                    special=False,
                )
                piece = b.decode("utf-8", errors="replace")
            except Exception:
                piece = ""
            self.token_pieces.append(piece)

        if (
            not self.early_emitted
            and len(self.token_ids) >= self.n_early
            and token_trace_enabled()
        ):
            self._emit_early()

    def _emit_early(self) -> None:
        self.early_emitted = True
        pieces = self.token_pieces[: self.n_early]
        cls = classify_first_token_piece(pieces[0] if pieces else "")
        risk = risk_from_classification(cls)
        payload: dict[str, Any] = {
            "event": "llm_token_trace_live",
            "phase": "early",
            "trace_n_target": self.n_early,
            "tokens": pieces,
            "token_ids": self.token_ids[: self.n_early],
            "token_classification": cls,
            "risk_assessment": risk,
            **_payload_preflight(self.preflight, self.chat_format),
        }
        logger.info(json.dumps(payload, ensure_ascii=False))

    def finalize(self, assistant_full_text: str) -> None:
        """Emit final live trace + post-hoc comparison. Safe to call once per request."""
        if not token_trace_enabled() or self.llama is None:
            return

        ph_ids, ph_pieces, ph_cum = extract_first_n_generation_tokens(
            self.llama, assistant_full_text, self.n_max
        )
        live_ids = self.token_ids[: self.n_max]
        diverges = live_ids != ph_ids[: len(live_ids)] if live_ids and ph_ids else False

        cls = classify_first_token_piece(
            self.token_pieces[0] if self.token_pieces else ""
        )
        risk = risk_from_classification(cls)

        payload: dict[str, Any] = {
            "event": "llm_token_trace",
            "phase": "complete",
            "source": "live_stream",
            "trace_n": self.n_max,
            "first_token_ids_live": live_ids,
            "first_tokens_live": self.token_pieces[: self.n_max],
            "first_tokens_decoded_live": "".join(self.token_pieces[: self.n_max]),
            "post_hoc_token_ids_fallback": ph_ids,
            "post_hoc_tokens_fallback": ph_pieces,
            "post_hoc_tokens_decoded_fallback": ph_cum,
            "live_vs_post_hoc_prefix_match": not diverges,
            "divergence_detected": diverges,
            "token_classification": cls,
            "risk_assessment": risk,
            "stream_finish_reason": self.last_finish_reason,
            **_payload_preflight(self.preflight, self.chat_format),
        }
        if diverges and live_ids and ph_ids:
            payload["divergence_note"] = (
                "live stream token-id prefix differs from post-hoc tokenization of full text; "
                "compare llm.cpp chunking vs canonical tokenize(full)."
            )
        logger.info(json.dumps(payload, ensure_ascii=False))


def emit_sampler_ground_truth_early(
    llama: Any,
    *,
    token_ids: list[int],
    prompt_tokens: list[int],
    trace_preflight: dict[str, Any],
    chat_format: str,
) -> None:
    """First N sampler token IDs + detokenized pieces (phase early)."""
    if not token_trace_enabled():
        return
    n = min(len(token_ids), token_trace_early_n())
    if n < 1:
        return
    ids = token_ids[:n]
    pieces = detokenize_sampler_token_ids(llama, prompt_tokens, ids)
    cls = classify_first_token_piece(pieces[0] if pieces else "")
    risk = risk_from_classification(cls)
    anchor = trace_preflight.get("assistant_anchor_present")
    payload: dict[str, Any] = {
        "event": "llm_token_trace_ground_truth",
        "phase": "early",
        "token_ids": ids,
        "tokens": pieces,
        "source": "sampler",
        "assistant_anchor_present": bool(anchor) if anchor is not None else None,
        "risk_assessment": risk,
        "token_classification": cls,
        **_payload_preflight(trace_preflight, chat_format),
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def emit_sampler_ground_truth_complete(
    llama: Any,
    *,
    gt_token_ids: list[int],
    prompt_tokens: list[int],
    live_trace: Optional["LiveStreamTokenTrace"],
    assistant_full_text: str,
    trace_preflight: dict[str, Any],
    chat_format: str,
) -> None:
    """Full sampler trace + divergence vs live-stream tokenization and post-hoc."""
    if not token_trace_enabled():
        return

    cap = token_trace_n()
    gt_ids = gt_token_ids[:cap]
    gt_texts = detokenize_sampler_token_ids(llama, prompt_tokens, gt_ids)
    gt_joined = "".join(gt_texts)

    live_ids = (
        (live_trace.token_ids[:cap] if live_trace is not None else []) or []
    )
    ph_ids, ph_pieces, ph_cum = extract_first_n_generation_tokens(
        llama, assistant_full_text, cap
    )

    live_score = sequence_prefix_match_score(live_ids, gt_ids)
    posthoc_score = sequence_prefix_match_score(ph_ids, gt_ids)
    ln = min(len(live_ids), len(gt_ids))
    pn = min(len(ph_ids), len(gt_ids))
    live_mismatch = bool(gt_ids) and (
        ln == 0 or live_ids[:ln] != gt_ids[:ln]
    )
    posthoc_mismatch = bool(gt_ids) and (pn == 0 or ph_ids[:pn] != gt_ids[:pn])

    cls = classify_first_token_piece(gt_texts[0] if gt_texts else "")
    risk = risk_from_classification(cls)
    anchor = trace_preflight.get("assistant_anchor_present")

    payload: dict[str, Any] = {
        "event": "llm_token_trace_ground_truth",
        "phase": "complete",
        "source": "sampler",
        "ground_truth_authority": "sampler",
        "trace_n": cap,
        "token_ids": gt_ids,
        "tokens": gt_texts,
        "tokens_joined": gt_joined,
        "live_trace_token_ids": live_ids,
        "post_hoc_token_ids": ph_ids,
        "live_match_score": live_score,
        "posthoc_match_score": posthoc_score,
        "live_vs_ground_truth_mismatch": live_mismatch,
        "posthoc_vs_ground_truth_mismatch": posthoc_mismatch,
        "token_classification": cls,
        "risk_assessment": risk,
        "assistant_anchor_present": bool(anchor) if anchor is not None else None,
        **_payload_preflight(trace_preflight, chat_format),
    }
    if live_mismatch or posthoc_mismatch:
        payload["divergence_note"] = (
            "live = delta re-tokenized; post_hoc = full-string tokenize; "
            "ground_truth = sampler token ids from Llama.generate."
        )
    logger.info(json.dumps(payload, ensure_ascii=False))


def emit_posthoc_token_trace_fallback(
    llama: Any,
    *,
    assistant_generated_text: str,
    trace_preflight: Optional[dict[str, Any]],
    chat_format: str,
) -> None:
    """
    Non-streaming or empty-live fallback: single JSON line (event llm_token_trace), post-hoc only.
    """
    if not token_trace_enabled():
        return

    n = token_trace_n()
    ids, pieces, cum = extract_first_n_generation_tokens(llama, assistant_generated_text, n)
    cls = classify_first_token_piece(pieces[0] if pieces else "")
    risk = risk_from_classification(cls)
    pre = trace_preflight or {}
    payload: dict[str, Any] = {
        "event": "llm_token_trace",
        "phase": "complete",
        "source": "post_hoc_fallback",
        "trace_n": n,
        "first_token_ids": ids,
        "first_tokens": pieces,
        "first_tokens_decoded": cum,
        "token_classification": cls,
        "risk_assessment": risk,
        **_payload_preflight(pre, chat_format),
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
