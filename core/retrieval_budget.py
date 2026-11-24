"""Token-aware character budget for the retrieval block in the prompt (plan S10, R5).

The worker formats retrieved sources into one numbered block before the system
prompt is assembled. Until S10 that block was capped at a fixed 8000 characters
whatever the model's context size, so a 4k-context model could receive a prompt
larger than its window (llama.cpp then refuses the request).

``retrieval_char_budget`` derives the cap from the active ``n_ctx``:

    n_ctx − output reserve − prompt overhead − history − question − margin

and converts the remaining tokens to characters. When ``n_ctx`` is unknown the
budget is today's 8000 characters. The budget never exceeds ``cap_chars``
(8000 by default), so larger contexts keep the pre-S10 behaviour; only tight
contexts shrink.

Token counts are estimates (no tokenizer is loaded here). The S10 measurement
on five local GGUF tokenizers (Qwen3, Qwen3.5, Qwen2, Phi-4-mini, Gemma 4)
gave 3.98–5.32 characters per token on English sources, system prompts and
chat history, so ASCII text is estimated at the measured floor of 4.0
characters per token (3.5 shrank 4k prompts that fitted by about a third).
Non-ASCII characters (CJK, accented scripts) are counted as one token each,
and the worker re-fits the block when its own estimate exceeds the budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

DEFAULT_RETRIEVAL_CHARS = 8000
CHARS_PER_TOKEN_ESTIMATE = 4.0
# Smallest block still worth sending (about one source excerpt).
MIN_RETRIEVAL_CHARS = 1000
# Persona + suffixes (measured max ~2900 chars), the user-turn retrieval wrapper
# (~800 chars), and preference / reply-shape / topic hints. Skills add their own
# configurable budget on top (``skills_chars``).
PROMPT_OVERHEAD_CHARS = 4800
# Chat-template tokens per message and general slack for the estimate.
PER_MESSAGE_TEMPLATE_TOKENS = 8
SAFETY_MARGIN_TOKENS = 128
MIN_OUTPUT_RESERVE_TOKENS = 512


def estimate_tokens(text: str) -> int:
    """Token estimate: ASCII at 4 chars/token (measured floor), other chars 1 each."""
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    return int(math.ceil(ascii_chars / CHARS_PER_TOKEN_ESTIMATE + non_ascii))


def output_reserve_tokens(n_ctx: int, max_output_tokens: int | None) -> int:
    """Tokens kept free for the answer.

    The native engine re-clamps ``max_tokens`` to what is left after the prompt,
    so reserving the whole completion budget would starve retrieval on small
    contexts. A quarter of the window (at least 512, at most the completion
    budget) leaves room for a full answer at the usual sizes.
    """
    quarter = max(MIN_OUTPUT_RESERVE_TOKENS, int(n_ctx) // 4)
    if max_output_tokens is not None and int(max_output_tokens) > 0:
        return max(MIN_OUTPUT_RESERVE_TOKENS, min(int(max_output_tokens), quarter))
    return quarter


@dataclass(frozen=True)
class RetrievalBudget:
    chars: int
    tokens: int | None  # room left for the block in tokens (None: default / floor)
    n_ctx: int | None
    reason: str  # "default" | "cap" | "n_ctx" | "floor"

    def as_trace(self) -> dict:
        return {
            "chars": self.chars,
            "tokens": self.tokens,
            "n_ctx": self.n_ctx,
            "reason": self.reason,
        }


def retrieval_char_budget(
    *,
    n_ctx: int | None,
    history: Iterable[Mapping] | None = None,
    query: str = "",
    extra_context_chars: int = 0,
    skills_chars: int = 0,
    max_output_tokens: int | None = None,
    cap_chars: int = DEFAULT_RETRIEVAL_CHARS,
) -> RetrievalBudget:
    """Character budget for the retrieval block on this turn."""
    cap = max(MIN_RETRIEVAL_CHARS, int(cap_chars))
    try:
        ctx = int(n_ctx) if n_ctx is not None else 0
    except (TypeError, ValueError):
        ctx = 0
    if ctx <= 0:
        return RetrievalBudget(chars=cap, tokens=None, n_ctx=None, reason="default")

    messages = list(history or ())
    history_tokens = sum(
        estimate_tokens(str(m.get("content") or "")) + PER_MESSAGE_TEMPLATE_TOKENS
        for m in messages
        if isinstance(m, Mapping)
    )
    overhead_tokens = int(
        math.ceil(
            (PROMPT_OVERHEAD_CHARS + max(0, int(skills_chars)) + max(0, int(extra_context_chars)))
            / CHARS_PER_TOKEN_ESTIMATE
        )
    )
    available = (
        ctx
        - output_reserve_tokens(ctx, max_output_tokens)
        - overhead_tokens
        - history_tokens
        - estimate_tokens(query or "")
        - SAFETY_MARGIN_TOKENS
    )
    # ``tokens`` is the room actually left for the block, so a caller re-fitting a
    # token-dense (non-ASCII) block shrinks it only when the window needs it. At
    # the floor there is no room to protect, so no token limit is given.
    chars = int(max(0, available) * CHARS_PER_TOKEN_ESTIMATE)
    if chars >= cap:
        return RetrievalBudget(chars=cap, tokens=available, n_ctx=ctx, reason="cap")
    if chars < MIN_RETRIEVAL_CHARS:
        return RetrievalBudget(chars=MIN_RETRIEVAL_CHARS, tokens=None, n_ctx=ctx, reason="floor")
    return RetrievalBudget(chars=chars, tokens=available, n_ctx=ctx, reason="n_ctx")


__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "DEFAULT_RETRIEVAL_CHARS",
    "MIN_RETRIEVAL_CHARS",
    "PROMPT_OVERHEAD_CHARS",
    "RetrievalBudget",
    "estimate_tokens",
    "output_reserve_tokens",
    "retrieval_char_budget",
]
