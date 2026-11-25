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

Plan S17 (V20) adds the history side: ``history_token_budget`` gives the tokens
the chat history may use on this turn, and ``fit_history_to_token_budget``
drops the oldest messages until it fits. On a turn with retrieval the window is
split explicitly: the retrieval block keeps up to its cap or half of the room
left after the fixed costs, whichever is smaller, and history gets the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

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
# Plan S17: a message that does not fit whole is kept truncated when at least
# this much room is left; the marker matches the worker's single-message cap.
MIN_PARTIAL_MESSAGE_TOKENS = 128
TRUNCATION_SUFFIX = "\n\n[…message truncated for context window]"

TokenCounter = Callable[[str], int]


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


def message_tokens(message: Mapping, count_tokens: TokenCounter | None = None) -> int:
    """Prompt tokens for one chat message, template tokens included.

    ``count_tokens`` is the loaded model's tokenizer when the caller has one
    (plan S17, native engine); otherwise the S10 estimate.
    """
    count = count_tokens or estimate_tokens
    return count(str(message.get("content") or "")) + PER_MESSAGE_TEMPLATE_TOKENS


def _room_after_fixed_costs(
    ctx: int,
    *,
    extra_context_chars: int,
    skills_chars: int,
    max_output_tokens: int | None,
) -> int:
    """Window minus answer reserve, prompt overhead and margin (before history and retrieval)."""
    overhead_tokens = int(
        math.ceil(
            (PROMPT_OVERHEAD_CHARS + max(0, int(skills_chars)) + max(0, int(extra_context_chars)))
            / CHARS_PER_TOKEN_ESTIMATE
        )
    )
    return ctx - output_reserve_tokens(ctx, max_output_tokens) - overhead_tokens - SAFETY_MARGIN_TOKENS


def _context_size(n_ctx: object) -> int:
    try:
        return int(n_ctx) if n_ctx is not None else 0
    except (TypeError, ValueError):
        return 0


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
    count_tokens: TokenCounter | None = None,
) -> RetrievalBudget:
    """Character budget for the retrieval block on this turn.

    ``count_tokens`` (plan S17) counts history and the question with the loaded
    model's tokenizer; the fixed overhead and the block stay estimates.
    """
    cap = max(MIN_RETRIEVAL_CHARS, int(cap_chars))
    ctx = _context_size(n_ctx)
    if ctx <= 0:
        return RetrievalBudget(chars=cap, tokens=None, n_ctx=None, reason="default")

    messages = list(history or ())
    count = count_tokens or estimate_tokens
    history_tokens = sum(message_tokens(m, count) for m in messages if isinstance(m, Mapping))
    available = (
        _room_after_fixed_costs(
            ctx,
            extra_context_chars=extra_context_chars,
            skills_chars=skills_chars,
            max_output_tokens=max_output_tokens,
        )
        - history_tokens
        - (count(query) if query else 0)
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


def history_token_budget(
    *,
    n_ctx: int | None,
    extra_context_chars: int = 0,
    skills_chars: int = 0,
    max_output_tokens: int | None = None,
    reserve_retrieval: bool = True,
    cap_chars: int = DEFAULT_RETRIEVAL_CHARS,
) -> int | None:
    """Tokens the chat history, current question included, may use on this turn (plan S17).

    The fixed costs are the ones ``retrieval_char_budget`` subtracts. With
    ``reserve_retrieval`` (the turn has sources), the retrieval block keeps
    ``min(its cap, half the room)``, so on large windows history only gives way
    once retrieval would drop below its cap, and on small ones the two split the
    room evenly. Unknown ``n_ctx`` gives ``None``: no token limit, as before S17.
    """
    ctx = _context_size(n_ctx)
    if ctx <= 0:
        return None
    room = max(
        0,
        _room_after_fixed_costs(
            ctx,
            extra_context_chars=extra_context_chars,
            skills_chars=skills_chars,
            max_output_tokens=max_output_tokens,
        ),
    )
    if reserve_retrieval:
        cap_tokens = int(math.ceil(max(MIN_RETRIEVAL_CHARS, int(cap_chars)) / CHARS_PER_TOKEN_ESTIMATE))
        room -= min(cap_tokens, room // 2)
    return room


@dataclass(frozen=True)
class HistoryWindow:
    messages: list
    dropped: int
    tokens: int  # kept messages, by the counter used
    budget_tokens: int | None
    reason: str  # "no_ctx" | "fits" | "trimmed" | "over_budget"
    truncated: bool = False  # the oldest kept message was cut to fit

    def as_trace(self) -> dict:
        return {
            "kept": len(self.messages),
            "dropped": self.dropped,
            "tokens": self.tokens,
            "budget_tokens": self.budget_tokens,
            "reason": self.reason,
            "truncated": self.truncated,
        }


def _role(message: Mapping) -> str:
    return str(message.get("role") or "")


def _truncate_to_tokens(message: Mapping, room: int, count: TokenCounter) -> dict | None:
    """Longest head of ``message`` that, with the truncation marker, fits ``room`` tokens."""
    content = str(message.get("content") or "")

    def cost(n: int) -> int:
        return count(content[:n] + TRUNCATION_SUFFIX) + PER_MESSAGE_TEMPLATE_TOKENS

    if not content or cost(0) > room:
        return None
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= room:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return None
    out = dict(message)
    out["content"] = content[:lo].rstrip() + TRUNCATION_SUFFIX
    return out


def fit_history_to_token_budget(
    history: Iterable[Mapping] | None,
    budget_tokens: int | None,
    *,
    count_tokens: TokenCounter | None = None,
    min_partial_tokens: int = MIN_PARTIAL_MESSAGE_TOKENS,
) -> HistoryWindow:
    """Drop the oldest messages until the history fits ``budget_tokens`` (plan S17).

    Messages are taken newest first. The newest (the current question) is always
    kept, so a question that alone exceeds the budget gives ``over_budget``
    rather than an empty history. When the next message does not fit whole, at
    least ``min_partial_tokens`` are left and it is not an assistant turn, its
    head is kept with the truncation marker, so a small window keeps the last
    exchange instead of dropping it. A leading assistant message is then
    dropped, as in the message-count window (chat templates expect a user turn
    first). Kept messages are the original dicts; only a truncated one is a copy.
    Messages older than the cut are never tokenized.
    """
    count = count_tokens or estimate_tokens
    messages = [m for m in (history or ()) if isinstance(m, Mapping)]
    if budget_tokens is None:
        return HistoryWindow(messages, 0, sum(message_tokens(m, count) for m in messages), None, "no_ctx")

    kept_rev: list = []
    costs_rev: list[int] = []
    used = 0
    i = len(messages) - 1
    while i >= 0:
        cost = message_tokens(messages[i], count)
        if kept_rev and used + cost > budget_tokens:
            break
        kept_rev.append(messages[i])
        costs_rev.append(cost)
        used += cost
        i -= 1
    if i < 0 and used <= budget_tokens:
        return HistoryWindow(messages, 0, used, budget_tokens, "fits")

    kept = kept_rev[::-1]
    costs = costs_rev[::-1]
    truncated = False
    if i >= 0 and _role(messages[i]) != "assistant":
        room = budget_tokens - used
        if room >= min_partial_tokens:
            part = _truncate_to_tokens(messages[i], room, count)
            if part is not None:
                kept.insert(0, part)
                costs.insert(0, message_tokens(part, count))
                truncated = True
    while len(kept) > 1 and _role(kept[0]) == "assistant":
        kept.pop(0)
        costs.pop(0)
    used = sum(costs)
    reason = "trimmed" if used <= budget_tokens else "over_budget"
    return HistoryWindow(kept, len(messages) - len(kept), used, budget_tokens, reason, truncated)


__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "DEFAULT_RETRIEVAL_CHARS",
    "MIN_PARTIAL_MESSAGE_TOKENS",
    "MIN_RETRIEVAL_CHARS",
    "PROMPT_OVERHEAD_CHARS",
    "TRUNCATION_SUFFIX",
    "HistoryWindow",
    "TokenCounter",
    "RetrievalBudget",
    "estimate_tokens",
    "fit_history_to_token_budget",
    "history_token_budget",
    "message_tokens",
    "output_reserve_tokens",
    "retrieval_char_budget",
]
