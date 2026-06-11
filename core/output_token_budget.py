"""Resolve per-turn output token budgets and detect probable max-token truncation."""
from __future__ import annotations

from core.output_degeneration import detect_output_degeneration

ABSOLUTE_MIN_OUTPUT_TOKENS = 256
PROMPT_RESERVE_WHEN_UNKNOWN = 512
PROMPT_MARGIN_WHEN_KNOWN = 64
DEFAULT_OUTPUT_TOKEN_LIMIT = 4096


def resolve_output_token_budget(
    *,
    context_window: int,
    limit_enabled: bool,
    user_limit: int,
    prompt_tokens: int | None = None,
) -> int:
    """
    Compute max_tokens for create_chat_completion / external APIs.

    When ``limit_enabled`` is False, the budget is all remaining context after the
    prompt (minus a small margin). When True, cap at ``user_limit`` as well.
    """
    ctx = max(512, int(context_window))
    if prompt_tokens is not None:
        remaining = max(
            ABSOLUTE_MIN_OUTPUT_TOKENS,
            ctx - int(prompt_tokens) - PROMPT_MARGIN_WHEN_KNOWN,
        )
    else:
        remaining = max(
            ABSOLUTE_MIN_OUTPUT_TOKENS,
            ctx - PROMPT_RESERVE_WHEN_UNKNOWN,
        )

    if not limit_enabled:
        return remaining

    cap = max(
        ABSOLUTE_MIN_OUTPUT_TOKENS,
        min(int(user_limit), ctx - ABSOLUTE_MIN_OUTPUT_TOKENS),
    )
    return min(cap, remaining)


def clamp_max_tokens_to_context(
    *,
    n_ctx: int,
    prompt_token_count: int,
    requested_max_tokens: int,
    limit_enabled: bool,
) -> int:
    """Final clamp once the prompt token count is known (native engine path)."""
    return resolve_output_token_budget(
        context_window=n_ctx,
        limit_enabled=limit_enabled,
        user_limit=requested_max_tokens,
        prompt_tokens=prompt_token_count,
    )


def describe_output_token_budget(
    *,
    context_window: int,
    limit_enabled: bool,
    user_limit: int,
    chat_history_messages: int | None = None,
) -> str:
    """Human-readable hint for settings UI."""
    ctx = max(512, int(context_window))
    history_clause = ""
    if chat_history_messages is not None:
        n = max(2, int(chat_history_messages))
        history_clause = (
            f" Chat history (toolbar, currently {n} messages) and retrieved "
            f"documents also count toward the prompt."
        )
    shared_note = (
        f"The {ctx:,}-token context window is shared: prompt first, then reply."
        f"{history_clause}"
    )
    if not limit_enabled:
        typical = resolve_output_token_budget(
            context_window=ctx,
            limit_enabled=False,
            user_limit=user_limit,
        )
        return (
            f"Reply length is not capped separately — the model may use whatever "
            f"tokens remain after the prompt (often up to ~{typical:,} with this "
            f"context size, but less when the prompt is large). {shared_note}"
        )
    effective = resolve_output_token_budget(
        context_window=ctx,
        limit_enabled=True,
        user_limit=user_limit,
    )
    return (
        f"Each reply may generate up to {effective:,} tokens (your cap is "
        f"{int(user_limit):,}). Actual room can be lower if the prompt is large. "
        f"{shared_note}"
    )


def probable_max_tokens_truncation(
    text: str,
    *,
    stream_finish_reason: str = "",
    max_tokens: int,
    limit_enabled: bool,
    completion_token_count: int | None = None,
) -> str | None:
    """
    Return a notice reason when output likely hit a token ceiling.

    ``finish_reason == length`` is handled by callers; this covers backends that
    omit that signal while still stopping at max_tokens.
    """
    reason = (stream_finish_reason or "").strip().lower()
    if reason == "length":
        return "finish_reason_length"

    body = (text or "").strip()
    if not body:
        return None

    if not limit_enabled and completion_token_count is None:
        return None

    near_cap = False
    if completion_token_count is not None and max_tokens > 0:
        near_cap = int(completion_token_count) >= int(max_tokens * 0.92)
    elif limit_enabled and max_tokens > 0:
        # Rough char heuristic when exact completion count is unavailable.
        near_cap = len(body) >= int(max_tokens * 2.5)

    if not near_cap:
        return None

    scored = detect_output_degeneration(body)
    trunc = float(scored.components.truncation)
    bullet = float(scored.components.unfinished_bullet)
    if max(trunc, bullet) >= 0.65:
        return "heuristic_truncated_output"
    return None
