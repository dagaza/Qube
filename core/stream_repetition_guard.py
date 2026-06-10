"""
Streaming degeneration guard for LLM output.

Some local models (especially when the system prompt pushes a short fallback
token like a citation tag, e.g. ``[W]`` for web sources) can lock into a
repeat-loop and emit the same short atom indefinitely until the wall-time
cap ends the stream. This guard watches the tail of the live stream and
signals when the output has collapsed into degenerate repetition so the
worker can cancel the generation instead of waiting minutes.

Observer-only: it never mutates tokens and never modifies what the user
sees while the stream is still making real progress.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.generation_risk_profile import GenerationRiskProfile

_PUNCT_RUN = re.compile(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{24,}")
_PUNCT_CHARS_RE = re.compile(r"[.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]")
_SPACED_PUNCT_TAIL_RE = re.compile(
    r"[\s\u00a0\u202f\u2009\u200a\u2028.…?\u2026\u2010\u2011\u2012\u2013\u2014\-]{28,}"
)
_BARE_NUMBERED_LINE = re.compile(r"^\s*(\d+)\.\s*(.*)$", re.M)


def _numbered_list_loop_degenerate(tail: str) -> bool:
    """Detect runaway or empty numbered-list generation in the stream tail."""
    if not tail:
        return False
    window = tail[-500:]
    bare_run: list[int] = []
    item_run: list[str] = []
    for line in window.splitlines():
        m = _BARE_NUMBERED_LINE.match(line)
        if not m:
            if len(bare_run) >= 4:
                recent = bare_run[-4:]
                if recent == list(range(recent[0], recent[0] + 4)):
                    return True
            bare_run.clear()
            item_run.clear()
            continue
        num = int(m.group(1))
        item = (m.group(2) or "").strip()
        bare_run.append(num)
        item_run.append(item)
        if len(bare_run) >= 4:
            recent_nums = bare_run[-4:]
            if recent_nums == list(range(recent_nums[0], recent_nums[0] + 4)):
                recent_items = item_run[-4:]
                if all(len(x) <= 2 for x in recent_items):
                    return True
        if len(item_run) >= 4:
            recent_items = item_run[-4:]
            if all(len(x) <= 2 for x in recent_items):
                return True
            if len(set(recent_items)) == 1 and recent_items[0]:
                return True
    if len(bare_run) >= 4:
        recent = bare_run[-4:]
        if recent == list(range(recent[0], recent[0] + 4)):
            return True
    return False


def create_stream_repetition_guard(
    profile: "GenerationRiskProfile | None" = None,
    *,
    min_repeats: int | None = None,
    tail_chars: int | None = None,
    enable_list_loop_guard: bool | None = None,
) -> "StreamRepetitionGuard":
    """Build a guard tuned to the turn's generation risk profile."""
    kwargs: dict = {}
    if profile is not None:
        kwargs["min_repeats"] = profile.stream_guard_min_repeats
        kwargs["tail_chars"] = profile.stream_guard_tail_chars
        kwargs["enable_list_loop_guard"] = profile.enable_list_loop_guard
    if min_repeats is not None:
        kwargs["min_repeats"] = min_repeats
    if tail_chars is not None:
        kwargs["tail_chars"] = tail_chars
    if enable_list_loop_guard is not None:
        kwargs["enable_list_loop_guard"] = enable_list_loop_guard
    return StreamRepetitionGuard(**kwargs)


class StreamRepetitionGuard:
    """Detects runaway token-level repetition in streaming LLM output.

    Heuristic: tokenize the recent tail of the stream by whitespace and
    look at the last ``min_repeats`` "atoms". If every one of them is the
    same short string (length ``<= max_atom_chars``), the stream is almost
    certainly in a degenerate loop (``[W] [W] [W] ...``, ``lol lol lol ...``,
    ``[1] [1] [1] ...``). Legitimate prose does not produce that many
    consecutive identical whitespace-separated tokens.

    The guard is fed every streamed delta via :meth:`observe`; once it trips
    it stays tripped (idempotent) and the caller should break out of the
    stream loop and cancel the underlying generation.
    """

    __slots__ = (
        "_buffer",
        "_tail_chars",
        "_min_repeats",
        "_max_atom_chars",
        "_min_atom_chars",
        "_enable_list_loop_guard",
        "_tripped",
        "_trip_reason",
    )

    def __init__(
        self,
        *,
        min_repeats: int = 10,
        max_atom_chars: int = 12,
        min_atom_chars: int = 1,
        tail_chars: int = 600,
        enable_list_loop_guard: bool = False,
    ) -> None:
        self._buffer: str = ""
        self._tail_chars: int = int(tail_chars)
        self._min_repeats: int = int(min_repeats)
        self._max_atom_chars: int = int(max_atom_chars)
        self._min_atom_chars: int = int(min_atom_chars)
        self._enable_list_loop_guard: bool = bool(enable_list_loop_guard)
        self._tripped: bool = False
        self._trip_reason: Optional[str] = None

    def observe(self, delta: str) -> bool:
        """Feed a streamed delta; return True if degeneration was detected.

        Once tripped, further calls always return True without re-scanning.
        """
        if self._tripped:
            return True
        if not delta:
            return False

        self._buffer += delta
        if len(self._buffer) > self._tail_chars:
            self._buffer = self._buffer[-self._tail_chars:]

        if _PUNCT_RUN.search(self._buffer):
            self._tripped = True
            self._trip_reason = "punctuation run degeneration in stream tail"
            return True
        spaced_tail = _SPACED_PUNCT_TAIL_RE.search(self._buffer)
        if spaced_tail is not None and len(_PUNCT_CHARS_RE.findall(spaced_tail.group(0))) >= 12:
            self._tripped = True
            self._trip_reason = "spaced punctuation degeneration in stream tail"
            return True

        if self._enable_list_loop_guard and _numbered_list_loop_degenerate(self._buffer):
            self._tripped = True
            self._trip_reason = "numbered list loop degeneration in stream tail"
            return True

        atoms = self._buffer.split()
        if len(atoms) < self._min_repeats:
            return False

        tail = atoms[-self._min_repeats:]
        first = tail[0]
        n = len(first)
        if n < self._min_atom_chars or n > self._max_atom_chars:
            return False

        for other in tail[1:]:
            if other != first:
                return False

        self._tripped = True
        self._trip_reason = (
            f"repeated atom {first!r} x{self._min_repeats} in stream tail"
        )
        return True

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> Optional[str]:
        return self._trip_reason
