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

from typing import Optional


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
    ) -> None:
        self._buffer: str = ""
        self._tail_chars: int = int(tail_chars)
        self._min_repeats: int = int(min_repeats)
        self._max_atom_chars: int = int(max_atom_chars)
        self._min_atom_chars: int = int(min_atom_chars)
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
