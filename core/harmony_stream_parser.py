"""
Streaming Harmony output parser — emits only stable final-channel text for UI/TTS.

Handles partial control tokens across chunk boundaries. Non-final channels are buffered
for optional diagnostics, not user-visible emission.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from core.harmony_degeneration import find_degeneration_start

HarmonyChannel = Literal["final", "analysis", "commentary", "other", "unknown"]

# Longest match first so ``<|channel|>final`` wins over ``<|channel|>``.
_CONTROL_TOKENS: tuple[tuple[str, str], ...] = (
    ("<|end|><|start|>assistant<|channel|>final<|message|>", "bridge_final"),
    ("<|start|>assistant<|channel|>final<|message|>", "open_final"),
    ("<|channel|>analysis", "channel_analysis"),
    ("<|channel|>commentary", "channel_commentary"),
    ("<|channel|>final", "channel_final"),
    ("<|start|>assistant", "role_assistant"),
    ("<|start|>user", "role_user"),
    ("<|start|>system", "role_system"),
    ("<|return|>", "return"),
    ("<|message|>", "message"),
    ("<|end|>", "end"),
    ("<|start|>", "start"),
)

_MALFORMED_CHANNEL = re.compile(
    r"(?i)<\|?channel\|?>|<\|?message\|?>|<\|?start\|?>|<\|?final\|?>|<\|?end\|?>"
)

# Untagged planning often appears before the first <|channel|> marker in leaked completions.
_UNTAGGED_PLANNING = re.compile(
    r"(?is)^\s*(?:let'?s\s+clarify|we\s+need\s+to|we\s+should|we\s+have|the\s+user\s+says)\b"
)

# Mid-stream scratchpad / meta (final channel leaks after a real answer clause).
_SCRATCHPAD_BOUNDARY = re.compile(
    r"(?is)"
    r"(?:\n|^|[.!?…]\s{0,3})"
    r"\s*(?:"
    r"we\s+(?:need|have|should|must|are|'re)\b|"
    r"the\s+(?:question|user)\s+(?:says|wants)\b|"
    r"they\s+may\s+be\s+asking\b|"
    r"no\s+meta\s+commentary\b|"
    r"provide\s+(?:final\s+)?answer\b"
    r")"
)

_WE_PUNCT_LOOP = re.compile(
    r"(?is)\bwe\s*(?:\.\.\.|…|[.\?\‐‑–—\-]){2,}"
)


@dataclass
class HarmonyStreamParser:
    """
    Incremental parser for Harmony completion deltas.

    ``assume_final_channel=True`` matches prompts that pre-fill the final channel
    (Qube's ``render_harmony_final_prompt`` path).
    """

    assume_final_channel: bool = True
    _channel: HarmonyChannel = "unknown"
    _pending: str = ""
    _diagnostic_parts: list[str] = field(default_factory=list)
    _saw_channel_tag: bool = False
    _final_muted: bool = False
    _raw_seen: str = ""
    _degeneration_cut: Optional[int] = None

    def __post_init__(self) -> None:
        if self.assume_final_channel:
            self._channel = "final"

    @property
    def current_channel(self) -> HarmonyChannel:
        return self._channel

    @property
    def diagnostic_text(self) -> str:
        return "".join(self._diagnostic_parts)

    @property
    def final_muted(self) -> bool:
        """True after scratchpad/meta was detected in the final channel."""
        return self._final_muted

    @property
    def raw_seen(self) -> str:
        """All completion deltas observed (including non-emitted planning)."""
        return self._raw_seen

    @property
    def degeneration_cut(self) -> Optional[int]:
        """Index in ``raw_seen`` where degeneration started, if detected."""
        return self._degeneration_cut

    @property
    def degeneration_detected(self) -> bool:
        return self._degeneration_cut is not None

    def _note_degeneration_cut(self) -> None:
        cut = find_degeneration_start(self._raw_seen)
        if cut is not None:
            self._degeneration_cut = cut

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        offset = len(self._raw_seen)
        self._raw_seen += chunk
        self._note_degeneration_cut()
        if self._degeneration_cut is not None:
            cut = self._degeneration_cut
            if cut <= offset:
                self._final_muted = True
                return ""
            if cut < offset + len(chunk):
                chunk = chunk[: cut - offset]
        if not chunk:
            return ""
        self._pending += chunk
        out = self._drain(emit=True)
        if self._degeneration_cut is not None:
            self._final_muted = True
        return out

    def flush(self) -> str:
        if not self._pending:
            return ""
        # Drop trailing partial control token (incomplete ``<|...``).
        if "<|" in self._pending:
            idx = self._pending.rfind("<|")
            tail = self._pending[idx:]
            if "|>" not in tail and not any(
                tail.startswith(tok) for tok, _ in _CONTROL_TOKENS
            ):
                self._pending = self._pending[:idx]
        out = self._emit_visible(self._pending)
        self._pending = ""
        return out

    def _hold_incomplete_control_suffix(self) -> tuple[str, str]:
        """Split ``pending`` so a trailing partial ``<|...`` token stays buffered."""
        idx = self._pending.rfind("<|")
        if idx < 0:
            return self._pending, ""
        tail = self._pending[idx:]
        if any(len(tail) < len(tok) and tok.startswith(tail) for tok, _ in _CONTROL_TOKENS):
            return self._pending[:idx], tail
        if "|>" in tail and not any(tail == tok for tok, _ in _CONTROL_TOKENS):
            # Malformed complete-ish token — keep buffering until flush.
            return self._pending[:idx], tail
        if any(tail.startswith(tok[: len(tail)]) for tok, _ in _CONTROL_TOKENS if tok.startswith("<|")):
            return self._pending[:idx], tail
        return self._pending, ""

    def _drain(self, *, emit: bool) -> str:
        out: list[str] = []
        while self._pending:
            match = self._find_earliest_token(self._pending)
            if match is None:
                safe, hold = self._hold_incomplete_control_suffix()
                if hold:
                    if emit and self._channel == "final":
                        out.append(self._emit_visible(safe))
                    elif safe and self._channel != "final":
                        self._diagnostic_parts.append(safe)
                    self._pending = hold
                    break
                if emit and self._channel == "final":
                    out.append(self._emit_visible(self._pending))
                elif self._channel != "final":
                    self._diagnostic_parts.append(self._pending)
                self._pending = ""
                break

            pos, tok, kind = match
            if pos > 0:
                prefix = self._pending[:pos]
                if emit and self._channel == "final":
                    out.append(self._emit_visible(prefix))
                elif self._channel != "final":
                    self._diagnostic_parts.append(prefix)
                self._pending = self._pending[pos:]

            self._apply_token(kind)
            self._pending = self._pending[len(tok) :]

        return "".join(out)

    def _find_earliest_token(self, text: str) -> tuple[int, str, str] | None:
        best: tuple[int, str, str] | None = None
        for tok, kind in _CONTROL_TOKENS:
            pos = text.find(tok)
            if pos < 0:
                continue
            if best is None or pos < best[0] or (pos == best[0] and len(tok) > len(best[1])):
                best = (pos, tok, kind)
        return best

    def _apply_token(self, kind: str) -> None:
        self._saw_channel_tag = True
        if kind in ("bridge_final", "open_final", "channel_final"):
            self._channel = "final"
            self._final_muted = False
        elif kind == "channel_analysis":
            self._channel = "analysis"
            self._final_muted = True
        elif kind == "channel_commentary":
            self._channel = "commentary"
            self._final_muted = True
        elif kind in ("return", "end"):
            self._final_muted = True
        elif kind.startswith("role_") or kind == "channel_unknown":
            self._channel = "other"
            self._final_muted = True

    def _emit_visible(self, text: str) -> str:
        if not text or self._channel != "final":
            if text and self._channel != "final":
                self._diagnostic_parts.append(text)
            return ""
        if self._final_muted:
            self._diagnostic_parts.append(text)
            return ""
        we_loop = _WE_PUNCT_LOOP.search(text)
        if we_loop:
            head = text[: we_loop.start()]
            self._diagnostic_parts.append(text[we_loop.start() :])
            self._final_muted = True
            text = head
        if not self._saw_channel_tag and _UNTAGGED_PLANNING.search(text):
            self._diagnostic_parts.append(text)
            return ""
        cut = find_degeneration_start(text)
        if cut is not None:
            head = text[:cut]
            self._diagnostic_parts.append(text[cut:])
            self._final_muted = True
            text = head
        boundary = _SCRATCHPAD_BOUNDARY.search(text)
        if boundary:
            head = text[: boundary.start()]
            tail = text[boundary.start() :]
            self._diagnostic_parts.append(tail)
            self._final_muted = True
            text = head
        cut = re.search(r"<\|?\s*chan", text, re.I)
        if cut:
            text = text[: cut.start()]
        cleaned = _MALFORMED_CHANNEL.sub("", text)
        if "\n" in cleaned:
            lines = cleaned.split("\n")
            kept: list[str] = [lines[0]]
            for line in lines[1:]:
                if _MALFORMED_CHANNEL.search(line) or re.search(
                    r"<\|?\s*chan|channel\|?>", line, re.I
                ):
                    break
                kept.append(line)
            cleaned = "\n".join(kept)
            if text.endswith("\n") and not cleaned.endswith("\n"):
                cleaned += "\n"
        return cleaned
