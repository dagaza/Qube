"""Pure helpers for composer @-mention arming (modifier-release activation)."""

from __future__ import annotations


def is_valid_mention_anchor(text_before_cursor: str) -> bool:
    """True when ``@`` may be inserted (word boundary or extending an ``@`` run)."""
    if not text_before_cursor:
        return True
    if text_before_cursor[-1].isspace():
        return True
    # Extend a literal @ run (e.g. @@@@ then one more @ opens the menu).
    prefix = text_before_cursor.rstrip("@")
    return prefix == "" or prefix[-1].isspace()


def at_run_bounds(text: str, arm_at: int) -> tuple[int, int]:
    """Half-open interval ``[start, end)`` spanning consecutive ``@`` at ``arm_at``."""
    if arm_at < 0 or arm_at >= len(text) or text[arm_at] != "@":
        return arm_at, arm_at
    start = arm_at
    while start > 0 and text[start - 1] == "@":
        start -= 1
    end = start
    while end < len(text) and text[end] == "@":
        end += 1
    return start, end


def at_run_length(text: str, arm_at: int) -> int:
    start, end = at_run_bounds(text, arm_at)
    return max(0, end - start)


def resolve_mention_release(arm_count: int) -> str:
    """Return ``menu``, ``escape``, or ``invalid`` after modifier release.

    Exactly one ``@`` typed before release opens the picker, regardless of any
    literal ``@`` characters already in the composer. Two or more ``@`` keystrokes
    before release strip one trailing ``@`` (escape).
    """
    if arm_count <= 0:
        return "invalid"
    if arm_count == 1:
        return "menu"
    return "escape"


def escape_strip_index(text: str, arm_at: int) -> int:
    """Index of the trailing ``@`` to remove when escaping."""
    _, end = at_run_bounds(text, arm_at)
    if at_run_length(text, arm_at) < 2:
        return -1
    return end - 1


def menu_trigger_strip_index(text: str, arm_at: int) -> int:
    """Index of the lone ``@`` removed when opening the picker."""
    if arm_at < 0 or arm_at >= len(text) or text[arm_at] != "@":
        return -1
    return arm_at


def mention_query_suffix(text: str, arm_at: int) -> str:
    """Letters typed after the armed ``@``, before a space."""
    pos = arm_at + 1
    while pos < len(text) and text[pos] not in (" ", "\n", "@"):
        pos += 1
    return text[arm_at + 1 : pos]
